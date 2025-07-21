# Simplified two-phase file processor for perfmon3.py
# Phase 1: CPU-only data preparation (steepest fall detection + filtering)
# Phase 2: GPU batch processing (statistics calculation)

import os
import pandas as pd
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from modules.hardware_detector import get_hardware_detector

def detect_time_column(perfmon_data):
    """
    Detect the time column in the CSV data.
    Handles any timezone that relog.exe might generate (GMT, PST, EST, etc.) 
    with both Standard Time and Daylight/Summer Time variations.
    """
    # Processing strategy: CPU-based (data column analysis)
    print("Processing strategy: CPU-based time column detection")
    
    for column in perfmon_data.columns:
        # Look for PDH-CSV 4.0 format with any timezone and time offset
        if (column.startswith('(PDH-CSV 4.0) (') and 
            'Time)(' in column and 
            column.endswith(')')):
            return column
    
    # Fallback - return the first column if no PDH-CSV time column is found
    if len(perfmon_data.columns) > 0:
        return perfmon_data.columns[0]
    
    raise ValueError("No time column found in the CSV data")

def process_batch_with_gpu(filtered_file_data: List[Dict], metric_names: List[str]) -> List[pd.DataFrame]:
    """
    Phase 2: GPU batch processing.
    Process all filtered data through GPU acceleration in batches.
    """
    if not filtered_file_data:
        return []
    
    # Processing strategy: GPU batch processing
    print(f"Phase 2 - GPU batch processing: {len(filtered_file_data)} files, {len(metric_names)} metrics")
    
    all_statistics_list = []
    batch_start_time = pd.Timestamp.now()
    
    # Process each file's filtered data
    for file_data in filtered_file_data:
        if file_data is None:
            continue
            
        file_path = file_data['file_path']
        filtered_perfmon_data = file_data['filtered_data']
        time_column = file_data['time_column']
        steepest_fall_time = file_data['steepest_fall_time']
        file_date_time = file_data['file_date_time']
        start_time = file_data['start_time']
        
        print(f"GPU processing: {os.path.basename(file_path)} - {len(filtered_perfmon_data)} rows")
        
        # Pre-filter data per metric for GPU batch processing
        metric_specific_data = {}
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            if metric_columns:
                columns_to_keep = [time_column] + metric_columns
                metric_specific_data[metric_name] = filtered_perfmon_data[columns_to_keep]
        
        # Process all metrics for this file using GPU
        for metric_name, metric_data in metric_specific_data.items():
            try:
                # Import calculate_statistics here to avoid circular imports
                from modules.calculate_statistics import calculate_statistics
                from shared.modules.ensure_consistent_structure import ensure_consistent_structure
                
                # Calculate statistics using GPU acceleration
                statistics_df = calculate_statistics(
                    metric_data, 
                    metric_name, 
                    file_date_time, 
                    start_time, 
                    steepest_fall_time.strftime('%H:%M')
                )
                
                # Ensure consistent structure
                statistics_df = ensure_consistent_structure(statistics_df)
                
                if not statistics_df.empty:
                    all_statistics_list.append(statistics_df)
                    
            except Exception as e:
                print(f"Error in GPU processing for metric {metric_name}: {e}")
        
        # Clear processed data
        del metric_specific_data
        gc.collect()
    
    batch_end_time = pd.Timestamp.now()
    batch_duration = (batch_end_time - batch_start_time).total_seconds()
    print(f"Phase 2 GPU processing completed in {batch_duration:.2f} seconds")
    
    return all_statistics_list

def process_single_file(args):
    """
    Phase 1: CPU-only data preparation.
    Find steepest fall and filter perfmon data - no GPU operations.
    """
    file_path, baseline_metric_name = args
    
    try:
        # Processing strategy: CPU-only data preparation
        print(f"Phase 1 - Data prep: {os.path.basename(file_path)}")
        
        file_start_time = pd.Timestamp.now()
        
        # Load the full file once to find steepest fall
        perfmon_data = pd.read_csv(file_path, low_memory=False)
        
        if perfmon_data.empty:
            print(f"No data found in file: {file_path}")
            return None
        
        # Detect the actual time column
        time_column = detect_time_column(perfmon_data)
        
        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(perfmon_data[time_column]):
            perfmon_data[time_column] = pd.to_datetime(perfmon_data[time_column])
        
        # Filter the DataFrame to only the needed columns before calling find_steepest_fall
        baseline_columns = [col for col in perfmon_data.columns if baseline_metric_name in col]
        if baseline_columns:
            small_df = perfmon_data[[time_column] + baseline_columns[:1]]
            
            # Import find_steepest_fall here to avoid circular imports
            from modules.find_steepest_fall import find_steepest_fall
            
            # Find the steepest fall for the baseline metric
            steepest_fall_time, steepest_fall_value, column_name = find_steepest_fall(
                small_df, baseline_metric_name, time_column
            )

        if not (steepest_fall_time and steepest_fall_value):
            print(f"No steepest fall found for {baseline_metric_name} in {file_path}")
            del perfmon_data
            return None
        
        # Extract date/time info
        file_date_time = steepest_fall_time.strftime('%d-%b')
        start_time = perfmon_data[time_column].min().strftime('%H:%M')
        
        # Filter perfmon data up to steepest fall time
        filtered_perfmon_data = perfmon_data[perfmon_data[time_column] <= steepest_fall_time]
        
        # Clear original data to free memory
        del perfmon_data
        gc.collect()
        
        file_end_time = pd.Timestamp.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        print(f"Phase 1 complete: {os.path.basename(file_path)} - {len(filtered_perfmon_data)} rows in {file_duration:.2f}s")
        
        # Return structured data for Phase 2
        return {
            'file_path': file_path,
            'filtered_data': filtered_perfmon_data,
            'time_column': time_column,
            'steepest_fall_time': steepest_fall_time,
            'file_date_time': file_date_time,
            'start_time': start_time
        }
        
    except Exception as e:
        print(f"Error in Phase 1 for file {file_path}: {e}")
        return None

def calculate_hardware_aware_workers(csv_file_paths: List[str]) -> Dict[str, int]:
    """
    Calculate optimal number of workers for Phase 1 (CPU-only file processing).
    Simplified for two-phase architecture.
    """
    # Processing strategy: Simplified worker allocation for two-phase architecture
    print("Processing strategy: Simplified worker allocation for two-phase architecture")
    
    if not csv_file_paths:
        return {'file_workers': 1}
    
    # Get hardware detector
    hardware = get_hardware_detector()
    
    # Simple file worker calculation - one worker per file up to CPU limit
    total_cpu_cores = hardware.profile.cpu.cores
    file_workers = min(len(csv_file_paths), max(1, total_cpu_cores // 2))  # Use half cores for file processing
    
    allocation = {
        'file_workers': file_workers,
        'total_cpu_cores': total_cpu_cores
    }
    
    print(f"Phase 1 allocation: {file_workers} file workers (CPU cores: {total_cpu_cores})")
    print(f"Processing {len(csv_file_paths)} files")
    
    return allocation

def file_processor_accelerated(log_directory: str, metric_names: List[str], baseline_metric_name: str, gpu_phase1: bool = False) -> pd.DataFrame:
    """
    Optimized two-phase processor:
    Phase 1: CPU-only OR GPU-accelerated data preparation (steepest fall detection + filtering)
    Phase 2: Batch GPU processing (statistics calculation)
    
    Args:
        gpu_phase1: If True, use GPU acceleration for Phase 1 data preparation
    """
    
    # Processing strategy based on acceleration flags
    if gpu_phase1:
        print("Processing strategy: GPU-accelerated Phase 1 + GPU Phase 2 architecture")
    else:
        print("Processing strategy: Optimized two-phase architecture (CPU prep → GPU batch)")
    
    # Collect all CSV files
    csv_file_paths = []
    for root, dirs, files in os.walk(log_directory):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                csv_file_paths.append(file_path)
    
    if not csv_file_paths:
        print("No CSV files found in the specified directory.")
        return pd.DataFrame()
    
    processing_type = "GPU-accelerated" if gpu_phase1 else "optimized"
    print(f"Found {len(csv_file_paths)} CSV files for {processing_type} processing")
    
    # ==================== PHASE 1: Data preparation ====================
    if gpu_phase1:
        phase1_type = "GPU-Accelerated"
    else:
        phase1_type = "CPU"
    
    print(f"\n--- Phase 1: {phase1_type} Data Preparation ---")
    phase1_start_time = pd.Timestamp.now()
    
    # Calculate optimal workers for Phase 1
    hardware_allocation = calculate_hardware_aware_workers(csv_file_paths)
    
    # Choose processing function based on acceleration flags
    if gpu_phase1:
        from modules.gpu_phase1_processor import process_single_file_gpu_accelerated
        process_function = process_single_file_gpu_accelerated
        print("Using GPU acceleration for Phase 1 data preparation")
    else:
        process_function = process_single_file
        print("Using CPU-only processing for Phase 1 data preparation")
    
    # Prepare arguments for Phase 1
    phase1_args = [
        (file_path, baseline_metric_name)
        for file_path in csv_file_paths
    ]
    
    # Phase 1: Parallel file processing for data preparation
    filtered_file_data = []
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as executor:
        future_to_file = {
            executor.submit(process_function, args): args[0] 
            for args in phase1_args
        }
        
        completed_files = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed_files += 1
            
            try:
                file_result = future.result()
                if file_result is not None:
                    filtered_file_data.append(file_result)
                print(f"Phase 1: {completed_files}/{len(csv_file_paths)} files prepared")
                
            except Exception as e:
                print(f"Phase 1 error for file {file_path}: {e}")
    
    phase1_end_time = pd.Timestamp.now()
    phase1_duration = (phase1_end_time - phase1_start_time).total_seconds()
    print(f"Phase 1 completed: {len(filtered_file_data)} files prepared in {phase1_duration:.2f}s")
    
    # ==================== PHASE 2: GPU batch processing ====================
    print(f"\n--- Phase 2: Optimized GPU Batch Processing ---")
    phase2_start_time = pd.Timestamp.now()
    
    # Phase 2: Single GPU batch processing call (no per-file overhead)
    from modules.enhanced_gpu_processor import process_file_metrics_with_parallel_gpu
    all_statistics_list = process_file_metrics_with_parallel_gpu(filtered_file_data, metric_names)
    
    phase2_end_time = pd.Timestamp.now()
    phase2_duration = (phase2_end_time - phase2_start_time).total_seconds()
    
    # Calculate total processing time
    total_duration = phase1_duration + phase2_duration
    throughput = len(csv_file_paths) / total_duration if total_duration > 0 else 0
    
    if gpu_phase1:
        phase1_label = "GPU prep"
    else:
        phase1_label = "CPU prep"
    
    print(f"\nOptimized processing completed in {total_duration:.2f} seconds")
    print(f"Phase 1 ({phase1_label}): {phase1_duration:.2f}s ({phase1_duration/total_duration*100:.1f}%)")
    print(f"Phase 2 (GPU batch): {phase2_duration:.2f}s ({phase2_duration/total_duration*100:.1f}%)")
    print(f"Throughput: {throughput:.2f} files/second")
    
    if gpu_phase1:
        print(f"GPU Phase 1 acceleration: Enabled")
    else:
        print(f"CPU Phase 1 processing: Standard")
    
    # Combine all results
    if all_statistics_list:
        all_statistics_df = pd.concat(all_statistics_list, axis=0)
        
        # Pivot the DataFrame to have metrics as rows and average/maximum values as columns
        all_statistics_df = all_statistics_df.pivot_table(index='Metric', aggfunc='first')
        
        # Ensure the baseline metric is the first row
        if baseline_metric_name in all_statistics_df.index:
            all_statistics_df = all_statistics_df.reindex(
                [baseline_metric_name] + [idx for idx in all_statistics_df.index if idx != baseline_metric_name]
            )
        
        # Ensure all columns are numeric before rounding
        all_statistics_df = all_statistics_df.apply(pd.to_numeric, errors='coerce')
        
        # Round the data to a specified number of decimal places
        all_statistics_df = all_statistics_df.round(0)
        
        return all_statistics_df
    else:
        print("No statistics data was generated.")
        return pd.DataFrame()

# Alias for backward compatibility
def file_processor(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> pd.DataFrame:
    """Backward-compatible wrapper for the optimized two-phase processor"""
    # Processing strategy: Optimized two-phase architecture (wrapper)
    print("Processing strategy: Optimized two-phase architecture (backward-compatible wrapper)")
    
    return file_processor_accelerated(log_directory, metric_names, baseline_metric_name)
