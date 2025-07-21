# Simplified two-phase file processor for perfmon3.py
# Phase 1: CPU-only data preparation (steepest fall detection + filtering)
# Phase 2: GPU batch processing (statistics calculation)

import os
import pandas as pd
import gc
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    
    # Phase 1: Parallel file processing with TRUE streaming Phase 2 pipeline using ThreadPoolExecutor
    filtered_file_data = []
    all_statistics_list = []
    
    def process_gpu_phase2_immediately(file_result, file_path):
        """Process GPU Phase 2 for a single file immediately - managed by ThreadPoolExecutor"""
        try:
            print(f"🚀 GPU Phase 2 starting immediately for {os.path.basename(file_path)}")
            
            # Process this single file through GPU Phase 2 immediately
            from modules.enhanced_gpu_processor import process_file_metrics_with_parallel_gpu
            single_file_stats = process_file_metrics_with_parallel_gpu([file_result], metric_names)
            
            if single_file_stats:
                print(f"✅ GPU Phase 2 complete for {os.path.basename(file_path)} - {len(single_file_stats)} statistics")
                return single_file_stats
            else:
                print(f"⚠️ No statistics generated for {os.path.basename(file_path)}")
                return []
                
        except Exception as e:
            print(f"❌ GPU Phase 2 error for {file_path}: {e}")
            return []
    
    # Calculate optimal GPU Phase 2 workers based on hardware capabilities
    hardware = get_hardware_detector()
    
    # Determine optimal GPU worker count based on system specs
    if hardware.profile.gpu:
        # Intel Arc GPU with 64 compute units - optimize for parallel GPU Phase 2
        gpu_compute_units = hardware.profile.gpu.compute_units  # 64 for Intel Arc
        gpu_memory_gb = hardware.profile.gpu.memory_gb  # 18GB shared
        
        # Optimal GPU worker count based on performance testing results:
        # - Intel Arc Graphics has 128 execution units (Xe-cores) 
        # - Performance testing showed 32 queues = 134.30/sec (OPTIMAL)
        # - ThreadPoolExecutor optimal: 8-16 threads for Phase 2 processing
        if gpu_compute_units >= 64:  # Intel Arc or equivalent
            optimal_gpu_workers = min(16, max(8, gpu_compute_units // 4))
        elif gpu_compute_units >= 32:  # Mid-range GPU
            optimal_gpu_workers = min(8, max(4, gpu_compute_units // 4))
        else:  # Lower-end GPU
            optimal_gpu_workers = min(4, max(2, gpu_compute_units // 8))
        
        # Adjust for available memory (each worker processes ~1-2GB data)
        memory_limited_workers = max(2, int(gpu_memory_gb / 2))
        optimal_gpu_workers = min(optimal_gpu_workers, memory_limited_workers)
        
        print(f"GPU Hardware: {hardware.profile.gpu.name}")
        print(f"Compute Units: {gpu_compute_units}, Memory: {gpu_memory_gb:.1f}GB")
        print(f"Optimal GPU Phase 2 workers: {optimal_gpu_workers}")
    else:
        optimal_gpu_workers = 4  # Fallback for systems without GPU detection
        print("GPU not detected, using fallback: 4 workers")
    
    # Start Phase 2 GPU processing as soon as first file completes Phase 1
    print("🚀 Starting TRUE streaming pipeline: Phase 1 → Immediate Phase 2 (ThreadPoolExecutor)")
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as process_executor:
        # Use ThreadPoolExecutor for GPU Phase 2 - optimized worker count based on GPU specs
        with ThreadPoolExecutor(max_workers=optimal_gpu_workers, thread_name_prefix="GPU-Phase2") as gpu_executor:
            phase1_futures = {
                process_executor.submit(process_function, args): args[0] 
                for args in phase1_args
            }
            
            gpu_phase2_futures = []  # Track GPU Phase 2 futures
            completed_files = 0
            
            # Process Phase 1 results as they complete and immediately start Phase 2
            for phase1_future in as_completed(phase1_futures):
                file_path = phase1_futures[phase1_future]
                completed_files += 1
                
                try:
                    file_result = phase1_future.result()
                    if file_result is not None:
                        filtered_file_data.append(file_result)
                        
                        print(f"📋 Phase 1: {completed_files}/{len(csv_file_paths)} files prepared → Starting GPU Phase 2 for {os.path.basename(file_path)}")
                        
                        # 🎯 TRUE STREAMING: Submit GPU Phase 2 to ThreadPoolExecutor immediately
                        gpu_future = gpu_executor.submit(process_gpu_phase2_immediately, file_result, file_path)
                        gpu_phase2_futures.append(gpu_future)
                        
                    else:
                        print(f"📋 Phase 1: {completed_files}/{len(csv_file_paths)} files prepared (skipped due to error)")
                    
                except Exception as e:
                    print(f"❌ Phase 1 error for file {file_path}: {e}")
            
            # Wait for all GPU Phase 2 processing to complete and collect results
            print("⏳ Waiting for all GPU Phase 2 threads to complete...")
            for gpu_future in as_completed(gpu_phase2_futures):
                try:
                    gpu_stats = gpu_future.result()
                    if gpu_stats:
                        all_statistics_list.extend(gpu_stats)
                except Exception as e:
                    print(f"❌ Error collecting GPU Phase 2 results: {e}")
    
    phase1_end_time = pd.Timestamp.now()
    phase1_duration = (phase1_end_time - phase1_start_time).total_seconds()
    print(f"🏁 TRUE streaming pipeline completed: {len(filtered_file_data)} files processed in {phase1_duration:.2f}s")
    
    # ==================== PHASE 2: Completed via ThreadPoolExecutor streaming! ====================
    print(f"\n--- Phase 2: Completed via ThreadPoolExecutor streaming pipeline ---")
    phase2_duration = 0.0  # Phase 2 was done during Phase 1 streaming
    
    # Calculate total processing time (streaming approach)
    total_duration = phase1_duration  # Phase 2 completed during Phase 1 via streaming
    throughput = len(csv_file_paths) / total_duration if total_duration > 0 else 0
    
    print(f"\n💯 ThreadPoolExecutor Streaming Pipeline Complete:")
    print(f"   📁 Phase 1 processed: {len(filtered_file_data)} files")
    print(f"   🎯 GPU Phase 2 processed: {len(all_statistics_list)} statistics")
    print(f"   ⚡ Total time: {phase1_duration:.2f}s")
    print(f"   🚀 Throughput: {throughput:.2f} files/second")
    
    if gpu_phase1:
        print(f"   🔥 Architecture: GPU Phase 1 + ThreadPoolExecutor GPU Phase 2")
        architecture_type = "GPU+GPU"
    else:
        print(f"   💻 Architecture: CPU Phase 1 + ThreadPoolExecutor GPU Phase 2")
        architecture_type = "CPU+GPU"
    
    # Calculate and display performance metrics for comparison
    total_metrics_processed = len(all_statistics_list)
    if total_metrics_processed > 0:
        gpu_phase2_efficiency = total_metrics_processed / phase1_duration
        print(f"\n📊 PERFORMANCE METRICS ({architecture_type}):")
        print(f"   🎯 Total Statistics Generated: {total_metrics_processed}")
        print(f"   ⚡ GPU Phase 2 Processing Rate: {gpu_phase2_efficiency:.1f} statistics/second")
        print(f"   🚀 Overall System Throughput: {throughput:.3f} files/second")
        
        # Store performance data for comparison
        performance_data = {
            'architecture': architecture_type,
            'total_time': phase1_duration,
            'files_processed': len(filtered_file_data),
            'statistics_generated': total_metrics_processed,
            'gpu_processing_rate': gpu_phase2_efficiency,
            'overall_throughput': throughput
        }
    
    # Combine all results (already processed via streaming)
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
