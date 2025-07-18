# Hardware-accelerated file processor for perfmon3.py
# Utilizes CPU, GPU, and NPU capabilities for optimal CSV processing performance

import os
import pandas as pd
import psutil
import gc
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from modules.hardware_detector import get_optimal_workers, get_hardware_detector

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

def process_single_metric(args):
    """Process a single metric for a given file with hardware-aware optimization."""
    metric_filtered_data, time_column, metric_name, steepest_fall_time, file_date_time, start_time = args
    
    try:
        # Processing strategy: Hardware-accelerated metric processing
        #print(f"Processing strategy: Hardware-accelerated metric analysis for {metric_name}")
        
        # Import calculate_statistics here to avoid circular imports
        from modules.calculate_statistics import calculate_statistics
        from shared.modules.ensure_consistent_structure import ensure_consistent_structure
        
        # Calculate statistics for this metric using hardware optimization
        statistics_df = calculate_statistics(
            metric_filtered_data, 
            metric_name, 
            file_date_time, 
            start_time, 
            steepest_fall_time.strftime('%H:%M')
        )
        
        # Ensure consistent structure
        statistics_df = ensure_consistent_structure(statistics_df)
        
        return statistics_df
        
    except Exception as e:
        print(f"Error processing metric {metric_name}: {e}")
        return pd.DataFrame()

def process_single_file(args):
    """Process a single CSV file with hardware-accelerated optimization."""
    file_path, metric_names, baseline_metric_name, hardware_allocation = args
    
    try:
        # Processing strategy: Hardware-accelerated file processing
        print(f"Processing strategy: Hardware-accelerated file processing for {os.path.basename(file_path)}")
        
        file_start_time = pd.Timestamp.now()
        
        # Load the full file once to find steepest fall
        perfmon_data = pd.read_csv(file_path, low_memory=False)
        
        if perfmon_data.empty:
            print(f"No data found in file: {file_path}")
            return []
        
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
            return []
        
        # Extract date/time info
        file_date_time = steepest_fall_time.strftime('%d-%b')
        filtered_perfmon_data = perfmon_data[perfmon_data[time_column] <= steepest_fall_time]
        start_time = filtered_perfmon_data[time_column].min().strftime('%H:%M')

        # Clear original data to free memory
        del perfmon_data
        
        # Pre-filter data per metric to enable efficient parallel processing
        metric_specific_data = {}
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            if metric_columns:
                columns_to_keep = [time_column] + metric_columns
                metric_specific_data[metric_name] = filtered_perfmon_data[columns_to_keep]
        
        # Clear the large filtered DataFrame immediately
        del filtered_perfmon_data
        gc.collect()
        
        # Prepare arguments for parallel metric processing
        metric_args = [
            (metric_data, time_column, metric_name, steepest_fall_time, file_date_time, start_time)
            for metric_name, metric_data in metric_specific_data.items()
        ]
        
        # Use hardware-aware worker allocation for metric processing
        metric_workers = min(len(metric_names), hardware_allocation['cpu_per_file'])
        
        print(f"Using {metric_workers} workers for {len(metric_args)} metrics in {os.path.basename(file_path)}")
        
        # Process metrics in parallel with hardware optimization
        statistics_list = []
        with ThreadPoolExecutor(max_workers=metric_workers) as executor:
            future_to_metric = {
                executor.submit(process_single_metric, args): args[2]  # args[2] is metric_name
                for args in metric_args
            }
            
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    if not result.empty:
                        statistics_list.append(result)
                except Exception as e:
                    print(f"Error processing metric {metric_name}: {e}")
        
        # Clear metric-specific data and force garbage collection
        del metric_specific_data
        gc.collect()
        
        file_end_time = pd.Timestamp.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        print(f"File {os.path.basename(file_path)} completed in {file_duration:.2f} seconds")
        
        return statistics_list
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def calculate_hardware_aware_workers(csv_file_paths: List[str]) -> Dict[str, int]:
    """
    Calculate optimal number of workers based on hardware capabilities and file characteristics.
    Returns hardware-aware worker allocation.
    """
    # Processing strategy: Hardware-accelerated worker allocation
    print("Processing strategy: Hardware-accelerated worker allocation calculation")
    
    if not csv_file_paths:
        return {'file_workers': 1, 'cpu_per_file': 1}
    
    # Get hardware detector for intelligent allocation
    hardware = get_hardware_detector()
    
    # Estimate workload size
    sample_size = min(3, len(csv_file_paths))
    total_sample_size = 0
    
    for i in range(sample_size):
        try:
            file_size_gb = os.path.getsize(csv_file_paths[i]) / (1024**3)
            total_sample_size += file_size_gb
        except OSError:
            total_sample_size += 1.0  # Conservative estimate
    
    avg_file_size_gb = total_sample_size / sample_size if sample_size > 0 else 1.0
    total_workload_gb = avg_file_size_gb * len(csv_file_paths)
    
    # Get optimal worker allocation for CPU-intensive tasks
    worker_allocation = get_optimal_workers('cpu_bound', total_workload_gb)
    
    # Calculate file-level and metric-level workers
    file_workers = min(len(csv_file_paths), worker_allocation['cpu'])
    
    # Allocate remaining CPU cores for metric processing within each file
    total_cpu_cores = hardware.profile.cpu.cores
    cpu_per_file = max(1, total_cpu_cores // file_workers)
    
    allocation = {
        'file_workers': file_workers,
        'cpu_per_file': cpu_per_file,
        'total_cpu_cores': total_cpu_cores,
        'gpu_cores': worker_allocation.get('gpu', 0),
        'npu_tops': worker_allocation.get('npu', 0)
    }
    
    print(f"Hardware allocation: {file_workers} file workers, {cpu_per_file} CPU cores per file")
    print(f"Total workload: {total_workload_gb:.1f}GB across {len(csv_file_paths)} files")
    
    return allocation

def file_processor_accelerated(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> pd.DataFrame:
    """
    Hardware-accelerated file processor with intelligent worker allocation.
    Processes CSV files in parallel with CPU+GPU+NPU optimization.
    """
    
    # Processing strategy: Hardware-accelerated file processing orchestration
    print("Processing strategy: Hardware-accelerated file processing orchestration")
    
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
    
    print(f"Found {len(csv_file_paths)} CSV files to process")
    
    # Calculate hardware-aware worker allocation
    hardware_allocation = calculate_hardware_aware_workers(csv_file_paths)
    
    # Prepare arguments for parallel file processing
    file_args = [
        (file_path, metric_names, baseline_metric_name, hardware_allocation)
        for file_path in csv_file_paths
    ]
    
    # Process files in parallel with hardware acceleration
    all_statistics_list = []
    
    print(f"Starting hardware-accelerated processing with {hardware_allocation['file_workers']} file workers...")
    parallel_start_time = pd.Timestamp.now()
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as executor:
        future_to_file = {
            executor.submit(process_single_file, args): args[0] 
            for args in file_args
        }
        
        completed_files = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed_files += 1
            
            try:
                file_statistics = future.result()
                if file_statistics:
                    all_statistics_list.extend(file_statistics)
                print(f"Completed {completed_files}/{len(csv_file_paths)} files")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    parallel_end_time = pd.Timestamp.now()
    parallel_duration = (parallel_end_time - parallel_start_time).total_seconds()
    throughput = len(csv_file_paths) / parallel_duration if parallel_duration > 0 else 0
    
    print(f"Hardware-accelerated processing completed in {parallel_duration:.2f} seconds")
    print(f"Throughput: {throughput:.2f} files/second")
    
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
    """Backward-compatible wrapper for the accelerated file processor"""
    # Processing strategy: Hardware-accelerated (wrapper function)
    print("Processing strategy: Hardware-accelerated file processor (backward-compatible wrapper)")
    
    return file_processor_accelerated(log_directory, metric_names, baseline_metric_name)
