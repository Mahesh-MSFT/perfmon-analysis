# Optimized two-phase file processor for perfmon3.py
# Phase 1: CPU-optimized data preparation (I/O, parsing, filtering)  
# Phase 2: GPU-accelerated statistics processing (intensive computations)
# Architecture minimizes CPU↔GPU context switching for optimal performance

import os
import pandas as pd
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from .hardware_detector import get_hardware_detector

def detect_time_column(perfmon_data):
    """Detect the time column in the CSV data."""
    
    for column in perfmon_data.columns:
        if (column.startswith('(PDH-CSV 4.0) (') and 
            'Time)(' in column and 
            column.endswith(')')):
            return column
    
    if len(perfmon_data.columns) > 0:
        return perfmon_data.columns[0]
    
    raise ValueError("No time column found in the CSV data")

def process_single_file(args):
    """
    Phase 1: CPU-optimized data preparation.
    Find steepest fall and filter perfmon data using efficient CPU processing.
    GPU acceleration reserved for Phase 2 statistical computations.
    """
    file_path, baseline_metric_name = args
    
    try:
        # Processing strategy: CPU-optimized data preparation (Phase 1)
        # Reserve GPU acceleration for computationally intensive Phase 2 operations
        print(f"Processing file: {os.path.basename(file_path)}")
        
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
            
            # CPU-optimized steepest fall detection (avoid GPU context switch for small operations)
            from .find_steepest_fall import find_steepest_fall
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
        
        # CPU-optimized filtering - keep data from steepest fall time onwards  
        # (GPU transfer overhead not justified for simple DataFrame filtering)
        filtered_perfmon_data = perfmon_data[perfmon_data[time_column] >= steepest_fall_time].copy()
        
        # Clear original data to free memory
        del perfmon_data
        gc.collect()
        
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
        print(f"Error processing file {file_path}: {e}")
        # Return None to indicate processing failure
        return None

def collect_csv_files(log_directory: str) -> List[str]:
    """Collect all CSV files from the log directory."""
    csv_file_paths = []
    for root, dirs, files in os.walk(log_directory):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                csv_file_paths.append(file_path)
    return csv_file_paths

def calculate_optimal_gpu_workers() -> int:
    """Calculate optimal GPU workers based on hardware capabilities."""
    hardware = get_hardware_detector()
    
    if not hardware.profile.gpu:
        return 4  # Fallback for systems without GPU detection
    
    gpu_compute_units = hardware.profile.gpu.compute_units
    gpu_memory_gb = hardware.profile.gpu.memory_gb
    
    # Calculate based on compute units
    if gpu_compute_units >= 64:  # Intel Arc or equivalent
        optimal_gpu_workers = min(16, max(8, gpu_compute_units // 4))
    elif gpu_compute_units >= 32:  # Mid-range GPU
        optimal_gpu_workers = min(8, max(4, gpu_compute_units // 4))
    else:  # Lower-end GPU
        optimal_gpu_workers = min(4, max(2, gpu_compute_units // 8))
    
    # Adjust for available memory (each worker processes ~1-2GB data)
    memory_limited_workers = max(2, int(gpu_memory_gb / 2))
    optimal_gpu_workers = min(optimal_gpu_workers, memory_limited_workers)
    
    return optimal_gpu_workers

def calculate_hardware_aware_workers(csv_file_paths: List[str]) -> Dict[str, int]:
    """Calculate optimal workers for parallel processing using 80% of available CPU cores."""
    
    if not csv_file_paths:
        return {'file_workers': 1}
    
    hardware = get_hardware_detector()
    total_cpu_cores = hardware.profile.cpu.cores
    available_memory_gb = hardware.profile.available_memory_gb
    
    # Use 80% of CPU cores for optimal performance while leaving headroom for system processes
    optimal_cpu_workers = max(1, int(total_cpu_cores * 0.8))
    
    # Memory-based limit: assume each worker needs ~2GB for large CSV processing
    memory_limited_workers = max(1, int(available_memory_gb / 2))
    
    # Take the minimum of file count, CPU-based, and memory-based limits
    file_workers = min(len(csv_file_paths), optimal_cpu_workers, memory_limited_workers)
    
    allocation = {
        'file_workers': file_workers,
        'total_cpu_cores': total_cpu_cores,
        'available_memory_gb': available_memory_gb,
        'cpu_utilization_percent': (file_workers / total_cpu_cores) * 100,
        'memory_per_worker_gb': available_memory_gb / file_workers if file_workers > 0 else 0
    }
    
    return allocation

def process_single_file_phase2(file_result, file_path, metric_names, baseline_metric_name):
    """Process Phase 2 for a single file."""
    try:
        # Pre-filter file data to include only relevant metric columns + time column
        filtered_perfmon_data = file_result['filtered_data']
        time_column = file_result['time_column']
        
        # Find columns that match any of the requested metrics
        relevant_columns = [time_column]  # Always include time column
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            relevant_columns.extend(metric_columns)
        
        # Remove duplicates while preserving order
        relevant_columns = list(dict.fromkeys(relevant_columns))
        
        # Create filtered dataset with only relevant columns
        filtered_data_subset = filtered_perfmon_data[relevant_columns]
        
        # Create a new file_result with only the relevant data
        filtered_file_result = {
            **file_result,  # Copy all metadata
            'filtered_data': filtered_data_subset
        }
        
        from .gpu_processor import process_file_metrics
        single_file_stats = process_file_metrics([filtered_file_result], metric_names, baseline_metric_name)
        
        if single_file_stats:
            return single_file_stats
        else:
            return []
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def prepare_cpu_allocation(csv_file_paths: List[str], baseline_metric_name: str):
    """Prepare arguments and allocate CPU resources for data preparation phase."""
    
    hardware_allocation = calculate_hardware_aware_workers(csv_file_paths)
    
    phase1_args = [
        (file_path, baseline_metric_name)
        for file_path in csv_file_paths
    ]
    
    return phase1_args, hardware_allocation

def execute_parallel_pipeline(phase1_args: List[Tuple], hardware_allocation: Dict[str, int], 
                              optimal_gpu_workers: int, metric_names: List[str], baseline_metric_name: str):
    """Execute the parallel pipeline with Phase 1 (CPU) and Phase 2 (GPU) processing."""
    
    filtered_file_data = []
    all_statistics_list = []
    
    # Timing tracking
    phase1_start = pd.Timestamp.now()
    first_phase2_start = None
    last_phase1_complete = None
    phase2_only_start = None
    phase2_only_end = None
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as process_executor:
        with ThreadPoolExecutor(max_workers=optimal_gpu_workers, thread_name_prefix="Phase2") as gpu_executor:
            
            # Submit Phase 1 tasks
            phase1_futures = {
                process_executor.submit(process_single_file, args): args[0] 
                for args in phase1_args
            }
            
            gpu_phase2_futures = []
            completed_files = 0
            
            # Process Phase 1 results and immediately start Phase 2
            for phase1_future in as_completed(phase1_futures):
                file_path = phase1_futures[phase1_future]
                completed_files += 1
                
                try:
                    file_result = phase1_future.result()
                    if file_result is not None:
                        filtered_file_data.append(file_result)
                        
                        print(f"Filtered {completed_files}/{len(phase1_args)} files")
                        
                        # Track first Phase 2 start
                        if first_phase2_start is None:
                            first_phase2_start = pd.Timestamp.now()
                        
                        # Submit Phase 2 task immediately
                        gpu_future = gpu_executor.submit(process_single_file_phase2, file_result, file_path, metric_names, baseline_metric_name)
                        gpu_phase2_futures.append(gpu_future)
                        
                        last_phase1_complete = pd.Timestamp.now()
                    else:
                        print(f"Filtered {completed_files}/{len(phase1_args)} files")
                        last_phase1_complete = pd.Timestamp.now()
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    last_phase1_complete = pd.Timestamp.now()
            
            # Wait for all Phase 2 processing to complete
            phase2_only_start = pd.Timestamp.now()
            
            for gpu_future in as_completed(gpu_phase2_futures):
                try:
                    gpu_stats = gpu_future.result()
                    if gpu_stats:
                        all_statistics_list.extend(gpu_stats)
                except Exception as e:
                    print(f"Error collecting Phase 2 results: {e}")
            
            phase2_only_end = pd.Timestamp.now()
    
    # Calculate timing metrics
    total_pipeline_duration = (phase2_only_end - phase1_start).total_seconds()
    phase1_only_duration = (last_phase1_complete - phase1_start).total_seconds()
    phase2_total_duration = (phase2_only_end - first_phase2_start).total_seconds() if first_phase2_start else 0
    phase2_only_duration = (phase2_only_end - phase2_only_start).total_seconds() if phase2_only_start else 0
    
    # Calculate overlap
    if first_phase2_start and last_phase1_complete and first_phase2_start < last_phase1_complete:
        overlap_duration = (min(last_phase1_complete, phase2_only_end) - first_phase2_start).total_seconds()
    else:
        overlap_duration = 0
    
    timing_data = {
        'total_time': total_pipeline_duration,
        'phase1_duration': phase1_only_duration,
        'phase2_duration': phase2_total_duration,
        'overlap_duration': overlap_duration,
        'files_processed': len(filtered_file_data),
        'statistics_generated': len(all_statistics_list),
        'gpu_processing_rate': len(all_statistics_list) / total_pipeline_duration if total_pipeline_duration > 0 else 0,
        'overall_throughput': len(phase1_args) / total_pipeline_duration if total_pipeline_duration > 0 else 0
    }
    
    return filtered_file_data, all_statistics_list, timing_data

def print_timing_analysis(timing_data: Dict):
    """Print timing analysis."""
    print(f"Parallel processing completed in {timing_data['total_time']:.2f} seconds")

def print_performance_summary(timing_data: Dict):
    """Print performance summary."""
    # Keep minimal output similar to cpuonly
    pass

def combine_results(all_statistics_list: List, baseline_metric_name: str) -> pd.DataFrame:
    """Combine all statistics results into a single DataFrame."""
    if not all_statistics_list:
        print("No statistics data was generated.")
        return pd.DataFrame()
    
    all_statistics_df = pd.concat(all_statistics_list, axis=0)
    
    # Pivot the DataFrame to have metrics as rows and average/maximum values as columns
    all_statistics_df = all_statistics_df.pivot_table(index='Metric', aggfunc='first')
    
    # Ensure the baseline metric is the first row
    if baseline_metric_name in all_statistics_df.index:
        all_statistics_df = all_statistics_df.reindex(
            [baseline_metric_name] + [idx for idx in all_statistics_df.index if idx != baseline_metric_name]
        )
    
    # Ensure all columns are numeric and round
    all_statistics_df = all_statistics_df.apply(pd.to_numeric, errors='coerce')
    all_statistics_df = all_statistics_df.round(0)
    
    return all_statistics_df

def file_processor(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process performance logs to generate statistical analysis.
    
    Two-phase approach:
    - Phase 1: Data preparation (steepest fall detection + filtering)
    - Phase 2: Statistics processing (parallel metric calculation)
    """
    
    # Step 1: Collect CSV files
    csv_file_paths = collect_csv_files(log_directory)
    
    if not csv_file_paths:
        print("No CSV files found in the specified directory.")
        return pd.DataFrame(), {}
    
    print(f"Found {len(csv_file_paths)} CSV files to process")
    
    # Step 2: Prepare CPU allocation for data preparation
    phase1_args, hardware_allocation = prepare_cpu_allocation(csv_file_paths, baseline_metric_name)
    
    # Step 3: Setup Phase 2 configuration
    optimal_gpu_workers = calculate_optimal_gpu_workers()
    
    # Step 4: Execute parallel pipeline
    filtered_file_data, all_statistics_list, timing_data = execute_parallel_pipeline(
        phase1_args, hardware_allocation, optimal_gpu_workers, metric_names, baseline_metric_name
    )
    
    # Step 5: Print timing summary
    print_timing_analysis(timing_data)
    print_performance_summary(timing_data)
    
    # Step 6: Combine results
    all_statistics_df = combine_results(all_statistics_list, baseline_metric_name)
    
    # Step 7: Prepare performance data for return
    performance_data = {
        'architecture': 'GPU+GPU',
        **timing_data
    }
    
    return all_statistics_df, performance_data
