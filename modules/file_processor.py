# Parallel file processor with simplified approach
import os
import pandas as pd
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from modules.calculate_statistics import calculate_statistics
from modules.ensure_consistent_structure import ensure_consistent_structure
from modules.find_steepest_fall import find_steepest_fall

def detect_time_column(perfmon_data):
    """
    Detect the time column in the CSV data.
    Handles any timezone that relog.exe might generate (GMT, PST, EST, etc.) 
    with both Standard Time and Daylight/Summer Time variations.
    """
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
    """Process a single metric for a given file and steepest fall data."""
    metric_filtered_data, time_column, metric_name, steepest_fall_time, file_date_time, start_time = args
    
    try:
        #print(f"Processing metric: {metric_name}")

        # The metric_filtered_data is already pre-filtered to contain only relevant columns
        # No need for additional column filtering here
        
        # Calculate statistics for this metric
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
    """Process a single CSV file and return all statistics for all metrics."""
    file_path, metric_names, baseline_metric_name, file_level_workers = args
    
    try:
        print(f"Processing file: {file_path}")
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
        
        # filter the DataFrame to only the needed columns before calling find_steepest_fall:
        baseline_columns = [col for col in perfmon_data.columns if baseline_metric_name in col]
        if baseline_columns:
            small_df = perfmon_data[[time_column] + baseline_columns[:1]]
            # Find the steepest fall for the baseline metric
            steepest_fall_time, steepest_fall_value, column_name = find_steepest_fall(
                small_df, baseline_metric_name, time_column
            )

            print(f"Steepest fall time for {file_path}: {steepest_fall_time}, value: {steepest_fall_value}")

        if not (steepest_fall_time and steepest_fall_value):
            print(f"No steepest fall found for {baseline_metric_name} in {file_path}")
            # Clear memory before returning
            del perfmon_data
            return []
        
        # Extract date/time info
        file_date_time = steepest_fall_time.strftime('%d-%b')
        filtered_perfmon_data = perfmon_data[perfmon_data[time_column] <= steepest_fall_time]
        start_time = filtered_perfmon_data[time_column].min().strftime('%H:%M')

        # Clear original data to free memory - we only need the filtered data
        del perfmon_data
        
        # Pre-filter data per metric to enable ProcessPoolExecutor usage
        # Create metric-specific DataFrames with only relevant columns
        metric_specific_data = {}
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            if metric_columns:
                columns_to_keep = [time_column] + metric_columns
                metric_specific_data[metric_name] = filtered_perfmon_data[columns_to_keep]
        
        # Clear the large filtered DataFrame immediately after creating metric-specific data
        del filtered_perfmon_data
        gc.collect()
        
        # Prepare arguments for parallel metric processing using pre-filtered data
        metric_args = [
            (metric_data, time_column, metric_name, steepest_fall_time, file_date_time, start_time)
            for metric_name, metric_data in metric_specific_data.items()
        ]
        
        # Process metrics in parallel
        statistics_list = []
        # Adjust metric parallelism based on file-level parallelism
        # If we're processing fewer files, we can use more CPU cores per file
        base_cpu_fraction = 0.8 / max(1, file_level_workers * 0.5)  # Scale down with more file workers
        max_workers_metrics = max(1, min(len(metric_names), int(os.cpu_count() * base_cpu_fraction)))
        
        print(f"Using {max_workers_metrics} workers for {len(metric_args)} metrics")
        
        # Use ProcessPoolExecutor for metric processing since we're now passing small DataFrames
        # which are much cheaper to serialize between processes
        with ThreadPoolExecutor(max_workers=max_workers_metrics) as executor:
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

def calculate_optimal_file_workers(csv_file_paths):
    """
    Calculate optimal number of file workers based on available memory and estimated file sizes.
    Returns a conservative estimate to prevent memory exhaustion.
    """
    if not csv_file_paths:
        return 1
    
    # Get available memory (leave 20% buffer for system)
    available_memory_gb = psutil.virtual_memory().available / (1024**3) * 0.8
    
    # Estimate average file size by sampling a few files
    sample_size = min(3, len(csv_file_paths))
    total_sample_size = 0
    
    for i in range(sample_size):
        try:
            file_size_gb = os.path.getsize(csv_file_paths[i]) / (1024**3)
            total_sample_size += file_size_gb
        except OSError:
            # If we can't get file size, assume 1GB as conservative estimate
            total_sample_size += 1.0
    
    avg_file_size_gb = total_sample_size / sample_size if sample_size > 0 else 1.0
    
    # Calculate how many files can fit in memory
    max_concurrent_files = max(1, int(available_memory_gb / avg_file_size_gb))
    
    # Cap at number of available files and reasonable CPU limit
    cpu_limit = max(1, int(os.cpu_count() * 0.75))
    optimal_workers = min(max_concurrent_files, len(csv_file_paths), cpu_limit)
    
    print(f"Memory-based calculation: {available_memory_gb:.1f}GB available, "
          f"{avg_file_size_gb:.1f}GB avg file size, "
          f"{cpu_limit} CPU limit, "
          f"optimal file workers: {optimal_workers}")
    
    return optimal_workers

def file_processor(log_directory, metric_names, baseline_metric_name):
    """
    Parallel version of file_processor with simplified approach.
    Processes files in parallel, then metrics within each file in parallel.
    """
    
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
    
    # Calculate optimal file workers based on memory and file sizes
    max_workers = calculate_optimal_file_workers(csv_file_paths)
    
    # Prepare arguments for parallel file processing
    file_args = [
        (file_path, metric_names, baseline_metric_name, max_workers)
        for file_path in csv_file_paths
    ]
    
    # Process files in parallel
    all_statistics_list = []
    
    print(f"Starting parallel processing with {max_workers} workers...")
    parallel_start_time = pd.Timestamp.now()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    print(f"Parallel processing completed in {parallel_duration:.2f} seconds")
    
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