# Enhanced batch GPU processor for true parallel metric processing
# Processes multiple metrics simultaneously on different GPU cores

import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from .parallel_gpu_processor import get_parallel_gpu_processor

def process_file_metrics_with_parallel_gpu(filtered_file_data: List[Dict], metric_names: List[str]) -> List[pd.DataFrame]:
    """
    Enhanced GPU batch processing that utilizes multiple GPU cores in parallel.
    Each metric gets processed on a separate GPU core simultaneously.
    """
    if not filtered_file_data:
        return []
    
    gpu_processor = get_parallel_gpu_processor()
    gpu_info = gpu_processor.get_gpu_utilization_info()
    
    print(f"\n=== Enhanced Parallel GPU Processing ===")
    print(f"GPU Device: {gpu_info['device_name']}")
    print(f"Compute Units: {gpu_info['compute_units']}")
    print(f"Max Parallel Jobs: {gpu_info['max_parallel_jobs']}")
    print(f"Command Queues: {gpu_info['command_queues']}")
    
    all_statistics_list = []
    batch_start_time = pd.Timestamp.now()
    
    # Process each file's metrics in parallel batches
    for file_data in filtered_file_data:
        if file_data is None:
            continue
            
        file_path = file_data['file_path']
        filtered_perfmon_data = file_data['filtered_data']
        time_column = file_data['time_column']
        steepest_fall_time = file_data['steepest_fall_time']
        file_date_time = file_data['file_date_time']
        start_time = file_data['start_time']
        
        file_name = file_path.split('\\')[-1]
        print(f"\nProcessing {len(metric_names)} metrics for {file_name} in parallel...")
        
        # Prepare all metric data for parallel GPU processing
        file_metric_data = {}
        valid_metrics = []
        column_to_metric_map = {}  # Map full column names to base metric names
        
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            if metric_columns:
                # Process EACH matching column individually, not just the first one
                for column_name in metric_columns:
                    column_data = pd.to_numeric(filtered_perfmon_data[column_name], errors='coerce').dropna()
                    if len(column_data) > 0:
                        # Use the full column name as the key to ensure uniqueness
                        file_metric_data[column_name] = column_data.values
                        valid_metrics.append(column_name)
                        # Keep track of which base metric this column belongs to
                        column_to_metric_map[column_name] = metric_name
        
        if not file_metric_data:
            print(f"No valid metric data found for {file_path}")
            continue
        
        file_start_time = time.time()
        
        # Process ALL metrics for this file in parallel on GPU
        parallel_results = gpu_processor.process_metrics_parallel(file_metric_data)
        
        file_end_time = time.time()
        file_duration = file_end_time - file_start_time
        
        print(f"Parallel GPU processing completed in {file_duration:.3f}s for {len(valid_metrics)} metrics")
        
        # Convert results to DataFrame format
        for column_name in valid_metrics:
            if column_name not in parallel_results:
                continue
                
            try:
                result = parallel_results[column_name]
                
                # Import required functions
                from .calculate_statistics import extract_header_from_column_name, remove_first_word_after_backslashes
                from shared.modules.ensure_consistent_structure import ensure_consistent_structure
                
                # Extract header information directly from the column name
                column_header = extract_header_from_column_name(column_name)
                modified_metric_name = remove_first_word_after_backslashes(column_name)
                modified_metric_names = [remove_first_word_after_backslashes(col) for col in metric_columns]
                
                # Create duration string
                duration = f"({start_time} - {steepest_fall_time.strftime('%H:%M')})"
                
                # Create statistics data with parallel GPU results for individual column
                statistics_data = {
                    'Metric': [modified_metric_name],
                    f"{file_date_time}\n{column_header}\nAvg.\n{duration}": [result['mean']],
                    f"{file_date_time}\n{column_header}\nMax.\n{duration}": [result['max']]
                }
                
                # Add bytes to Mbps conversion if applicable
                if 'bytes total/sec' in modified_metric_name.lower():
                    mbps_metric = modified_metric_name.replace('Bytes', 'Mbps')
                    
                    # Convert to Mbps
                    avg_mbps = (result['mean'] * 8) / 1_000_000
                    max_mbps = (result['max'] * 8) / 1_000_000
                    
                    avg_col = f"{file_date_time}\n{column_header}\nAvg.\n{duration}"
                    max_col = f"{file_date_time}\n{column_header}\nMax.\n{duration}"
                    
                    # Add Mbps metric to the data
                    statistics_data['Metric'].append(mbps_metric)
                    statistics_data[avg_col].append(avg_mbps)
                    statistics_data[max_col].append(max_mbps)
                
                # Create DataFrame
                statistics_df = pd.DataFrame(statistics_data)
                statistics_df = ensure_consistent_structure(statistics_df)
                
                if not statistics_df.empty:
                    all_statistics_list.append(statistics_df)
                    
                #print(f"  ✓ {modified_metric_name}: {result.get('processed_on', 'GPU')} - Mean: {result['mean']:.2f}, Max: {result['max']:.2f}")
                    
            except Exception as e:
                print(f"Error creating DataFrame for column {column_name}: {e}")
    
    batch_end_time = pd.Timestamp.now()
    batch_duration = (batch_end_time - batch_start_time).total_seconds()
    
    # Calculate total actual columns processed (not just base metric names)
    total_columns_processed = sum(len([col for col in file_data['filtered_data'].columns 
                                      for metric_name in metric_names if metric_name in col]) 
                                 for file_data in filtered_file_data if file_data is not None)
    throughput = total_columns_processed / batch_duration if batch_duration > 0 else 0
    
    print(f"\n=== Parallel GPU Processing Complete ===")
    print(f"Total Duration: {batch_duration:.2f}s")
    print(f"Total Columns Processed: {total_columns_processed} (individual performance counters)")
    print(f"Base Metric Names: {len(metric_names)}")
    print(f"Throughput: {throughput:.2f} columns/second")
    print(f"Results: {len(all_statistics_list)} DataFrames generated")
    
    return all_statistics_list

def demonstrate_gpu_parallelism(metric_names: List[str], sample_data_size: int = 10000):
    """
    Demonstrate true GPU parallelism by processing multiple metrics simultaneously.
    """
    print(f"\n=== GPU Parallelism Demonstration ===")
    
    # Create sample data for multiple metrics
    print(f"Creating sample data for {len(metric_names)} metrics ({sample_data_size} data points each)")
    
    metric_data_dict = {}
    for metric_name in metric_names[:8]:  # Limit to 8 metrics for demo
        # Generate realistic performance counter data
        base_value = np.random.uniform(10, 1000)
        trend = np.linspace(0, base_value * 0.3, sample_data_size)
        noise = np.random.normal(0, base_value * 0.1, sample_data_size)
        data = np.maximum(0, base_value + trend + noise)  # Ensure non-negative
        metric_data_dict[metric_name] = data
    
    # Get GPU processor
    gpu_processor = get_parallel_gpu_processor()
    gpu_info = gpu_processor.get_gpu_utilization_info()
    
    print(f"\nGPU Configuration:")
    print(f"  Device: {gpu_info['device_name']}")
    print(f"  Compute Units: {gpu_info['compute_units']}")
    print(f"  Parallel Queues: {gpu_info['command_queues']}")
    print(f"  Max Parallel Jobs: {gpu_info['max_parallel_jobs']}")
    
    # Sequential CPU processing for comparison
    print(f"\n--- Sequential CPU Processing ---")
    cpu_start = time.time()
    cpu_results = {}
    for metric_name, data in metric_data_dict.items():
        cpu_results[metric_name] = {
            'mean': float(np.mean(data)),
            'max': float(np.max(data))
        }
    cpu_end = time.time()
    cpu_duration = cpu_end - cpu_start
    print(f"CPU Sequential: {cpu_duration:.3f}s for {len(metric_data_dict)} metrics")
    
    # Parallel GPU processing
    print(f"\n--- Parallel GPU Processing ---")
    gpu_start = time.time()
    gpu_results = gpu_processor.process_metrics_parallel(metric_data_dict)
    gpu_end = time.time()
    gpu_duration = gpu_end - gpu_start
    
    print(f"GPU Parallel: {gpu_duration:.3f}s for {len(metric_data_dict)} metrics")
    
    # Calculate speedup
    speedup = cpu_duration / gpu_duration if gpu_duration > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster with parallel GPU processing")
    
    # Verify results match
    differences = []
    for metric_name in metric_data_dict.keys():
        if metric_name in cpu_results and metric_name in gpu_results:
            cpu_mean = cpu_results[metric_name]['mean']
            gpu_mean = gpu_results[metric_name]['mean']
            diff = abs(cpu_mean - gpu_mean) / cpu_mean * 100 if cpu_mean != 0 else 0
            differences.append(diff)
    
    avg_difference = np.mean(differences) if differences else 0
    print(f"Average difference between CPU and GPU results: {avg_difference:.6f}%")
    
    return {
        'cpu_duration': cpu_duration,
        'gpu_duration': gpu_duration,
        'speedup': speedup,
        'avg_difference': avg_difference
    }
