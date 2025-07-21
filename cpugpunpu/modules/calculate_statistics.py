# Hardware-accelerated statistics calculation for perfmon3.py
# Utilizes CPU and GPU capabilities for optimal statistical computation

import pandas as pd
import numpy as np
from typing import Dict, List, Any

def extract_header_from_column_name(column_name: str) -> str:
    """Extract header from performance monitor column name"""
    # Processing strategy: CPU-based (string processing)
    # This is a lightweight string operation that doesn't benefit from GPU acceleration
    
    # Import the existing function
    from modules.extract_header_from_column_name import extract_header_from_column_name as cpu_extract_header
    return cpu_extract_header(column_name)

def remove_first_word_after_backslashes(column_name: str) -> str:
    """Remove first word after backslashes in column name"""
    # Processing strategy: CPU-based (string processing)
    # This is a lightweight string operation that doesn't benefit from GPU acceleration
    
    # Import the existing function
    from modules.remove_first_word_after_backslashes import remove_first_word_after_backslashes as cpu_remove_first_word
    return cpu_remove_first_word(column_name)

def calculate_statistics(df: pd.DataFrame, metric_name: str, file_date_time: str, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Hardware-accelerated statistics calculation with intelligent processing strategy selection.
    
    Args:
        df: Performance monitor DataFrame
        metric_name: Name of the metric to calculate statistics for
        file_date_time: Date/time identifier for the file
        start_time: Start time for the analysis window
        end_time: End time for the analysis window
        
    Returns:
        DataFrame with calculated statistics
    """
    # Processing strategy: Hardware-accelerated statistical computation
    #print(f"Processing strategy: Hardware-accelerated statistics calculation for {metric_name}")
    
    # Import parallel GPU processor for true multi-core utilization
    from modules.batch_processor import get_parallel_gpu_processor
    gpu_processor = get_parallel_gpu_processor()
    
    # Filter columns based on the metric name
    metric_columns = [col for col in df.columns if metric_name in col]
    
    if not metric_columns:
        print(f"No columns found for metric: {metric_name}")
        return pd.DataFrame()
    
    # Get the header
    column_header = extract_header_from_column_name(metric_columns[0])
    
    # Ensure the data is numeric - use hardware-accelerated conversion where possible
    df_numeric = df[metric_columns].copy()
    
    # For large datasets, this could benefit from GPU acceleration
    # Currently using pandas (CPU-based) but optimized for memory efficiency
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    
    # Remove rows where ALL metric columns are NaN (no data for this metric)
    df_numeric = df_numeric.dropna(how='all')
    
    # Hardware-accelerated statistical calculations (Phase 2: Always use GPU in batch processing)
    dataset_size = len(df_numeric)
    
    # Convert to numpy for GPU processing (GPU logging handled by accelerator)
    numeric_data = df_numeric.values
    
    # Use parallel GPU processing for true multi-core utilization
    try:
        # Create metric data dictionary for parallel processing
        metric_data_dict = {metric_name: numeric_data}
        
        # Process using parallel GPU processor
        gpu_results = gpu_processor.process_metrics(metric_data_dict)
        
        if metric_name in gpu_results:
            result = gpu_results[metric_name]
            average_values_array = np.full(len(metric_columns), result['mean'])
            maximum_values_array = np.full(len(metric_columns), result['max'])
            #print(f"✓ {metric_name}: Processed on {result.get('processed_on', 'GPU')}")
        else:
            raise Exception("No results returned from parallel GPU processor")
        
        # Convert back to pandas Series with proper index
        average_values = pd.Series(average_values_array, index=metric_columns)
        maximum_values = pd.Series(maximum_values_array, index=metric_columns)
        
    except Exception as e:
        print(f"Parallel GPU acceleration failed, falling back to CPU: {e}")
        # Fallback to CPU-based calculation
        average_values = df_numeric.mean()
        maximum_values = df_numeric.max()
    
    # Use remove_first_word_after_backslashes to get the modified column names
    modified_metric_names = [remove_first_word_after_backslashes(col) for col in metric_columns]

    # Combine start_time and end_time as (start_time - end_time) to create a new string
    duration = f"({start_time} - {end_time})"

    # Initialize lists in the statistics_data dictionary
    statistics_data = {
        'Metric': list(modified_metric_names),
        f"{file_date_time}\n{column_header}\nAvg.\n{duration}": list(average_values.values),
        f"{file_date_time}\n{column_header}\nMax.\n{duration}": list(maximum_values.values)
    }

    # Add additional metrics for bytes to Mbps conversion
    for i, metric in enumerate(modified_metric_names):
        if 'bytes total/sec' in metric.lower():
            mbps_metric = metric.replace('Bytes', 'Mbps')
            statistics_data['Metric'].append(mbps_metric)
            
            # Ensure the columns exist
            avg_col = f"{file_date_time}\n{column_header}\nAvg.\n{duration}"
            max_col = f"{file_date_time}\n{column_header}\nMax.\n{duration}"
            
            if avg_col not in statistics_data:
                statistics_data[avg_col] = []
            if max_col not in statistics_data:
                statistics_data[max_col] = []
                
            # Simple CPU-based unit conversion (bytes/sec to Mbps)
            avg_mbps = (average_values.values[i] * 8) / 1_000_000
            max_mbps = (maximum_values.values[i] * 8) / 1_000_000
            
            statistics_data[avg_col].append(avg_mbps)
            statistics_data[max_col].append(max_mbps)

    statistics_df = pd.DataFrame(statistics_data)
    
    return statistics_df

# Backward compatibility alias
def calculate_statistics_accelerated(df: pd.DataFrame, metric_name: str, file_date_time: str, start_time: str, end_time: str) -> pd.DataFrame:
    """Backward-compatible wrapper for calculate_statistics"""
    return calculate_statistics(df, metric_name, file_date_time, start_time, end_time)
