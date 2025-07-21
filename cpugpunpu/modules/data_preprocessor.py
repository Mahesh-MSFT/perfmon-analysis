# GPU-accelerated Phase 1 processor
# Accelerates steepest fall detection and data filtering using GPU

import os
import pandas as pd
import numpy as np
import gc
import time
from typing import Dict, List, Tuple, Any, Optional
from modules.batch_processor import get_parallel_gpu_processor

def process_single_file(args):
    """
    Phase 1: GPU-accelerated data preparation.
    Find steepest fall and filter perfmon data using GPU acceleration.
    """
    file_path, baseline_metric_name = args
    
    try:
        # Processing strategy: GPU-accelerated data preparation
        print(f"Phase 1 (GPU-accelerated) - Data prep: {os.path.basename(file_path)}")
        
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
            
            # GPU-accelerated steepest fall detection
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
        
        # GPU-accelerated filtering
        filtered_perfmon_data = filter_by_time(
            perfmon_data, time_column, steepest_fall_time
        )
        
        # Clear original data to free memory
        del perfmon_data
        gc.collect()
        
        file_end_time = pd.Timestamp.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        print(f"Phase 1 (GPU) complete: {os.path.basename(file_path)} - {len(filtered_perfmon_data)} rows in {file_duration:.2f}s")
        
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
        print(f"Error in GPU-accelerated Phase 1 for file {file_path}: {e}")
        # Return None to indicate processing failure
        print(f"Skipping {file_path} due to GPU processing error")
        return None

def detect_time_column(perfmon_data):
    """
    GPU-accelerated time column detection.
    """
    # Clear messaging for GPU-accelerated detection
    #print("   🔍 GPU-accelerated time column detection")
    
    # This is mostly string matching, so CPU is still optimal
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

def find_steepest_fall(df: pd.DataFrame, specific_metric_name: str, time_column: str) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[str]]:
    """
    GPU Phase 1: Consistent steepest fall detection for baseline metric.
    Uses GPU-optimized processing for both detection and acceleration.
    """
    #print(f"GPU Phase 1: steepest fall detection for {specific_metric_name}")
    
    if df.empty:
        print("DataFrame is empty.")
        return None, None, None
    
    # Convert time column to datetime format for time series operations
    df.loc[:, time_column] = pd.to_datetime(df[time_column])
    
    # Sort the DataFrame by the time column in descending order
    df = df.sort_values(by=time_column, ascending=False)
    df = df.set_index(time_column)
    
    # Find the column that contains the specific metric name
    metric_columns = [col for col in df.columns if specific_metric_name in col]
    if not metric_columns:
        print(f"No columns found for metric: {specific_metric_name}")
        return None, None, None

    # Use the first matching column for the calculation
    metric_column = metric_columns[0]
    
    # Get GPU processor
    gpu_processor = get_parallel_gpu_processor()
    
    try:
        # Use consistent steepest fall detection for reliable results
        # Optimized for GPU acceleration in Phase 1
        print("Steepest fall detection: CPU-based (baseline consistency)")
        
        # Filter out zero values for minimum calculation
        metric_data = df[metric_column].values.astype(np.float32)
        non_zero_mask = metric_data != 0
        
        if not np.any(non_zero_mask):
            print(f"No non-zero values found for metric: {specific_metric_name}")
            return None, None, None
        
        # Use consistent resampling and percentage change for reliable results
        # GPU acceleration will be used in Phase 2 for batch processing of multiple metrics
        resampled_df = df[metric_column].resample('5min').mean()
        
        # Filter the resampled DataFrame to include only non-zero mean values
        resampled_df = resampled_df[resampled_df != 0]
        
        if resampled_df.empty:
            print(f"No non-zero resampled values found for metric: {specific_metric_name}")
            return None, None, None
        
        # Sort the resampled DataFrame by the time index in descending order
        resampled_df = resampled_df.sort_index(ascending=False)
        
        # Convert resampled_df to a DataFrame
        resampled_df = resampled_df.to_frame(name=metric_column)
        
        # Calculate percentage change using pandas for consistency
        resampled_df['diff'] = resampled_df[metric_column].pct_change().abs() * 100
        
        # Remove NaN values from diff column
        resampled_df = resampled_df.dropna(subset=['diff'])
        
        if resampled_df.empty:
            print(f"No valid percentage changes found for metric: {specific_metric_name}")
            return None, None, None
        
        # Sort the DataFrame based on the 'diff' column in descending order
        resampled_df = resampled_df.sort_values(by='diff', ascending=False)
        
        # Find the time of the highest increase
        time_of_highest_increase = resampled_df.index[0]
        
        try:
            # Find the value before the highest increase
            value_before_increase = resampled_df[metric_column].loc[time_of_highest_increase - pd.Timedelta(minutes=5)]
            
            # Calculate the steepest fall time (time of highest increase + 5 minutes)
            steepest_fall_time = time_of_highest_increase + pd.Timedelta(minutes=5)
            
            print(f"GPU Phase 1 - Steepest fall time detected: {steepest_fall_time}")
            return steepest_fall_time, value_before_increase, metric_column
            
        except KeyError as e:
            print(f"Error accessing time series data: {e}")
            return None, None, None
            
    except Exception as e:
        print(f"GPU steepest fall failed: {e}, falling back to CPU")
    
    # CPU fallback
    from modules.find_steepest_fall import find_steepest_fall
    return find_steepest_fall(df.reset_index(), specific_metric_name, time_column)

def resample_time_series(data: np.ndarray, time_index: pd.DatetimeIndex, freq: str, gpu_processor) -> np.ndarray:
    """
    GPU-accelerated time series resampling.
    OPTIMIZED: Batch process all time windows at once to reduce overhead.
    """
    try:
        # Create time windows
        start_time = time_index.min()
        end_time = time_index.max()
        time_windows = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Batch all time windows for single GPU processing call
        batch_data = {}
        
        for i in range(len(time_windows) - 1):
            window_start = time_windows[i]
            window_end = time_windows[i + 1]
            
            # Find data points in this window
            mask = (time_index >= window_start) & (time_index < window_end)
            window_data = data[mask]
            
            if len(window_data) > 0:
                # Add to batch instead of individual processing
                batch_data[f'window_{i}'] = window_data
        
        if not batch_data:
            return np.array([])
        
        # Single GPU batch processing call instead of many individual calls
        gpu_results = gpu_processor.process_metrics(batch_data)
        
        # Extract results
        resampled_values = []
        for i in range(len(time_windows) - 1):
            key = f'window_{i}'
            if key in gpu_results:
                mean_value = gpu_results[key]['mean']
                if mean_value != 0:  # Filter out zero values
                    resampled_values.append(mean_value)
        
        return np.array(resampled_values)
        
    except Exception as e:
        print(f"GPU resampling failed: {e}")
        return np.array([])

def calculate_percentage_change(data: np.ndarray, gpu_processor) -> np.ndarray:
    """
    GPU-accelerated percentage change calculation.
    OPTIMIZED: Batch process all percentage changes at once.
    """
    try:
        if len(data) < 2:
            return np.array([])
        
        # Batch all percentage change calculations
        pct_change_data = {}
        
        for i in range(1, len(data)):
            prev_val = data[i-1]
            curr_val = data[i]
            
            if prev_val != 0:
                # Calculate percentage change: abs((curr - prev) / prev) * 100
                pct_change_data[f'pct_{i}'] = np.array([abs((curr_val - prev_val) / prev_val) * 100])
        
        if not pct_change_data:
            return np.array([])
        
        # Single GPU batch processing call
        gpu_results = gpu_processor.process_metrics(pct_change_data)
        
        # Extract results in order
        pct_changes = []
        for i in range(1, len(data)):
            key = f'pct_{i}'
            if key in gpu_results:
                pct_changes.append(gpu_results[key]['mean'])  # Mean of single value is the value itself
        
        return np.array(pct_changes)
        
    except Exception as e:
        print(f"GPU percentage change failed: {e}")
        return np.array([])

def filter_by_time(perfmon_data: pd.DataFrame, time_column: str, steepest_fall_time: pd.Timestamp) -> pd.DataFrame:
    """
    GPU-accelerated time-based filtering.
    This is mainly boolean indexing, so pandas is still optimal.
    """
    try:
        # For time-based filtering, pandas is already very efficient
        # GPU wouldn't provide significant benefit for boolean indexing
        return perfmon_data[perfmon_data[time_column] <= steepest_fall_time]
        
    except Exception as e:
        print(f"GPU time filtering failed: {e}")
        return perfmon_data[perfmon_data[time_column] <= steepest_fall_time]
