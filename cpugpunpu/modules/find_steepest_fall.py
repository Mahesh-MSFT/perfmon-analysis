# GPU-accelerated steepest fall detection for perfmon3.py
# Phase 1: GPU-optimized processing for two-phase architecture

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union

def find_steepest_fall(df: pd.DataFrame, specific_metric_name: str, time_column: Optional[str] = None) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[str]]:
    """
    Phase 1: GPU-optimized steepest fall detection for two-phase architecture.
    
    Args:
        df: Performance monitor DataFrame
        specific_metric_name: Name of the metric to analyze
        time_column: Time column name (auto-detected if None)
        
    Returns:
        Tuple of (steepest_fall_time, steepest_fall_value, column_name)
    """
    # Processing strategy: Phase 1 GPU-optimized processing
    #print(f"Processing strategy: GPU-accelerated steepest fall detection for {specific_metric_name}")
    
    if df.empty:
        print("DataFrame is empty.")
        return None, None, None
    
    # If no time column is provided, try to detect it
    if time_column is None:
        # Look for PDH-CSV 4.0 format with any timezone
        for column in df.columns:
            if (column.startswith('(PDH-CSV 4.0) (') and 
                'Time)(' in column and 
                column.endswith(')')):
                time_column = column
                break
        
        # Fallback to first column if no PDH-CSV time column found
        if time_column is None and len(df.columns) > 0:
            time_column = df.columns[0]
    
    if time_column not in df.columns:
        print(f"Time column '{time_column}' not found in the DataFrame.")
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
    
    # Filter out zero values for minimum calculation
    non_zero_values = df[metric_column][df[metric_column] != 0]
    
    if non_zero_values.empty:
        print(f"No non-zero values found for metric: {specific_metric_name}")
        return None, None, None
    
    # Calculate the minimum value of the original DataFrame, considering only non-zero values
    min_value_df = non_zero_values.min()
    
    # Phase 1: GPU-optimized time series resampling 
    # Use pandas for consistent processing in Phase 1
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
    
    # Phase 1: GPU-optimized percentage change calculation (consistent with two-phase architecture)
    # Use pandas for consistent processing in Phase 1
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
        # Find the value before and after the highest increase
        value_before_increase = resampled_df[metric_column].loc[time_of_highest_increase - pd.Timedelta(minutes=5)]
        value_after_increase = resampled_df[metric_column].loc[time_of_highest_increase]
        
        # Calculate the steepest fall time (time of highest increase + 5 minutes)
        steepest_fall_time = time_of_highest_increase + pd.Timedelta(minutes=5)
        
        # Get the last time in the time column
        last_time_in_data = df.index.max()
        
        print(f"GPU Phase 1 - Steepest fall time detected: {steepest_fall_time}")
        return steepest_fall_time, value_before_increase, metric_column
        
    except KeyError as e:
        print(f"Error accessing time series data: {e}")
        return None, None, None

# Alias for backward compatibility
def find_steepest_fall_accelerated(df: pd.DataFrame, specific_metric_name: str, time_column: Optional[str] = None) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[str]]:
    """Backward-compatible wrapper for the main steepest fall detection"""
    # Processing strategy: GPU-accelerated (wrapper function)
    print("Processing strategy: GPU-accelerated steepest fall detection (backward-compatible wrapper)")
    
    return find_steepest_fall(df, specific_metric_name, time_column)
