import pandas as pd


def find_steepest_fall(df, specific_metric_name, time_column=None):
    if df.empty:
        print("DataFrame is empty.")
        return None, None, None  # Return None if the DataFrame is empty
    
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
    
    df[time_column] = pd.to_datetime(df[time_column])
    
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
    
    df[metric_column][df[metric_column] != 0]

    # Calculate the minimum value of the original DataFrame, considering only non-zero values
    min_value_df = df[metric_column][df[metric_column] != 0].min()

    # Resample the data to 1-minute intervals and calculate the mean
    resampled_df = df[metric_column].resample('5min').mean()

    # Filter the resampled DataFrame to include only non-zero mean values
    resampled_df = resampled_df[resampled_df != 0]

    # Sort the resampled DataFrame by the time index in descending order
    resampled_df = resampled_df.sort_index(ascending=False)

    # Convert resampled_df to a DataFrame
    resampled_df = resampled_df.to_frame(name=metric_column)

    # Calculate the percentage increase between consecutive values for the resampled DataFrame
    resampled_df['diff'] = resampled_df[metric_column].pct_change().abs() * 100

    # Sort the DataFrame based on the 'diff' column in descending order
    resampled_df = resampled_df.sort_values(by='diff', ascending=False)

    # Find the time of the highest increase
    time_of_highest_increase = resampled_df.index[0]

    # Find the value before and after the highest increase
    value_before_increase = resampled_df[metric_column].loc[time_of_highest_increase - pd.Timedelta(minutes=5)]
    value_after_increase = resampled_df[metric_column].loc[time_of_highest_increase]
    
    return time_of_highest_increase + pd.Timedelta(minutes=5), value_before_increase, value_after_increase
