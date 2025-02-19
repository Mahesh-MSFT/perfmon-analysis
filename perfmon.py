import os
import pandas as pd
import subprocess
import numpy as np

def convert_blg_to_csv(log_dir):
    """
    Convert all .blg files in the log directory to .csv files using relog.exe,
    only if the corresponding .csv file does not already exist.
    Provide status updates on the console.
    """
    blg_files = [os.path.join(root, file_name) for root, dirs, files in os.walk(log_dir) for file_name in files if file_name.endswith('.blg')]
    total_files = len(blg_files)
    converted_files = 0

    for blg_file_path in blg_files:
        csv_file_path = os.path.splitext(blg_file_path)[0] + '.csv'
        if not os.path.exists(csv_file_path):
            try:
                subprocess.run(['relog.exe', blg_file_path, '-f', 'csv', '-o', csv_file_path], check=True)
                converted_files += 1
                print(f"Converted {blg_file_path} to {csv_file_path} ({converted_files}/{total_files})")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {blg_file_path} to CSV: {e}")
        else:
            print(f"CSV file already exists for {blg_file_path}, skipping conversion. ({converted_files}/{total_files})")

def extract_perfmon_data(log_dir, metric_names, chunk_size=100000):
    data_frames = []

    for root, dirs, files in os.walk(log_dir):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                chunk_iter = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)

                for chunk in chunk_iter:
                    # Filter columns based on the metric names
                    filtered_columns = []
                    for metric_name in metric_names:
                        filtered_columns.extend(chunk.filter(like=metric_name).columns.tolist())
                    time_column = '(PDH-CSV 4.0) (GMT Standard Time)(0)'
                    if time_column in chunk.columns:
                        filtered_columns.append(time_column)
                    else:
                        print(f"'{time_column}' column not found in {file_name}")
                        continue  # Skip this chunk if 'Time' column is not found

                    filtered_chunk = chunk[filtered_columns]
                    data_frames.append(filtered_chunk)

    # Concatenate all data frames
    if data_frames:
        result_df = pd.concat(data_frames)
    else:
        result_df = pd.DataFrame()  # Return an empty DataFrame if no data is found
    return result_df

def find_steepest_fall(df, specific_metric_name):
    if df.empty:
        print("DataFrame is empty.")
        return None, None, None  # Return None if the DataFrame is empty
    
    time_column = '(PDH-CSV 4.0) (GMT Standard Time)(0)'
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

def extract_header_from_column_name(column_name):
    """
    Extract the word between the first two backslashes and the next backslash from the column name.
    """
    try:
        start = column_name.find('\\\\') + 2
        end = column_name.find('\\', start)
        header = column_name[start:end]
    except IndexError:
        header = column_name.split()[0]
    return header

def remove_first_word_after_backslashes(col):
    """
    Remove the first word following the first pair of backslashes and the backslash immediately after that.
    """
    first_backslash = col.find('\\\\')
    second_backslash = col.find('\\', first_backslash + 2)
    if second_backslash != -1:
        return col[second_backslash + 1:]
    return col

def calculate_statistics(df, metric_name, file_date_time, start_time, end_time):
    # Filter columns based on the metric name
    metric_columns = [col for col in df.columns if metric_name in col]

    # Get the header
    column_header = extract_header_from_column_name(metric_columns[0])
    
    # Ensure the data is numeric
    df.loc[:, metric_columns] = df[metric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate average and maximum values for each column
    average_values = df[metric_columns].mean()
    maximum_values = df[metric_columns].max()
    
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
            if f"{file_date_time}\n{column_header}\nAvg.\n{duration}" not in statistics_data:
                statistics_data[f"{file_date_time}\n{column_header}\nAvg.\n{duration}"] = []
            if f"{file_date_time}\n{column_header}\nMax.\n{duration}" not in statistics_data:
                statistics_data[f"{file_date_time}\n{column_header}\nMax.\n{duration}"] = []
            statistics_data[f"{file_date_time}\n{column_header}\nAvg.\n{duration}"].append((average_values.values[i] * 8) / 1_000_000)
            statistics_data[f"{file_date_time}\n{column_header}\nMax.\n{duration}"].append((maximum_values.values[i] * 8) / 1_000_000)

    statistics_df = pd.DataFrame(statistics_data)
    
    return statistics_df

# Ensure all DataFrames have the same structure before concatenating
def ensure_consistent_structure(df):
    return df

# usage
log_directory = r'C:\Path\To\All\BLGs'
metric_names = ['Request Execution Time',
         '# of Exceps Thrown', 
         '# of current logical Threads',
         '# of current physical Threads',
         'Contention Rate / sec',
         'Current Queue Length',
         'Queue Length Peak',
         '% Time in GC',
         'NumberOfActiveConnectionPools',
         'NumberOfActiveConnections',
         'NumberOfPooledConnections',
         'Total Application Pool Recycles',
         '% Managed Processor Time (estimated)',
         'Managed Memory Used (estimated)',
         'Request Wait Time',
         'Requests Failed',
         'Requests/Sec',
         'ArrivalRate',
         'CurrentQueueSize',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\Bytes Total/sec',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\Current Bandwidth',
         'Network Adapter(vmxnet3 Ethernet Adapter _2)\TCP RSC Coalesced Packets/sec',
         'NUMA Node Memory(0)\Available MBytes',
         'NUMA Node Memory(1)\Available MBytes',
         '(_Total)\% Processor Time']
baseline_metric_name = 'ASP.NET Applications(__Total__)\Request Execution Time'

# Convert .blg files to .csv files
convert_blg_to_csv(log_directory)

# Initialize a list to store all statistics DataFrames
all_statistics_list = []

# Process each CSV file separately
file_counter = 0
for root, dirs, files in os.walk(log_directory):
    csv_files = [file_name for file_name in files if file_name.endswith('.csv')]
    for file_name in csv_files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)

            # Report the CSV file being processed along with CSV files remaining to be processed
            print(f"\nProcessing file {file_counter + 1}/{len(csv_files)}: {file_path}")

            perfmon_data = pd.read_csv(file_path, low_memory=False)
            
            # Find the steepest fall for the specific metric
            if not perfmon_data.empty:
                steepest_fall_time, steepest_fall_value, column_name = find_steepest_fall(perfmon_data, baseline_metric_name)
                if steepest_fall_time and steepest_fall_value:
                    # Extract the DateTime from the steepest fall time
                    file_date_time = steepest_fall_time.strftime('%d-%b')
                    #print(f"Steepest fall found for {specific_metric_name} at {steepest_fall_time}: {steepest_fall_value}")

                    # Filter the original dataset to include only rows up to the time of the steepest fall
                    filtered_perfmon_data = perfmon_data[perfmon_data['(PDH-CSV 4.0) (GMT Standard Time)(0)'] <= steepest_fall_time]

                    # Find the first point in time in filtered_perfmon_data after sorting it and convert it to HH:MM format
                    start_time = filtered_perfmon_data['(PDH-CSV 4.0) (GMT Standard Time)(0)'].min().strftime('%H:%M')
                    
                    # Calculate average and maximum values for columns whose names include the specific metric name
                    statistics_df = calculate_statistics(filtered_perfmon_data, baseline_metric_name, file_date_time, start_time, steepest_fall_time.strftime('%H:%M'))
                    
                    # Ensure consistent structure
                    statistics_df = ensure_consistent_structure(statistics_df)
                    
                    # Append the statistics DataFrame to the list
                    all_statistics_list.append(statistics_df)
                else:
                    print(f"No steepest fall found for the specified metric: {baseline_metric_name}.")
            else:
                print(f"No data found for the specified metric: {baseline_metric_name}.")

            # Calculate statistics for other metrics
            for metric_name in metric_names:
                if metric_name == baseline_metric_name:
                    continue  # Skip the specific metric as it has already been processed
                
                # Calculate average and maximum values for columns whose names include the metric name
                statistics_df = calculate_statistics(filtered_perfmon_data, metric_name, file_date_time, start_time, steepest_fall_time.strftime('%H:%M'))
                
                # Ensure consistent structure
                statistics_df = ensure_consistent_structure(statistics_df)
                
                # Append the statistics DataFrame to the list
                all_statistics_list.append(statistics_df)
            
            file_counter += 1

# Concatenate all statistics DataFrames along the rows
if all_statistics_list:
    all_statistics_df = pd.concat(all_statistics_list, axis=0)
else:
    all_statistics_df = pd.DataFrame()

# Pivot the DataFrame to have metrics as rows and average/maximum values as columns
all_statistics_df = all_statistics_df.pivot_table(index='Metric', aggfunc='first')

# Ensure the specific metric is the first row
if baseline_metric_name in all_statistics_df.index:
    all_statistics_df = all_statistics_df.reindex([baseline_metric_name] + [idx for idx in all_statistics_df.index if idx != baseline_metric_name])

# Ensure all columns are numeric before rounding
all_statistics_df = all_statistics_df.apply(pd.to_numeric, errors='coerce')

# Round the data to a specified number of decimal places (e.g., 0 decimal places)
all_statistics_df = all_statistics_df.round(0)

# Write the combined statistics to an Excel file
output_excel_path = os.path.join(log_directory, 'combined_metrics.xlsx')
os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)  # Ensure the directory exists
all_statistics_df.to_excel(output_excel_path, index=True)
print(f"\nCombined metrics have been written to {output_excel_path}")