import pandas as pd
from modules.extract_header_from_column_name import extract_header_from_column_name
from modules.remove_first_word_after_backslashes import remove_first_word_after_backslashes


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
