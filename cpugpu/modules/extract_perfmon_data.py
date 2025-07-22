import os
import pandas as pd

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
                    
                    # Detect time column dynamically
                    time_column = None
                    for column in chunk.columns:
                        if (column.startswith('(PDH-CSV 4.0) (') and 
                            'Time)(' in column and 
                            column.endswith(')')):
                            time_column = column
                            break
                    
                    if time_column is None and len(chunk.columns) > 0:
                        time_column = chunk.columns[0]  # Fallback to first column
                    
                    if time_column in chunk.columns:
                        filtered_columns.append(time_column)
                    else:
                        print(f"No valid time column found in {file_name}")
                        continue  # Skip this chunk if no time column is found

                    filtered_chunk = chunk[filtered_columns]
                    data_frames.append(filtered_chunk)

    # Concatenate all data frames
    if data_frames:
        result_df = pd.concat(data_frames)
    else:
        result_df = pd.DataFrame()  # Return an empty DataFrame if no data is found
    return result_df
