import os

def excel_creator(all_statistics_df, log_directory):
    # Check if DataFrame has any data
    if all_statistics_df.empty:
        print("\nNo data to export - DataFrame is empty")
        return
    
    # Check if log directory exists
    if not os.path.exists(log_directory):
        print(f"\nLog directory does not exist: {log_directory}")
        return
    
    output_excel_path = os.path.join(log_directory, 'combined_metrics.xlsx')
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)  # Ensure the directory exists
    all_statistics_df.to_excel(output_excel_path, index=True)
    print(f"\nCombined metrics have been written to {output_excel_path}")