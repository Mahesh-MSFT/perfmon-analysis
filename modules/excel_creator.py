import os


def excel_creator(all_statistics_df, log_directory):
    output_excel_path = os.path.join(log_directory, 'combined_metrics.xlsx')
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)  # Ensure the directory exists
    all_statistics_df.to_excel(output_excel_path, index=True)
    print(f"\nCombined metrics have been written to {output_excel_path}")