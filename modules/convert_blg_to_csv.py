import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_single_blg_file(blg_file_path):
    """
    Convert a single .blg file to .csv using relog.exe.
    Returns a tuple: (success: bool, blg_file_path: str, message: str)
    """
    csv_file_path = os.path.splitext(blg_file_path)[0] + '.csv'
    
    if os.path.exists(csv_file_path):
        return True, blg_file_path, f"CSV file already exists for {blg_file_path}, skipping conversion."
    
    try:
        subprocess.run(['relog.exe', blg_file_path, '-f', 'csv', '-o', csv_file_path], check=True)
        return True, blg_file_path, f"Converted {blg_file_path} to {csv_file_path}"
    except subprocess.CalledProcessError as e:
        return False, blg_file_path, f"Failed to convert {blg_file_path} to CSV: {e}"

def convert_blg_to_csv(log_dir):
    """
    Convert all .blg files in the log directory to .csv files using relog.exe in parallel.
    """
    blg_files = [os.path.join(root, file_name) 
                 for root, dirs, files in os.walk(log_dir) 
                 for file_name in files if file_name.endswith('.blg')]
    
    total_files = len(blg_files)
    converted_files = 0
    failed_files = 0
    skipped_files = 0
    
    print(f"Found {total_files} .blg files to process")
    
    if not blg_files:
        print("No .blg files found.")
        return
    
    # Calculate max_workers as 75% of available CPU cores
    max_workers = max(1, round(os.cpu_count() * 0.75))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(convert_single_blg_file, blg_file): blg_file 
                          for blg_file in blg_files}
        
        for future in as_completed(future_to_file):
            blg_file = future_to_file[future]
            try:
                success, file_path, message = future.result()
                
                # No lock needed - this runs sequentially in main process
                if success:
                    if "already exists" in message:
                        skipped_files += 1
                    else:
                        converted_files += 1
                else:
                    failed_files += 1
                
                completed = converted_files + skipped_files + failed_files
                print(f"[{completed}/{total_files}] {message}")
                    
            except Exception as e:
                failed_files += 1
                completed = converted_files + skipped_files + failed_files
                print(f"[{completed}/{total_files}] Unexpected error processing {blg_file}: {e}")
    
    print(f"\nConversion complete! Converted: {converted_files}, Skipped: {skipped_files}, Failed: {failed_files}")