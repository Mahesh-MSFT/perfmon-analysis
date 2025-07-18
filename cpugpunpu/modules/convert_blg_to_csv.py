# Hardware-accelerated BLG to CSV conversion for perfmon3.py
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict
from modules.hardware_detector import get_optimal_workers

def convert_single_blg_file(blg_file_path: str) -> Tuple[bool, str, str]:
    """
    Convert a single .blg file to .csv using relog.exe.
    Returns a tuple: (success: bool, blg_file_path: str, message: str)
    """
    # Processing strategy: CPU-based (using relog.exe system process)
    print(f"Processing strategy: CPU-based conversion for {os.path.basename(blg_file_path)}")
    
    csv_file_path = os.path.splitext(blg_file_path)[0] + '.csv'
    
    if os.path.exists(csv_file_path):
        return True, blg_file_path, f"CSV file already exists, skipping conversion."
    
    try:
        subprocess.run(['relog.exe', blg_file_path, '-f', 'csv', '-o', csv_file_path], 
                      check=True, capture_output=True)
        return True, blg_file_path, f"Converted successfully"
    except subprocess.CalledProcessError as e:
        return False, blg_file_path, f"Failed to convert: {e}"
    except Exception as e:
        return False, blg_file_path, f"Unexpected error: {e}"

def estimate_workload_size(blg_files: List[str]) -> float:
    """Estimate total workload size in GB based on BLG files"""
    # Processing strategy: CPU-based (file system operations)
    print("Processing strategy: CPU-based workload estimation")
    
    total_size = 0
    for file_path in blg_files[:3]:  # Sample first 3 files
        try:
            total_size += os.path.getsize(file_path)
        except OSError:
            pass
    
    # Estimate total size and convert to GB
    if len(blg_files) > 3:
        avg_size = total_size / min(3, len(blg_files))
        total_size = avg_size * len(blg_files)
    
    return total_size / (1024**3)

def convert_blg_to_csv_accelerated(log_dir: str) -> Dict[str, int]:
    """
    Hardware-accelerated BLG to CSV conversion with intelligent worker allocation.
    
    Args:
        log_dir: Directory containing .blg files
        
    Returns:
        Dictionary with conversion statistics
    """
    # Processing strategy: Hardware-accelerated (CPU+GPU+NPU aware)
    print("Processing strategy: Hardware-accelerated conversion with intelligent worker allocation")
    
    # Find all BLG files
    blg_files = []
    for root, dirs, files in os.walk(log_dir):
        for file_name in files:
            if file_name.endswith('.blg'):
                blg_files.append(os.path.join(root, file_name))
    
    total_files = len(blg_files)
    if total_files == 0:
        return {'total': 0, 'converted': 0, 'skipped': 0, 'failed': 0}
    
    # Get optimal worker allocation for I/O bound tasks
    workload_size_gb = estimate_workload_size(blg_files)
    worker_allocation = get_optimal_workers('io_bound', workload_size_gb)
    max_workers = max(1, worker_allocation['cpu'])
    
    # Display detailed processing strategy
    print(f"Hardware allocation: CPU workers={max_workers}, GPU workers={worker_allocation.get('gpu', 0)}, NPU workers={worker_allocation.get('npu', 0)}")
    print(f"Converting {total_files} BLG files using {max_workers} workers...")
    
    # Initialize counters
    converted_files = 0
    failed_files = 0
    skipped_files = 0
    
    start_time = time.time()
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(convert_single_blg_file, blg_file): blg_file 
            for blg_file in blg_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            blg_file = future_to_file[future]
            try:
                success, file_path, message = future.result()
                
                if success:
                    if "already exists" in message:
                        skipped_files += 1
                    else:
                        converted_files += 1
                else:
                    failed_files += 1
                
                completed = converted_files + skipped_files + failed_files
                if completed % 10 == 0 or completed == total_files:  # Progress every 10 files
                    print(f"Progress: {completed}/{total_files} files processed")
                    
            except Exception as e:
                failed_files += 1
                print(f"Error processing {blg_file}: {e}")
    
    # Calculate performance metrics
    duration = time.time() - start_time
    
    return {
        'total': total_files,
        'converted': converted_files,
        'skipped': skipped_files,
        'failed': failed_files,
        'duration': duration,
        'throughput': total_files/duration if duration > 0 else 0
    }

# Alias for backward compatibility
def convert_blg_to_csv(log_dir: str) -> Dict[str, int]:
    """Backward-compatible wrapper for the accelerated conversion function"""
    # Processing strategy: Hardware-accelerated (wrapper function)
    print("Processing strategy: Hardware-accelerated (backward-compatible wrapper)")
    
    return convert_blg_to_csv_accelerated(log_dir)
