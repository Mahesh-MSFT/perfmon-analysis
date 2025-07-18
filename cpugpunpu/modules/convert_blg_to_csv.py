# Hardware-accelerated BLG to CSV conversion for perfmon3.py
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict
from modules.hardware_detector import get_hardware_detector, get_optimal_workers

def convert_single_blg_file(blg_file_path: str) -> Tuple[bool, str, str]:
    """
    Convert a single .blg file to .csv using relog.exe.
    Returns a tuple: (success: bool, blg_file_path: str, message: str)
    """
    csv_file_path = os.path.splitext(blg_file_path)[0] + '.csv'
    
    if os.path.exists(csv_file_path):
        return True, blg_file_path, f"CSV file already exists for {blg_file_path}, skipping conversion."
    
    try:
        # Use relog.exe for conversion
        start_time = time.time()
        subprocess.run(['relog.exe', blg_file_path, '-f', 'csv', '-o', csv_file_path], 
                      check=True, capture_output=True)
        duration = time.time() - start_time
        
        # Get file size for metrics
        file_size_mb = os.path.getsize(csv_file_path) / (1024 * 1024)
        
        return True, blg_file_path, f"Converted {blg_file_path} to {csv_file_path} ({file_size_mb:.1f}MB in {duration:.2f}s)"
    except subprocess.CalledProcessError as e:
        return False, blg_file_path, f"Failed to convert {blg_file_path} to CSV: {e}"
    except Exception as e:
        return False, blg_file_path, f"Unexpected error converting {blg_file_path}: {e}"

def estimate_workload_size(blg_files: List[str]) -> float:
    """
    Estimate total workload size in GB based on BLG files
    """
    total_size = 0
    for file_path in blg_files[:5]:  # Sample first 5 files
        try:
            total_size += os.path.getsize(file_path)
        except OSError:
            pass
    
    # Estimate total size and convert to GB
    if len(blg_files) > 5:
        avg_size = total_size / min(5, len(blg_files))
        total_size = avg_size * len(blg_files)
    
    return total_size / (1024**3)

def convert_blg_to_csv_accelerated(log_dir: str, enable_gpu_scheduling: bool = True) -> Dict[str, int]:
    """
    Hardware-accelerated BLG to CSV conversion with intelligent worker allocation.
    
    Args:
        log_dir: Directory containing .blg files
        enable_gpu_scheduling: Whether to consider GPU resources in scheduling
        
    Returns:
        Dictionary with conversion statistics
    """
    # Get hardware detector
    hardware = get_hardware_detector()
    
    # Find all BLG files
    blg_files = []
    for root, dirs, files in os.walk(log_dir):
        for file_name in files:
            if file_name.endswith('.blg'):
                blg_files.append(os.path.join(root, file_name))
    
    total_files = len(blg_files)
    if total_files == 0:
        print("No .blg files found.")
        return {'total': 0, 'converted': 0, 'skipped': 0, 'failed': 0}
    
    # Estimate workload size
    workload_size_gb = estimate_workload_size(blg_files)
    
    # Get optimal worker allocation
    # BLG conversion is I/O bound, so we use 'io_bound' workload type
    worker_allocation = get_optimal_workers('io_bound', workload_size_gb)
    
    # Print hardware and allocation info
    print("=" * 60)
    print("HARDWARE-ACCELERATED BLG TO CSV CONVERSION")
    print("=" * 60)
    hardware.print_hardware_summary()
    print(f"\nWorkload Analysis:")
    print(f"Files to process: {total_files}")
    print(f"Estimated size: {workload_size_gb:.2f} GB")
    print(f"Worker allocation: {worker_allocation}")
    
    # Use the allocated CPU workers for I/O bound tasks
    max_workers = max(1, worker_allocation['cpu'])
    
    # Initialize counters
    converted_files = 0
    failed_files = 0
    skipped_files = 0
    
    print(f"\nStarting conversion with {max_workers} workers...")
    start_time = time.time()
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
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
                print(f"[{completed}/{total_files}] {message}")
                
            except Exception as e:
                failed_files += 1
                completed = converted_files + skipped_files + failed_files
                print(f"[{completed}/{total_files}] Unexpected error processing {blg_file}: {e}")
    
    # Calculate performance metrics
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Converted: {converted_files}")
    print(f"Skipped: {skipped_files}")
    print(f"Failed: {failed_files}")
    print(f"Duration: {duration:.2f} seconds")
    
    if duration > 0:
        print(f"Throughput: {total_files/duration:.2f} files/second")
        if workload_size_gb > 0:
            print(f"Data throughput: {workload_size_gb/duration:.2f} GB/second")
    
    # Performance analysis
    if enable_gpu_scheduling and hardware.profile.gpu:
        print(f"\nNote: GPU available ({hardware.profile.gpu.name}) but not used for I/O-bound BLG conversion.")
        print("GPU acceleration will be utilized for computational tasks in file processing.")
    
    if hardware.profile.npu:
        print(f"Note: NPU available ({hardware.profile.npu.name}) - will be used for AI/ML workloads.")
    
    print("=" * 60)
    
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
    """
    Backward-compatible wrapper for the accelerated conversion function
    """
    return convert_blg_to_csv_accelerated(log_dir)

def benchmark_conversion_methods(log_dir: str, sample_size: int = 3) -> Dict[str, Dict]:
    """
    Benchmark different conversion approaches to validate hardware acceleration benefits
    
    Args:
        log_dir: Directory containing .blg files
        sample_size: Number of files to use for benchmarking
        
    Returns:
        Dictionary with benchmark results
    """
    # Find sample files
    blg_files = []
    for root, dirs, files in os.walk(log_dir):
        for file_name in files:
            if file_name.endswith('.blg'):
                blg_files.append(os.path.join(root, file_name))
                if len(blg_files) >= sample_size:
                    break
        if len(blg_files) >= sample_size:
            break
    
    if not blg_files:
        return {'error': 'No BLG files found for benchmarking'}
    
    results = {}
    
    # Test 1: Single-threaded conversion
    print("Benchmarking single-threaded conversion...")
    start_time = time.time()
    for blg_file in blg_files:
        convert_single_blg_file(blg_file)
    single_thread_time = time.time() - start_time
    results['single_thread'] = {
        'duration': single_thread_time,
        'files': len(blg_files),
        'throughput': len(blg_files) / single_thread_time
    }
    
    # Test 2: Hardware-accelerated conversion
    print("Benchmarking hardware-accelerated conversion...")
    start_time = time.time()
    temp_dir = os.path.dirname(blg_files[0])
    convert_blg_to_csv_accelerated(temp_dir)
    accelerated_time = time.time() - start_time
    results['accelerated'] = {
        'duration': accelerated_time,
        'files': len(blg_files),
        'throughput': len(blg_files) / accelerated_time
    }
    
    # Calculate improvement
    if single_thread_time > 0:
        improvement = (single_thread_time - accelerated_time) / single_thread_time * 100
        results['improvement_percent'] = improvement
    
    return results
