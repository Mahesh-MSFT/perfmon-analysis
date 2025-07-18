# perfmon3.py - Hardware-accelerated BLG conversion
# Utilizes CPU, GPU, and NPU capabilities for optimal performance

import os
import sys
import pandas as pd
import gc

# Add parent directory to Python path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware detection and conversion
from modules.hardware_detector import get_hardware_detector, print_hardware_info
from modules.convert_blg_to_csv import convert_blg_to_csv_accelerated

# Configuration
log_directory = r'C:\Users\maksh\OneDrive - Microsoft\Documents\AVS\PerfTest\ParallelTesting'

def validate_hardware_requirements() -> bool:
    """Validate that the system has the required hardware capabilities"""
    hardware = get_hardware_detector()
    
    # Minimum requirements
    min_cpu_cores = 4
    min_memory_gb = 8
    
    if hardware.profile.cpu.cores < min_cpu_cores:
        print(f"WARNING: System has {hardware.profile.cpu.cores} CPU cores, minimum recommended is {min_cpu_cores}")
        return False
    
    if hardware.profile.available_memory_gb < min_memory_gb:
        print(f"WARNING: System has {hardware.profile.available_memory_gb:.1f}GB available memory, minimum recommended is {min_memory_gb}GB")
        return False
    
    return True

def choose_processing_strategy() -> str:
    """Choose the optimal processing strategy based on available hardware"""
    hardware = get_hardware_detector()
    
    if hardware.profile.npu and hardware.profile.gpu:
        return 'cpu_gpu_npu'
    elif hardware.profile.gpu:
        return 'cpu_gpu'
    else:
        return 'cpu_only'

def main():
    """Main execution function - Hardware-accelerated BLG to CSV conversion"""
    
    print("PERFMON3 - Hardware-accelerated BLG conversion")
    print(f"Log directory: {log_directory}")
    
    # Print hardware information
    print_hardware_info()
    
    # Validate hardware requirements
    if not validate_hardware_requirements():
        print("System does not meet minimum hardware requirements!")
        return
    
    # Choose processing strategy
    strategy = choose_processing_strategy()
    print(f"Processing strategy: {strategy}")
    
    # Record start time
    start_time = pd.Timestamp.now()
    
    try:
        # Hardware-accelerated BLG to CSV conversion
        conversion_stats = convert_blg_to_csv_accelerated(log_directory)
        
        if conversion_stats['total'] == 0:
            print("No BLG files found to process.")
            return
        
        # Show results
        print(f"\nConversion completed:")
        print(f"  Total files: {conversion_stats['total']}")
        print(f"  Converted: {conversion_stats['converted']}")
        print(f"  Skipped: {conversion_stats['skipped']}")
        print(f"  Failed: {conversion_stats['failed']}")
        print(f"  Throughput: {conversion_stats.get('throughput', 0):.2f} files/second")
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        print(f"Error during BLG conversion: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate elapsed time
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print(f"\nTotal elapsed time: {minutes} minutes and {seconds:.2f} seconds")
    print("BLG conversion complete.")

if __name__ == '__main__':
    main()
