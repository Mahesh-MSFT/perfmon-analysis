# perfmon3.py - GPU-accelerated BLG conversion
# Utilizes GPU capabilities for optimal performance

import os
import sys
import pandas as pd
import gc

# Import hardware detection and processing modules
from cpugpunpu.modules.hardware_detector import get_hardware_detector, print_hardware_info
from cpugpunpu.modules.convert_blg_to_csv import convert_blg_to_csv
from cpugpunpu.modules.file_processor import file_processor

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
    
    if hardware.profile.gpu:
        return 'gpu_accelerated'
    else:
        return 'gpu_fallback'

def main():
    """Main execution function - Hardware-accelerated BLG conversion and CSV processing"""
    
    # Log Start Date and Time
    start_time = pd.Timestamp.now()
    print("Script started at:", start_time)
    
    try:
        # Hardware validation
        if not validate_hardware_requirements():
            print("WARNING: System may not meet minimum requirements for optimal performance")
        
        # Hardware detection and display
        hardware = get_hardware_detector()
        print_hardware_info()
        
        # Choose processing strategy
        strategy = choose_processing_strategy()
        print(f"Processing strategy: {strategy}")
        
        # Convert .blg files to .csv files
        convert_blg_to_csv(log_directory)
        
        # Define metrics to process (same as perfmon2 for consistency)
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
        
        baseline_metric_name = 'ASP.NET Applications(__Total__)\Request Execution Time'  # Same as perfmon2
        
        # Process the CSV files
        statistics_df, performance_data = file_processor(log_directory, metric_names, baseline_metric_name)
        
        if statistics_df.empty:
           print("No statistics data was generated.")
        
        # Explicitly clear large DataFrame from memory
        del statistics_df
        
        # Force garbage collection to free up memory
        gc.collect()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    
    # Format elapsed time in minutes and seconds
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("Script completed at:", end_time)
    print(f"Total elapsed time: {minutes} minutes and {seconds:.2f} seconds")

if __name__ == '__main__':
    main()
