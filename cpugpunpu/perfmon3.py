# perfmon3.py - Hardware-accelerated performance monitoring analysis
# Utilizes CPU, GPU, and NPU capabilities for optimal performance

import os
import sys
import pandas as pd
import numpy as np
import gc
import time
from typing import Dict, List, Optional

# Add parent directory to Python path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware detection
from modules.hardware_detector import get_hardware_detector, print_hardware_info
from modules.convert_blg_to_csv import convert_blg_to_csv_accelerated

# Import shared modules
from shared.modules.excel_creator import excel_creator

# TODO: Import accelerated versions of these modules once created
# from modules.file_processor_accelerated import file_processor_accelerated
# from modules.accelerated_statistics import calculate_statistics_accelerated

# Configuration
log_directory = r'C:\Users\maksh\OneDrive - Microsoft\Documents\AVS\PerfTest\ParallelTesting'

# Performance metrics to analyze
metric_names = [
    'Request Execution Time',
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
    'Network Adapter(vmxnet3 Ethernet Adapter _2)\\Bytes Total/sec',
    'Network Adapter(vmxnet3 Ethernet Adapter _2)\\Current Bandwidth',
    'Network Adapter(vmxnet3 Ethernet Adapter _2)\\TCP RSC Coalesced Packets/sec',
    'NUMA Node Memory(0)\\Available MBytes',
    'NUMA Node Memory(1)\\Available MBytes',
    '(_Total)\\% Processor Time'
]

baseline_metric_name = 'ASP.NET Applications(__Total__)\\Request Execution Time'

def print_performance_analysis_header():
    """Print the analysis header with hardware information"""
    print("=" * 80)
    print("PERFMON3 - HARDWARE-ACCELERATED PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis started at: {pd.Timestamp.now()}")
    print(f"Log directory: {log_directory}")
    print(f"Metrics to analyze: {len(metric_names)}")
    print(f"Baseline metric: {baseline_metric_name}")
    print("=" * 80)

def validate_hardware_requirements() -> bool:
    """
    Validate that the system has the required hardware capabilities
    
    Returns:
        bool: True if system meets requirements, False otherwise
    """
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
    """
    Choose the optimal processing strategy based on available hardware
    
    Returns:
        str: Processing strategy ('cpu_only', 'cpu_gpu', 'cpu_gpu_npu')
    """
    hardware = get_hardware_detector()
    
    if hardware.profile.npu and hardware.profile.gpu:
        return 'cpu_gpu_npu'
    elif hardware.profile.gpu:
        return 'cpu_gpu'
    else:
        return 'cpu_only'

def main():
    """Main execution function"""
    
    # Print header and hardware information
    print_performance_analysis_header()
    print_hardware_info()
    
    # Validate hardware requirements
    if not validate_hardware_requirements():
        print("System does not meet minimum hardware requirements!")
        print("Consider upgrading hardware or using the CPU-only version (perfmon2.py)")
        return
    
    # Choose processing strategy
    strategy = choose_processing_strategy()
    print(f"\nSelected processing strategy: {strategy}")
    
    # Record start time
    start_time = pd.Timestamp.now()
    
    try:
        # Phase 1: Hardware-accelerated BLG to CSV conversion
        print("\n" + "=" * 80)
        print("PHASE 1: BLG TO CSV CONVERSION")
        print("=" * 80)
        
        conversion_stats = convert_blg_to_csv_accelerated(log_directory)
        
        if conversion_stats['total'] == 0:
            print("No BLG files found to process. Exiting.")
            return
        
        # Phase 2: Hardware-accelerated file processing
        print("\n" + "=" * 80)
        print("PHASE 2: ACCELERATED FILE PROCESSING")
        print("=" * 80)
        
        # TODO: Implement accelerated file processing
        # For now, fall back to CPU-only processing
        print("Accelerated file processing not yet implemented.")
        print("Falling back to CPU-only processing...")
        
        # Import CPU-only file processor as fallback
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpuonly'))
        from modules.file_processor import file_processor
        
        all_statistics_df = file_processor(log_directory, metric_names, baseline_metric_name)
        
        # Phase 3: Results export
        print("\n" + "=" * 80)
        print("PHASE 3: RESULTS EXPORT")
        print("=" * 80)
        
        if not all_statistics_df.empty:
            excel_creator(all_statistics_df, log_directory)
            print(f"Results exported successfully!")
        else:
            print("No statistics data generated.")
        
        # Memory cleanup
        if 'all_statistics_df' in locals():
            del all_statistics_df
        gc.collect()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate and display performance metrics
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Processing strategy: {strategy}")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total elapsed time: {minutes} minutes and {seconds:.2f} seconds")
    
    if 'conversion_stats' in locals():
        print(f"BLG files processed: {conversion_stats['total']}")
        print(f"Conversion throughput: {conversion_stats.get('throughput', 0):.2f} files/second")
    
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
