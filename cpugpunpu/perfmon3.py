# perfmon3.py - GPU-accelerated BLG conversion
# Utilizes GPU capabilities for optimal performance

import os
import sys
import pandas as pd
import gc

# Add parent directory to Python path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware detection and processing modules
from modules.hardware_detector import get_hardware_detector, print_hardware_info
from modules.convert_blg_to_csv import convert_blg_to_csv_accelerated
from modules.file_processor import file_processor_accelerated

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
        return 'gpu_only'
    else:
        return 'gpu_fallback'

def main():
    """Main execution function - Hardware-accelerated BLG conversion and CSV processing"""
    
    print("PERFMON3 - Hardware-accelerated BLG conversion and CSV processing")
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
        # Step 1: Hardware-accelerated BLG to CSV conversion
        print("\n" + "="*50)
        print("STEP 1: BLG TO CSV CONVERSION")
        print("="*50)
        
        conversion_stats = convert_blg_to_csv_accelerated(log_directory)
        
        if conversion_stats['total'] == 0:
            print("No BLG files found to process.")
            return
        
        print(f"Conversion completed: {conversion_stats['converted']} files converted")
        
        # Step 2: Hardware-accelerated CSV processing
        print("\n" + "="*50)
        print("STEP 2: CSV FILE PROCESSING")
        print("="*50)
        
        # Define metrics to analyze (same as perfmon2)
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
        
        # GPU-accelerated processing
        print("\n� GPU-ACCELERATED PROCESSING �")
        
        print("\n--- Processing with GPU Architecture (GPU Phase 1 + GPU Phase 2) ---")
        start_time_gpu = pd.Timestamp.now()
        gpu_stats_df, gpu_performance_data = file_processor_accelerated(log_directory, metric_names, baseline_metric_name)
        gpu_duration = (pd.Timestamp.now() - start_time_gpu).total_seconds()
        
        print(f"\n🏆 GPU PROCESSING RESULTS:")
        print(f"════════════════════════════════════════════════════════")
        print(f"📊 GPU ARCHITECTURE (GPU + GPU):")
        print(f"   ⏱️  Total Elapsed Time: {gpu_duration:.2f}s")
        print(f"   🔄  Phase 1 Duration: {gpu_performance_data.get('phase1_duration', 0):.2f}s")
        print(f"   🎯  Phase 2 Duration: {gpu_performance_data.get('phase2_duration', 0):.2f}s")
        print(f"   ⚡  Overlap Duration: {gpu_performance_data.get('overlap_duration', 0):.2f}s")
        print(f"   📈  Statistics Generated: {gpu_performance_data.get('statistics_generated', 0)}")
        print(f"   �  Processing Rate: {gpu_performance_data.get('gpu_processing_rate', 0):.1f} stats/sec")
        if gpu_performance_data.get('overlap_duration', 0) > 0:
            overlap_pct_gpu = (gpu_performance_data.get('overlap_duration', 0) / gpu_performance_data.get('phase1_duration', 1)) * 100
            print(f"   �  Streaming Efficiency: {overlap_pct_gpu:.1f}% overlap")
        else:
            print(f"   ⚠️  No streaming overlap detected")
        print(f"════════════════════════════════════════════════════════")
        
        # Use the GPU results
        statistics_df = gpu_stats_df
        print("\n⚡ GPU PROCESSING COMPLETE - Results ready")
        
        if not statistics_df.empty:
            print(f"Statistics calculated for {len(statistics_df)} metrics")
            print(f"Columns: {list(statistics_df.columns)}")
        else:
            print("No statistics data was generated.")
        
        # Memory cleanup
        gc.collect()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate elapsed time
    end_time = pd.Timestamp.now()
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print(f"\nTotal elapsed time: {minutes} minutes and {seconds:.2f} seconds")
    print("Hardware-accelerated processing complete.")

if __name__ == '__main__':
    main()
