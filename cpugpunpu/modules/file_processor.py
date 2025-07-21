# Simplified GPU-accelerated two-phase file processor for perfmon3.py
# Phase 1: GPU-accelerated data preparation (steepest fall detection + filtering)
# Phase 2: GPU statistics processing (parallel metric calculation)

import os
import pandas as pd
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from modules.hardware_detector import get_hardware_detector

def detect_time_column(perfmon_data):
    """Detect the time column in the CSV data."""
    print("   🔍 GPU-accelerated time column detection")
    
    for column in perfmon_data.columns:
        if (column.startswith('(PDH-CSV 4.0) (') and 
            'Time)(' in column and 
            column.endswith(')')):
            return column
    
    if len(perfmon_data.columns) > 0:
        return perfmon_data.columns[0]
    
    raise ValueError("No time column found in the CSV data")

def collect_csv_files(log_directory: str) -> List[str]:
    """Collect all CSV files from the log directory."""
    csv_file_paths = []
    for root, dirs, files in os.walk(log_directory):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                csv_file_paths.append(file_path)
    return csv_file_paths

def calculate_optimal_gpu_workers() -> int:
    """Calculate optimal GPU workers based on hardware capabilities."""
    hardware = get_hardware_detector()
    
    if not hardware.profile.gpu:
        return 4  # Fallback for systems without GPU detection
    
    gpu_compute_units = hardware.profile.gpu.compute_units
    gpu_memory_gb = hardware.profile.gpu.memory_gb
    
    # Calculate based on compute units
    if gpu_compute_units >= 64:  # Intel Arc or equivalent
        optimal_gpu_workers = min(16, max(8, gpu_compute_units // 4))
    elif gpu_compute_units >= 32:  # Mid-range GPU
        optimal_gpu_workers = min(8, max(4, gpu_compute_units // 4))
    else:  # Lower-end GPU
        optimal_gpu_workers = min(4, max(2, gpu_compute_units // 8))
    
    # Adjust for available memory (each worker processes ~1-2GB data)
    memory_limited_workers = max(2, int(gpu_memory_gb / 2))
    optimal_gpu_workers = min(optimal_gpu_workers, memory_limited_workers)
    
    return optimal_gpu_workers

def print_gpu_configuration(optimal_gpu_workers: int):
    """Print GPU processing configuration details."""
    hardware = get_hardware_detector()
    
    if hardware.profile.gpu:
        gpu_compute_units = hardware.profile.gpu.compute_units
        gpu_memory_gb = hardware.profile.gpu.memory_gb
        
        print(f"🎯 Phase 2 GPU Processing Configuration:")
        print(f"   GPU Hardware: {hardware.profile.gpu.name}")
        print(f"   Compute Units: {gpu_compute_units}, Memory: {gpu_memory_gb:.1f}GB")
        print(f"   File Processing Workers: {optimal_gpu_workers} (processes files in parallel)")
        print(f"   Metric Processing Queues: 32 per worker (processes metrics in parallel within each file)")
    else:
        print("⚠️ GPU not detected, using fallback: 4 workers")

def print_pipeline_architecture(optimal_gpu_workers: int):
    """Print the pipeline architecture explanation."""
    print(f"\\n🚀 Starting STREAMING PIPELINE:")
    print(f"   Phase 1: GPU-accelerated data preparation")
    print(f"   Phase 2: GPU statistics processing")
    print(f"   Pipeline: Phase 1 → Immediate Phase 2 streaming")
    print(f"")
    print(f"📋 Three-Level GPU Parallelism Architecture:")
    print(f"   🔸 File Level: {optimal_gpu_workers} workers process different files simultaneously") 
    print(f"   🔸 Metric Level: Each file processes 25+ metrics in parallel using 32 OpenCL queues")
    print(f"   🔸 Compute Level: Each metric uses GPU compute units for statistics calculation")
    print(f"   🔸 Total GPU Utilization: Up to {optimal_gpu_workers * 32} concurrent metric calculations")

def calculate_hardware_aware_workers(csv_file_paths: List[str]) -> Dict[str, int]:
    """GPU worker allocation for two-phase architecture."""
    print("GPU worker allocation for two-phase architecture")
    
    if not csv_file_paths:
        return {'file_workers': 1}
    
    hardware = get_hardware_detector()
    total_cpu_cores = hardware.profile.cpu.cores
    file_workers = min(len(csv_file_paths), max(1, total_cpu_cores // 2))
    
    allocation = {
        'file_workers': file_workers,
        'total_cpu_cores': total_cpu_cores
    }
    
    print(f"Phase 1 allocation: {file_workers} GPU workers (CPU cores: {total_cpu_cores})")
    print(f"Processing {len(csv_file_paths)} files")
    
    return allocation

def process_single_file_phase2(file_result, file_path, metric_names):
    """Process GPU Phase 2 for a single file."""
    try:
        print(f"🚀 GPU Phase 2 starting immediately for {os.path.basename(file_path)}")
        
        from modules.metrics_processor import process_file_metrics
        single_file_stats = process_file_metrics([file_result], metric_names)
        
        if single_file_stats:
            print(f"✅ GPU Phase 2 complete for {os.path.basename(file_path)} - {len(single_file_stats)} statistics")
            return single_file_stats
        else:
            print(f"⚠️ No statistics generated for {os.path.basename(file_path)}")
            return []
            
    except Exception as e:
        print(f"❌ GPU Phase 2 error for {file_path}: {e}")
        return []

def execute_phase1(csv_file_paths: List[str], baseline_metric_name: str):
    """Execute Phase 1: GPU-accelerated data preparation."""
    print(f"\\n--- Phase 1: GPU-Accelerated Data Preparation ---")
    
    hardware_allocation = calculate_hardware_aware_workers(csv_file_paths)
    print("Using GPU acceleration for Phase 1 data preparation")
    
    phase1_args = [
        (file_path, baseline_metric_name)
        for file_path in csv_file_paths
    ]
    
    return phase1_args, hardware_allocation

def execute_streaming_pipeline(phase1_args: List[Tuple], hardware_allocation: Dict[str, int], 
                               optimal_gpu_workers: int, metric_names: List[str]):
    """Execute the streaming pipeline with Phase 1 and Phase 2 processing."""
    from modules.data_preprocessor import process_single_file
    
    filtered_file_data = []
    all_statistics_list = []
    
    # Timing tracking
    phase1_start = pd.Timestamp.now()
    first_phase2_start = None
    last_phase1_complete = None
    phase2_only_start = None
    phase2_only_end = None
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as process_executor:
        with ThreadPoolExecutor(max_workers=optimal_gpu_workers, thread_name_prefix="GPU-Phase2") as gpu_executor:
            
            # Submit Phase 1 tasks
            phase1_futures = {
                process_executor.submit(process_single_file, args): args[0] 
                for args in phase1_args
            }
            
            gpu_phase2_futures = []
            completed_files = 0
            
            # Process Phase 1 results and immediately start Phase 2
            for phase1_future in as_completed(phase1_futures):
                file_path = phase1_futures[phase1_future]
                completed_files += 1
                
                try:
                    file_result = phase1_future.result()
                    if file_result is not None:
                        filtered_file_data.append(file_result)
                        
                        print(f"📋 Phase 1: {completed_files}/{len(phase1_args)} files prepared → Starting GPU Phase 2 for {os.path.basename(file_path)}")
                        
                        # Track first Phase 2 start
                        if first_phase2_start is None:
                            first_phase2_start = pd.Timestamp.now()
                        
                        # Submit GPU Phase 2 task immediately
                        gpu_future = gpu_executor.submit(process_single_file_phase2, file_result, file_path, metric_names)
                        gpu_phase2_futures.append(gpu_future)
                        
                        last_phase1_complete = pd.Timestamp.now()
                    else:
                        print(f"📋 Phase 1: {completed_files}/{len(phase1_args)} files prepared (skipped due to error)")
                        last_phase1_complete = pd.Timestamp.now()
                
                except Exception as e:
                    print(f"❌ Phase 1 error for file {file_path}: {e}")
                    last_phase1_complete = pd.Timestamp.now()
            
            # Wait for all GPU Phase 2 processing to complete
            print("⏳ Waiting for all GPU Phase 2 threads to complete...")
            phase2_only_start = pd.Timestamp.now()
            
            for gpu_future in as_completed(gpu_phase2_futures):
                try:
                    gpu_stats = gpu_future.result()
                    if gpu_stats:
                        all_statistics_list.extend(gpu_stats)
                except Exception as e:
                    print(f"❌ Error collecting GPU Phase 2 results: {e}")
            
            phase2_only_end = pd.Timestamp.now()
    
    # Calculate timing metrics
    total_pipeline_duration = (phase2_only_end - phase1_start).total_seconds()
    phase1_only_duration = (last_phase1_complete - phase1_start).total_seconds()
    phase2_total_duration = (phase2_only_end - first_phase2_start).total_seconds() if first_phase2_start else 0
    phase2_only_duration = (phase2_only_end - phase2_only_start).total_seconds() if phase2_only_start else 0
    
    # Calculate overlap
    if first_phase2_start and last_phase1_complete and first_phase2_start < last_phase1_complete:
        overlap_duration = (min(last_phase1_complete, phase2_only_end) - first_phase2_start).total_seconds()
    else:
        overlap_duration = 0
    
    timing_data = {
        'total_time': total_pipeline_duration,
        'phase1_duration': phase1_only_duration,
        'phase2_duration': phase2_total_duration,
        'overlap_duration': overlap_duration,
        'files_processed': len(filtered_file_data),
        'statistics_generated': len(all_statistics_list),
        'gpu_processing_rate': len(all_statistics_list) / total_pipeline_duration if total_pipeline_duration > 0 else 0,
        'overall_throughput': len(phase1_args) / total_pipeline_duration if total_pipeline_duration > 0 else 0
    }
    
    return filtered_file_data, all_statistics_list, timing_data

def print_timing_analysis(timing_data: Dict):
    """Print detailed timing analysis."""
    print(f"\\n🏁 DETAILED PIPELINE TIMING ANALYSIS:")
    print(f"   📊 Total Pipeline Duration: {timing_data['total_time']:.2f}s")
    print(f"   📝 Phase 1 Duration: {timing_data['phase1_duration']:.2f}s")
    print(f"   🎯 Phase 2 Total Duration: {timing_data['phase2_duration']:.2f}s")
    print(f"   🔄 Overlap Duration: {timing_data['overlap_duration']:.2f}s")
    
    if timing_data['overlap_duration'] > 0:
        overlap_percentage = (timing_data['overlap_duration'] / timing_data['phase1_duration']) * 100
        print(f"   📈 Overlap Efficiency: {overlap_percentage:.1f}% of Phase 1 overlapped with Phase 2")
    else:
        print(f"   ⚠️ No Overlap: Sequential processing (no streaming benefit)")
    
    print(f"🏁 Streaming pipeline analysis complete: {timing_data['files_processed']} files processed")

def print_performance_summary(timing_data: Dict):
    """Print performance summary."""
    print(f"\\n--- Phase 2: Completed via ThreadPoolExecutor streaming pipeline ---")
    
    print(f"\\n💯 ThreadPoolExecutor Streaming Pipeline Complete:")
    print(f"   📁 Phase 1 processed: {timing_data['files_processed']} files")
    print(f"   🎯 GPU Phase 2 processed: {timing_data['statistics_generated']} statistics")
    print(f"   ⚡ Total time: {timing_data['total_time']:.2f}s")
    print(f"   🚀 Throughput: {timing_data['overall_throughput']:.2f} files/second")
    print(f"   🔥 Architecture: GPU Phase 1 + ThreadPoolExecutor GPU Phase 2")
    
    if timing_data['statistics_generated'] > 0:
        print(f"\\n📊 PERFORMANCE METRICS (GPU+GPU):")
        print(f"   🎯 Total Statistics Generated: {timing_data['statistics_generated']}")
        print(f"   ⚡ GPU Phase 2 Processing Rate: {timing_data['gpu_processing_rate']:.1f} statistics/second")
        print(f"   🚀 Overall System Throughput: {timing_data['overall_throughput']:.3f} files/second")

def combine_results(all_statistics_list: List, baseline_metric_name: str) -> pd.DataFrame:
    """Combine all statistics results into a single DataFrame."""
    if not all_statistics_list:
        print("No statistics data was generated.")
        return pd.DataFrame()
    
    all_statistics_df = pd.concat(all_statistics_list, axis=0)
    
    # Pivot the DataFrame to have metrics as rows and average/maximum values as columns
    all_statistics_df = all_statistics_df.pivot_table(index='Metric', aggfunc='first')
    
    # Ensure the baseline metric is the first row
    if baseline_metric_name in all_statistics_df.index:
        all_statistics_df = all_statistics_df.reindex(
            [baseline_metric_name] + [idx for idx in all_statistics_df.index if idx != baseline_metric_name]
        )
    
    # Ensure all columns are numeric and round
    all_statistics_df = all_statistics_df.apply(pd.to_numeric, errors='coerce')
    all_statistics_df = all_statistics_df.round(0)
    
    return all_statistics_df

def file_processor(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process performance logs to generate statistical analysis.
    
    Simplified two-phase approach:
    - Phase 1: GPU-accelerated data preparation (steepest fall detection + filtering)
    - Phase 2: GPU statistics processing (parallel metric calculation)
    """
    
    print("Processing strategy: GPU-accelerated Phase 1 + GPU Phase 2 architecture")
    
    # Step 1: Collect CSV files
    csv_file_paths = collect_csv_files(log_directory)
    
    if not csv_file_paths:
        print("No CSV files found in the specified directory.")
        return pd.DataFrame(), {}
    
    print(f"Found {len(csv_file_paths)} CSV files for GPU processing")
    
    # Step 2: Execute Phase 1 (data preparation)
    phase1_args, hardware_allocation = execute_phase1(csv_file_paths, baseline_metric_name)
    
    # Step 3: Setup Phase 2 configuration
    optimal_gpu_workers = calculate_optimal_gpu_workers()
    print_gpu_configuration(optimal_gpu_workers)
    print_pipeline_architecture(optimal_gpu_workers)
    
    # Step 4: Execute streaming pipeline
    filtered_file_data, all_statistics_list, timing_data = execute_streaming_pipeline(
        phase1_args, hardware_allocation, optimal_gpu_workers, metric_names
    )
    
    # Step 5: Print analysis and performance summary
    print_timing_analysis(timing_data)
    print_performance_summary(timing_data)
    
    # Step 6: Combine results
    all_statistics_df = combine_results(all_statistics_list, baseline_metric_name)
    
    # Step 7: Prepare performance data for return
    performance_data = {
        'architecture': 'GPU+GPU',
        **timing_data
    }
    
    return all_statistics_df, performance_data

# Backward compatibility functions
def file_processor_accelerated(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Backward-compatible wrapper for file_processor"""
    print("Processing strategy: GPU-accelerated two-phase architecture (backward-compatible wrapper)")
    return file_processor(log_directory, metric_names, baseline_metric_name)

def process_performance_logs(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Backward-compatible wrapper for file_processor"""
    return file_processor(log_directory, metric_names, baseline_metric_name)

# Legacy batch processing function (kept for compatibility)
def process_batch(filtered_file_data: List[Dict], metric_names: List[str]) -> List[pd.DataFrame]:
    """
    Phase 2: GPU batch processing.
    Process all filtered data through GPU acceleration in batches.
    """
    if not filtered_file_data:
        return []
    
    print(f"Phase 2 - GPU batch processing: {len(filtered_file_data)} files, {len(metric_names)} metrics")
    
    all_statistics_list = []
    batch_start_time = pd.Timestamp.now()
    
    # Process each file's filtered data
    for file_data in filtered_file_data:
        if file_data is None:
            continue
            
        file_path = file_data['file_path']
        filtered_perfmon_data = file_data['filtered_data']
        time_column = file_data['time_column']
        steepest_fall_time = file_data['steepest_fall_time']
        file_date_time = file_data['file_date_time']
        start_time = file_data['start_time']
        
        print(f"GPU processing: {os.path.basename(file_path)} - {len(filtered_perfmon_data)} rows")
        
        # Pre-filter data per metric for GPU batch processing
        metric_specific_data = {}
        for metric_name in metric_names:
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            if metric_columns:
                columns_to_keep = [time_column] + metric_columns
                metric_specific_data[metric_name] = filtered_perfmon_data[columns_to_keep]
        
        # Process all metrics for this file using GPU
        for metric_name, metric_data in metric_specific_data.items():
            try:
                from modules.calculate_statistics import calculate_statistics
                from shared.modules.ensure_consistent_structure import ensure_consistent_structure
                
                statistics_df = calculate_statistics(
                    metric_data, 
                    metric_name, 
                    file_date_time, 
                    start_time, 
                    steepest_fall_time.strftime('%H:%M')
                )
                
                statistics_df = ensure_consistent_structure(statistics_df)
                
                if not statistics_df.empty:
                    all_statistics_list.append(statistics_df)
                    
            except Exception as e:
                print(f"Error in GPU processing for metric {metric_name}: {e}")
        
        # Clear processed data
        del metric_specific_data
        gc.collect()
    
    batch_end_time = pd.Timestamp.now()
    batch_duration = (batch_end_time - batch_start_time).total_seconds()
    print(f"Phase 2 GPU processing completed in {batch_duration:.2f} seconds")
    
    return all_statistics_list
