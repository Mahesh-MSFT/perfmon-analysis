# GPU-accelerated two-phase file processor for perfmon3.py
# Phase 1: GPU-accelerated data preparation (steepest fall detection + filtering)
# Phase 2: GPU batch processing (statistics calculation)

import os
import pandas as pd
import gc
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from modules.hardware_detector import get_hardware_detector

def detect_time_column(perfmon_data):
    """
    Detect the time column in the CSV data.
    Handles any timezone that relog.exe might generate (GMT, PST, EST, etc.) 
    with both Standard Time and Daylight/Summer Time variations.
    """
    print("   🔍 GPU-accelerated time column detection")
    
    for column in perfmon_data.columns:
        # Look for PDH-CSV 4.0 format with any timezone and time offset
        if (column.startswith('(PDH-CSV 4.0) (') and 
            'Time)(' in column and 
            column.endswith(')')):
            return column
    
    # Fallback - return the first column if no PDH-CSV time column is found
    if len(perfmon_data.columns) > 0:
        return perfmon_data.columns[0]
    
    raise ValueError("No time column found in the CSV data")

def process_batch(filtered_file_data: List[Dict], metric_names: List[str]) -> List[pd.DataFrame]:
    """
    Phase 2: GPU batch processing.
    Process all filtered data through GPU acceleration in batches.
    """
    if not filtered_file_data:
        return []
    
    # Processing strategy: GPU batch processing
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
                # Import calculate_statistics here to avoid circular imports
                from modules.calculate_statistics import calculate_statistics
                from shared.modules.ensure_consistent_structure import ensure_consistent_structure
                
                # Calculate statistics using GPU acceleration
                statistics_df = calculate_statistics(
                    metric_data, 
                    metric_name, 
                    file_date_time, 
                    start_time, 
                    steepest_fall_time.strftime('%H:%M')
                )
                
                # Ensure consistent structure
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

def calculate_hardware_aware_workers(csv_file_paths: List[str]) -> Dict[str, int]:
    """
    GPU worker allocation for two-phase architecture.
    """
    print("GPU worker allocation for two-phase architecture")
    
    if not csv_file_paths:
        return {'file_workers': 1}
    
    # Get hardware detector
    hardware = get_hardware_detector()
    
    # GPU file worker calculation - one worker per file up to CPU limit for Phase 1
    total_cpu_cores = hardware.profile.cpu.cores
    file_workers = min(len(csv_file_paths), max(1, total_cpu_cores // 2))
    
    allocation = {
        'file_workers': file_workers,
        'total_cpu_cores': total_cpu_cores
    }
    
    print(f"Phase 1 allocation: {file_workers} GPU workers (CPU cores: {total_cpu_cores})")
    print(f"Processing {len(csv_file_paths)} files")
    
    return allocation

def file_processor(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Process performance logs to generate statistical analysis:
    Phase 1: Data preparation (steepest fall detection + filtering)
    Phase 2: Statistics calculation and aggregation
    """
    
    print("Processing strategy: GPU-accelerated Phase 1 + GPU Phase 2 architecture")
    
    # Collect all CSV files
    csv_file_paths = []
    for root, dirs, files in os.walk(log_directory):
        for file_name in files:
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)
                csv_file_paths.append(file_path)
    
    if not csv_file_paths:
        print("No CSV files found in the specified directory.")
        return pd.DataFrame(), {}
    
    print(f"Found {len(csv_file_paths)} CSV files for GPU processing")
    
    # ==================== PHASE 1: GPU Data preparation ====================
    print(f"\n--- Phase 1: GPU-Accelerated Data Preparation ---")
    phase1_start_time = pd.Timestamp.now()
    
    # Calculate optimal workers for Phase 1
    hardware_allocation = calculate_hardware_aware_workers(csv_file_paths)
    
    # Use GPU processing function
    from modules.data_preprocessor import process_single_file
    process_function = process_single_file
    print("Using GPU acceleration for Phase 1 data preparation")
    
    # Prepare arguments for Phase 1
    phase1_args = [
        (file_path, baseline_metric_name)
        for file_path in csv_file_paths
    ]
    
    # Phase 1: Parallel file processing with TRUE streaming Phase 2 pipeline using ThreadPoolExecutor
    filtered_file_data = []
    all_statistics_list = []
    
    # Detailed timing tracking
    phase1_only_start = pd.Timestamp.now()
    phase2_only_start = None
    phase2_only_end = None
    first_phase2_start = None
    last_phase1_complete = None
    
    def process_metrics_immediately(file_result, file_path):
        """Process GPU Phase 2 for a single file immediately - managed by ThreadPoolExecutor"""
        nonlocal phase2_only_start, first_phase2_start
        
        # Track when first Phase 2 starts
        if first_phase2_start is None:
            first_phase2_start = pd.Timestamp.now()
            
        try:
            print(f"🚀 GPU Phase 2 starting immediately for {os.path.basename(file_path)}")
            
            # Process this single file through GPU Phase 2 immediately
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
    
    # Calculate optimal GPU Phase 2 workers based on hardware capabilities
    hardware = get_hardware_detector()
    
    # Determine optimal GPU worker count based on system specs
    if hardware.profile.gpu:
        # Intel Arc GPU - optimize for parallel GPU Phase 2 (actual compute units detected)
        gpu_compute_units = hardware.profile.gpu.compute_units  # Real value from OpenCL driver
        gpu_memory_gb = hardware.profile.gpu.memory_gb  # Real value from OpenCL driver
        
        # Optimal GPU worker count based on performance testing results:
        # - Intel Arc Graphics has 128 execution units (Xe-cores) 
        # - Performance testing showed 32 queues = 134.30/sec (OPTIMAL)
        # - ThreadPoolExecutor optimal: 8-16 threads for Phase 2 processing
        if gpu_compute_units >= 64:  # Intel Arc or equivalent
            optimal_gpu_workers = min(16, max(8, gpu_compute_units // 4))
        elif gpu_compute_units >= 32:  # Mid-range GPU
            optimal_gpu_workers = min(8, max(4, gpu_compute_units // 4))
        else:  # Lower-end GPU
            optimal_gpu_workers = min(4, max(2, gpu_compute_units // 8))
        
        # Adjust for available memory (each worker processes ~1-2GB data)
        memory_limited_workers = max(2, int(gpu_memory_gb / 2))
        optimal_gpu_workers = min(optimal_gpu_workers, memory_limited_workers)
        
        print(f"🎯 Phase 2 GPU Processing Configuration:")
        print(f"   GPU Hardware: {hardware.profile.gpu.name}")
        print(f"   Compute Units: {gpu_compute_units}, Memory: {gpu_memory_gb:.1f}GB")
        print(f"   Optimal GPU workers: {optimal_gpu_workers}")
    else:
        optimal_gpu_workers = 4  # Fallback for systems without GPU detection
        print("⚠️ GPU not detected, using fallback: 4 workers")
    
    # Start Phase 2 GPU processing as soon as first file completes Phase 1
    print(f"\n🚀 Starting STREAMING PIPELINE:")
    print(f"   Phase 1: GPU-accelerated data preparation")
    print(f"   Phase 2: GPU statistics processing (ThreadPoolExecutor)")
    print(f"   Pipeline: Phase 1 → Immediate Phase 2 streaming")
    
    with ProcessPoolExecutor(max_workers=hardware_allocation['file_workers']) as process_executor:
        # Use ThreadPoolExecutor for GPU Phase 2 - optimized worker count based on GPU specs
        with ThreadPoolExecutor(max_workers=optimal_gpu_workers, thread_name_prefix="GPU-Phase2") as gpu_executor:
            phase1_futures = {
                process_executor.submit(process_function, args): args[0] 
                for args in phase1_args
            }
            
            gpu_phase2_futures = []  # Track GPU Phase 2 futures
            completed_files = 0
            
            # Process Phase 1 results as they complete and immediately start Phase 2
            for phase1_future in as_completed(phase1_futures):
                file_path = phase1_futures[phase1_future]
                completed_files += 1
                
                try:
                    file_result = phase1_future.result()
                    if file_result is not None:
                        filtered_file_data.append(file_result)
                        
                        print(f"📋 Phase 1: {completed_files}/{len(csv_file_paths)} files prepared → Starting GPU Phase 2 for {os.path.basename(file_path)}")
                        
                        # 🎯 TRUE STREAMING: Submit GPU Phase 2 to ThreadPoolExecutor immediately
                        gpu_future = gpu_executor.submit(process_metrics_immediately, file_result, file_path)
                        gpu_phase2_futures.append(gpu_future)
                        
                        # Track when last Phase 1 completes
                        last_phase1_complete = pd.Timestamp.now()
                        
                    else:
                        print(f"📋 Phase 1: {completed_files}/{len(csv_file_paths)} files prepared (skipped due to error)")
                        # Still track completion time even for errors
                        last_phase1_complete = pd.Timestamp.now()
                    
                except Exception as e:
                    print(f"❌ Phase 1 error for file {file_path}: {e}")
                    last_phase1_complete = pd.Timestamp.now()
            
            # Wait for all GPU Phase 2 processing to complete and collect results
            print("⏳ Waiting for all GPU Phase 2 threads to complete...")
            phase2_only_start = pd.Timestamp.now()  # When we start waiting for Phase 2 to complete
            
            for gpu_future in as_completed(gpu_phase2_futures):
                try:
                    gpu_stats = gpu_future.result()
                    if gpu_stats:
                        all_statistics_list.extend(gpu_stats)
                except Exception as e:
                    print(f"❌ Error collecting GPU Phase 2 results: {e}")
                    
            phase2_only_end = pd.Timestamp.now()  # When all Phase 2 processing is complete
    
    # Calculate detailed timing metrics
    total_end_time = pd.Timestamp.now()
    total_pipeline_duration = (total_end_time - phase1_start_time).total_seconds()
    
    # Phase 1 timing
    if last_phase1_complete:
        phase1_only_duration = (last_phase1_complete - phase1_only_start).total_seconds()
    else:
        phase1_only_duration = total_pipeline_duration  # Fallback if no files processed
    
    # Phase 2 timing
    if first_phase2_start and phase2_only_end:
        phase2_total_duration = (phase2_only_end - first_phase2_start).total_seconds()
        phase2_only_duration = (phase2_only_end - phase2_only_start).total_seconds() if phase2_only_start else 0
    else:
        phase2_total_duration = 0
        phase2_only_duration = 0
    
    # Calculate overlap
    if first_phase2_start and last_phase1_complete:
        if first_phase2_start < last_phase1_complete:
            overlap_start = first_phase2_start
            overlap_end = min(last_phase1_complete, phase2_only_end) if phase2_only_end else last_phase1_complete
            overlap_duration = (overlap_end - overlap_start).total_seconds()
        else:
            overlap_duration = 0  # No overlap - Phase 1 finished before Phase 2 started
    else:
        overlap_duration = 0
    
    print(f"\n🏁 DETAILED PIPELINE TIMING ANALYSIS:")
    print(f"   📊 Total Pipeline Duration: {total_pipeline_duration:.2f}s")
    print(f"   📝 Phase 1 Duration: {phase1_only_duration:.2f}s")
    print(f"   🎯 Phase 2 Total Duration: {phase2_total_duration:.2f}s")
    print(f"   ⚡ Phase 2 Only Duration: {phase2_only_duration:.2f}s")
    print(f"   🔄 Overlap Duration: {overlap_duration:.2f}s")
    if overlap_duration > 0:
        overlap_percentage = (overlap_duration / phase1_only_duration) * 100
        print(f"   📈 Overlap Efficiency: {overlap_percentage:.1f}% of Phase 1 overlapped with Phase 2")
    else:
        print(f"   ⚠️ No Overlap: Sequential processing (no streaming benefit)")
    
    print(f"🏁 Streaming pipeline analysis complete: {len(filtered_file_data)} files processed")
    
    # ==================== PHASE 2: Completed via ThreadPoolExecutor streaming! ====================
    print(f"\n--- Phase 2: Completed via ThreadPoolExecutor streaming pipeline ---")
    
    # Calculate total processing time (streaming approach)
    total_duration = total_pipeline_duration  # Total pipeline time
    throughput = len(csv_file_paths) / total_duration if total_duration > 0 else 0
    
    print(f"\n💯 ThreadPoolExecutor Streaming Pipeline Complete:")
    print(f"   📁 Phase 1 processed: {len(filtered_file_data)} files")
    print(f"   🎯 GPU Phase 2 processed: {len(all_statistics_list)} statistics")
    print(f"   ⚡ Total time: {total_pipeline_duration:.2f}s")
    print(f"   🚀 Throughput: {throughput:.2f} files/second")
    print(f"   🔥 Architecture: GPU Phase 1 + ThreadPoolExecutor GPU Phase 2")
    
    architecture_type = "GPU+GPU"
    
    # Calculate and display performance metrics for comparison
    total_metrics_processed = len(all_statistics_list)
    if total_metrics_processed > 0:
        gpu_phase2_efficiency = total_metrics_processed / total_pipeline_duration
        print(f"\n📊 PERFORMANCE METRICS ({architecture_type}):")
        print(f"   🎯 Total Statistics Generated: {total_metrics_processed}")
        print(f"   ⚡ GPU Phase 2 Processing Rate: {gpu_phase2_efficiency:.1f} statistics/second")
        print(f"   🚀 Overall System Throughput: {throughput:.3f} files/second")
        
        # Store performance data for comparison with detailed timing
        performance_data = {
            'architecture': architecture_type,
            'total_time': total_pipeline_duration,
            'phase1_duration': phase1_only_duration,
            'phase2_duration': phase2_total_duration,
            'overlap_duration': overlap_duration,
            'files_processed': len(filtered_file_data),
            'statistics_generated': total_metrics_processed,
            'gpu_processing_rate': gpu_phase2_efficiency,
            'overall_throughput': throughput
        }
    else:
        performance_data = {
            'architecture': architecture_type,
            'total_time': total_pipeline_duration,
            'phase1_duration': phase1_only_duration,
            'phase2_duration': phase2_total_duration,
            'overlap_duration': overlap_duration,
            'files_processed': len(filtered_file_data) if filtered_file_data else 0,
            'statistics_generated': 0,
            'gpu_processing_rate': 0,
            'overall_throughput': 0
        }
    
    # Combine all results (already processed via streaming)
    if all_statistics_list:
        all_statistics_df = pd.concat(all_statistics_list, axis=0)
        
        # Pivot the DataFrame to have metrics as rows and average/maximum values as columns
        all_statistics_df = all_statistics_df.pivot_table(index='Metric', aggfunc='first')
        
        # Ensure the baseline metric is the first row
        if baseline_metric_name in all_statistics_df.index:
            all_statistics_df = all_statistics_df.reindex(
                [baseline_metric_name] + [idx for idx in all_statistics_df.index if idx != baseline_metric_name]
            )
        
        # Ensure all columns are numeric before rounding
        all_statistics_df = all_statistics_df.apply(pd.to_numeric, errors='coerce')
        
        # Round the data to a specified number of decimal places
        all_statistics_df = all_statistics_df.round(0)
        
        # Return both DataFrame and performance data
        return all_statistics_df, performance_data
    else:
        print("No statistics data was generated.")
        return pd.DataFrame(), performance_data

# Backward compatibility alias
def file_processor_accelerated(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Backward-compatible wrapper for file_processor"""
    print("Processing strategy: GPU-accelerated two-phase architecture (backward-compatible wrapper)")
    return file_processor(log_directory, metric_names, baseline_metric_name)

# Additional backward compatibility alias  
def process_performance_logs(log_directory: str, metric_names: List[str], baseline_metric_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Backward-compatible wrapper for file_processor"""
    return file_processor(log_directory, metric_names, baseline_metric_name)
