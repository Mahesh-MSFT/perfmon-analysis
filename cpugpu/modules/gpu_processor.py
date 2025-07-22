# GPU Processing Module - Consolidated GPU functionality for perfmon3.py
# Combines batch processing, metrics processing, and OpenCL compute functionality

import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import OpenCL, fall back gracefully if not available
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    print("OpenCL not available, falling back to CPU processing")

class GPUProcessor:
    """Consolidated GPU processor for performance metrics"""
    
    def __init__(self, max_parallel_jobs: int = None):
        self.available = False
        self.context = None
        self.queues = []
        # Dynamic queue calculation based on GPU hardware capabilities
        self.max_parallel_jobs = max_parallel_jobs or self._calculate_optimal_queues()
        self.device_name = "CPU (Fallback)"
        
        if OPENCL_AVAILABLE:
            self._initialize_opencl()
    
    def _calculate_optimal_queues(self) -> int:
        """Calculate optimal queue count based on GPU hardware specifications"""
        try:
            from .hardware_detector import get_hardware_detector
            hardware = get_hardware_detector()
            
            if hardware.profile.gpu:
                compute_units = hardware.profile.gpu.compute_units
                
                # For simple operations (mean/max), we can use many more queues than compute units
                # Each queue handles lightweight operations with minimal resource usage
                # Testing 4:1 ratio to match EU count (512 queues for 512 EUs)
                if compute_units >= 128:  # High-end Intel Arc A770 (512 EUs)
                    return min(512, compute_units * 4)  # 4 queues per compute unit (1 per EU)
                elif compute_units >= 96:   # Mid-high Intel Arc A580 (384 EUs) 
                    return min(192, compute_units * 2)
                elif compute_units >= 64:   # Mid-range Intel Arc A380 (256 EUs)
                    return min(128, compute_units * 2)
                elif compute_units >= 32:   # Lower-end integrated graphics
                    return min(96, compute_units * 3)   # More queues per CU for integrated
                else:  # Very low-end
                    return min(64, compute_units * 4)   # Even more queues for very simple operations
            
            return 64  # Fallback for systems without GPU detection
            
        except Exception:
            return 64  # Conservative fallback
    
    def _initialize_opencl(self):
        """Initialize OpenCL context and command queues"""
        try:
            platforms = cl.get_platforms()
            if not platforms:
                return
            
            # Try to find GPU first, then any device
            device = None
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    device = devices[0]
                    break
            
            if not device:
                for platform in platforms:
                    devices = platform.get_devices()
                    if devices:
                        device = devices[0]
                        break
            
            if device:
                self.context = cl.Context([device])
                self.device_name = device.name.strip()
                
                # Create multiple command queues for parallel processing
                # For lightweight operations (mean/max), create more queues for better parallelism
                for i in range(self.max_parallel_jobs):
                    queue = cl.CommandQueue(self.context)
                    self.queues.append(queue)
                
                self.available = True
                # GPU initialization messages will be printed by caller when appropriate
                
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.available = False
    
    def print_initialization_info(self):
        """Print GPU initialization information in a controlled manner"""
        if self.available:
            print(f"Creating {self.max_parallel_jobs} GPU command queues for parallel processing...")
            print(f"GPU initialized: {self.device_name}")
            print(f"GPU queues created: {len(self.queues)} parallel command queues")
            print(f"Expected performance: ~{len(self.queues)} concurrent mean/max operations")
        else:
            print("GPU initialization failed, using CPU fallback")
    
    def get_utilization_info(self) -> Dict[str, Any]:
        """Get GPU utilization information"""
        try:
            from .hardware_detector import get_hardware_detector
            hardware = get_hardware_detector()
            actual_compute_units = hardware.profile.gpu.compute_units if hardware.profile.gpu else 0
        except:
            actual_compute_units = 128  # Fallback
            
        return {
            'device_name': self.device_name,
            'available': self.available,
            'compute_units': actual_compute_units,
            'command_queues': len(self.queues),
            'max_parallel_jobs': self.max_parallel_jobs,
            'queue_to_cu_ratio': len(self.queues) / actual_compute_units if actual_compute_units > 0 else 0
        }
    
    def process_single_metric(self, args) -> Dict[str, Any]:
        """Process a single metric on GPU or CPU fallback"""
        queue_id, data, metric_name = args
        
        # Check GPU availability first - no computation yet
        if not self.available or queue_id >= len(self.queues):
            # Only calculate CPU fallback when GPU unavailable
            return {
                'metric_name': metric_name,
                'mean': float(np.mean(data)),
                'max': float(np.max(data)),
                'queue_id': queue_id,
                'processed_on': 'CPU'
            }
        
        try:
            # Get dedicated queue for this metric
            queue = self.queues[queue_id]
            
            # Transfer data to GPU
            data_gpu = cl_array.to_device(queue, data.astype(np.float32))
            
            # Calculate statistics on GPU using dedicated queue
            mean_result = cl_array.sum(data_gpu).get() / len(data)
            max_result = cl_array.max(data_gpu).get()
            
            # Ensure completion on this queue
            queue.finish()
            
            return {
                'metric_name': metric_name,
                'mean': float(mean_result),
                'max': float(max_result),
                'queue_id': queue_id,
                'processed_on': f'GPU-Queue-{queue_id}'
            }
            
        except Exception as e:
            print(f"GPU processing failed for {metric_name}: {e}")
            # Only calculate CPU fallback when GPU fails
            return {
                'metric_name': metric_name,
                'mean': float(np.mean(data)),
                'max': float(np.max(data)),
                'queue_id': queue_id,
                'processed_on': 'CPU_Error_Fallback'
            }
    
    def process_metrics(self, metric_data_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Process multiple metrics in parallel on different GPU cores with adaptive queue sizing"""
        
        if not metric_data_dict:
            return {}
        
        if not self.available:
            print("GPU not available, using CPU fallback")
            results = {}
            for metric_name, data in metric_data_dict.items():
                results[metric_name] = {
                    'mean': float(np.mean(data)),
                    'max': float(np.max(data)),
                    'processed_on': 'CPU'
                }
            return results
        
        # Adaptive queue sizing based on actual workload
        workload_size = len(metric_data_dict)
        if workload_size < 150:
            optimal_queues = min(256, self.max_parallel_jobs)
            efficiency_note = "Small workload: using 256 queues for efficiency"
        elif workload_size < 300:
            optimal_queues = min(384, self.max_parallel_jobs) 
            efficiency_note = "Medium workload: using 384 queues for balance"
        else:
            optimal_queues = self.max_parallel_jobs
            efficiency_note = f"Large workload: using all {self.max_parallel_jobs} queues"
        
        queue_utilization = (workload_size / optimal_queues) * 100
        
        #print(f"Processing {workload_size} metrics with adaptive queue sizing")
        print(f"Adaptive queues: {optimal_queues} ({efficiency_note})")
        print(f"Queue utilization: {queue_utilization:.1f}% ({workload_size}/{optimal_queues})")
        
        # Prepare arguments for parallel processing with adaptive queue count
        metric_args = []
        queue_id = 0
        for metric_name, data in metric_data_dict.items():
            metric_args.append((queue_id % optimal_queues, data.flatten(), metric_name))
            queue_id += 1
        
        # Process metrics in parallel using ThreadPoolExecutor with optimal worker count
        results = {}
        
        with ThreadPoolExecutor(max_workers=optimal_queues) as executor:
            # Submit all metrics for parallel processing
            future_to_metric = {
                executor.submit(self.process_single_metric, args): args[2] 
                for args in metric_args
            }
            
            # Collect results
            for future in as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    result = future.result()
                    results[result['metric_name']] = {
                        'mean': result['mean'],
                        'max': result['max'],
                        'processed_on': result['processed_on'],
                        'queue_id': result['queue_id']
                    }
                    
                except Exception as e:
                    print(f"Error processing {metric_name}: {e}")
                    # CPU fallback for failed metrics
                    if metric_name in metric_data_dict:
                        data = metric_data_dict[metric_name]
                        results[metric_name] = {
                            'mean': float(np.mean(data)),
                            'max': float(np.max(data)),
                            'processed_on': 'CPU_Error_Fallback'
                        }
        
        # Summary
        gpu_processed = sum(1 for r in results.values() if 'GPU' in r.get('processed_on', ''))
        cpu_processed = len(results) - gpu_processed
        
        #print(f"Parallel processing complete: {gpu_processed} on GPU, {cpu_processed} on CPU")
        return results

# Global instance
_gpu_processor = None

def initialize_gpu_once():
    """Initialize GPU processor once globally"""
    global _gpu_processor
    if _gpu_processor is None:
        print("\n=== GPU INITIALIZATION ===")
        _gpu_processor = GPUProcessor()
        _gpu_processor.print_initialization_info()
        print("=== GPU INITIALIZATION COMPLETE ===\n")
    return _gpu_processor

def get_gpu_processor(print_init_info: bool = False) -> GPUProcessor:
    """Get or create the global GPU processor instance"""
    global _gpu_processor
    if _gpu_processor is None:
        _gpu_processor = GPUProcessor()
        if print_init_info:
            _gpu_processor.print_initialization_info()
    return _gpu_processor

def process_file_metrics(filtered_file_data: List[Dict], metric_names: List[str], baseline_metric_name: str = None) -> List[pd.DataFrame]:
    """
    Enhanced GPU batch processing that utilizes multiple GPU cores in parallel.
    Each metric gets processed on a separate GPU core simultaneously.
    """
    if not filtered_file_data:
        return []
    
    # Get the already-initialized GPU processor (no initialization messages)
    gpu_processor = get_gpu_processor()
    gpu_info = gpu_processor.get_utilization_info()
    
    all_statistics_list = []
    batch_start_time = pd.Timestamp.now()
    
    # Process each file's metrics in parallel batches
    for file_data in filtered_file_data:
        if file_data is None:
            continue
            
        file_path = file_data['file_path']
        filtered_perfmon_data = file_data['filtered_data']
        time_column = file_data['time_column']
        steepest_fall_time = file_data['steepest_fall_time']
        file_date_time = file_data['file_date_time']
        start_time = file_data['start_time']
        
        file_name = file_path.split('\\')[-1]
        print(f"\nProcessing {len(metric_names)} metrics for {file_name} in parallel...")
        
        # Prepare metric data for parallel GPU processing
        # Handle baseline metric (single column) and other metrics (multiple columns) differently
        file_metric_data = {}
        valid_metrics = []
        column_to_metric_map = {}  # Map full column names to base metric names
        
        for metric_name in metric_names:
            # Find all columns that match this metric
            metric_columns = [col for col in filtered_perfmon_data.columns if metric_name in col]
            
            if metric_columns:
                if baseline_metric_name and metric_name == baseline_metric_name:
                    # For baseline metric, process only the FIRST matching column
                    column_name = metric_columns[0]
                    column_data = pd.to_numeric(filtered_perfmon_data[column_name], errors='coerce').dropna()
                    if len(column_data) > 0:
                        file_metric_data[column_name] = column_data.values
                        valid_metrics.append(column_name)
                        column_to_metric_map[column_name] = metric_name
                        print(f"  Baseline metric '{metric_name}': Using single column '{column_name}'")
                else:
                    # For other metrics, process ALL matching columns individually
                    columns_added = 0
                    for column_name in metric_columns:
                        column_data = pd.to_numeric(filtered_perfmon_data[column_name], errors='coerce').dropna()
                        if len(column_data) > 0:
                            file_metric_data[column_name] = column_data.values
                            valid_metrics.append(column_name)
                            column_to_metric_map[column_name] = metric_name
                            columns_added += 1
                    #if columns_added > 0:
                        #print(f"  Metric '{metric_name}': Using {columns_added} columns")
        
        if not file_metric_data:
            print(f"No valid metric data found for {file_path}")
            continue
        
        file_start_time = time.time()
        
        print(f"\n=== GPU Phase 2: Processing {file_name} ===")
        print(f"  → Processing {len(file_metric_data)} individual columns/metrics in parallel on GPU")
        
        # Process ALL metrics for this file in parallel on GPU (reuse existing processor)
        parallel_results = gpu_processor.process_metrics(file_metric_data)
        
        file_end_time = time.time()
        file_duration = file_end_time - file_start_time
        
        #print(f"  ✓ GPU Phase 2 completed in {file_duration:.3f}s for {len(valid_metrics)} individual columns")
        print(f"=== GPU Phase 2 Complete for {file_name} in {file_duration:.3f}s for {len(valid_metrics)} metrics ===\n")
        
        # Convert results to DataFrame format
        processed_count = 0
        for column_name in valid_metrics:
            if column_name not in parallel_results:
                continue
                
            try:
                result = parallel_results[column_name]
                
                # Import required functions
                from .calculate_statistics import extract_header_from_column_name, remove_first_word_after_backslashes
                from shared.modules.ensure_consistent_structure import ensure_consistent_structure
                
                # Extract header information directly from the column name
                column_header = extract_header_from_column_name(column_name)
                modified_metric_name = remove_first_word_after_backslashes(column_name)
                
                # Create duration string
                duration = f"({start_time} - {steepest_fall_time.strftime('%H:%M')})"
                
                # Create statistics data with parallel GPU results for individual column
                statistics_data = {
                    'Metric': [modified_metric_name],
                    f"{file_date_time}\n{column_header}\nAvg.\n{duration}": [result['mean']],
                    f"{file_date_time}\n{column_header}\nMax.\n{duration}": [result['max']]
                }
                
                # Add bytes to Mbps conversion if applicable
                if 'bytes total/sec' in modified_metric_name.lower():
                    mbps_metric = modified_metric_name.replace('Bytes', 'Mbps')
                    
                    # Convert to Mbps
                    avg_mbps = (result['mean'] * 8) / 1_000_000
                    max_mbps = (result['max'] * 8) / 1_000_000
                    
                    avg_col = f"{file_date_time}\n{column_header}\nAvg.\n{duration}"
                    max_col = f"{file_date_time}\n{column_header}\nMax.\n{duration}"
                    
                    # Add Mbps metric to the data
                    statistics_data['Metric'].append(mbps_metric)
                    statistics_data[avg_col].append(avg_mbps)
                    statistics_data[max_col].append(max_mbps)
                
                # Create DataFrame
                statistics_df = pd.DataFrame(statistics_data)
                statistics_df = ensure_consistent_structure(statistics_df)
                
                if not statistics_df.empty:
                    all_statistics_list.append(statistics_df)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error creating DataFrame for column {column_name}: {e}")
        
        #print(f"  ✓ Created {processed_count} DataFrames from {len(valid_metrics)} processed columns")
    
    # Only show overall summary if processing multiple files
    if len(filtered_file_data) > 1:
        batch_end_time = pd.Timestamp.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        
        # Calculate total actual columns processed (not just base metric names)
        total_columns_processed = sum(len([col for col in file_data['filtered_data'].columns 
                                          for metric_name in metric_names if metric_name in col]) 
                                     for file_data in filtered_file_data if file_data is not None)
        throughput = total_columns_processed / batch_duration if batch_duration > 0 else 0
        
        print(f"\n=== All GPU Processing Complete ===")
        print(f"Total Duration: {batch_duration:.2f}s")
        print(f"Total Files Processed: {len(filtered_file_data)}")
        print(f"Total Columns Processed: {total_columns_processed} (individual performance counters)")
        print(f"Base Metric Names: {len(metric_names)}")
        print(f"Overall Throughput: {throughput:.2f} columns/second")
        print(f"Total Results: {len(all_statistics_list)} DataFrames generated")
    
    return all_statistics_list
