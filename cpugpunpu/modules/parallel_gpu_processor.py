# Parallel GPU processor for true multi-core GPU utilization
# Each metric gets its own GPU work-group for parallel processing

import numpy as np
import pandas as pd
import pyopencl as cl
import pyopencl.array as cl_array
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ParallelGPUProcessor:
    """GPU processor that utilizes multiple GPU cores for parallel metric processing"""
    
    def __init__(self):
        self.context = None
        self.device = None
        self.queues = []  # Multiple command queues for parallel processing
        self.available = False
        self.max_parallel_jobs = 0
        self.lock = threading.Lock()
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU with multiple command queues for parallel processing"""
        try:
            # Get Intel Arc Graphics device
            platforms = cl.get_platforms()
            for platform in platforms:
                if 'Intel' in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    if devices:
                        self.device = devices[0]
                        break
            
            if not self.device:
                print("No Intel GPU found for parallel processing")
                return
            
            # Create context
            self.context = cl.Context([self.device])
            
            # Get GPU compute units (equivalent to cores)
            compute_units = self.device.max_compute_units
            max_work_group_size = self.device.max_work_group_size
            
            # Create multiple command queues for parallel processing
            # Intel Arc Graphics has 128 execution units (Xe-cores)
            # Based on comprehensive performance testing with 100 metrics:
            # 16 queues: 126.97/sec, 32 queues: 134.30/sec (OPTIMAL), 64+ queues: slower
            self.max_parallel_jobs = 32  # OPTIMAL: Best performance balance for Intel Arc Graphics
            
            for i in range(self.max_parallel_jobs):
                queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
                self.queues.append(queue)
            
            self.available = True
            
            # Only log initialization details once
            global _initialization_logged
            if not _initialization_logged:
                print(f"Parallel GPU initialized: {self.device.name}")
                print(f"Compute Units: {compute_units}, Max Work Group Size: {max_work_group_size}")
                print(f"Parallel GPU Queues: {len(self.queues)}")
                print(f"Max Parallel Jobs: {self.max_parallel_jobs}")
                _initialization_logged = True
            
        except Exception as e:
            print(f"Parallel GPU initialization failed: {e}")
            self.available = False
    
    def _process_single_metric_gpu(self, args: Tuple[int, np.ndarray, str]) -> Dict[str, Any]:
        """Process a single metric on GPU using dedicated queue"""
        queue_id, data, metric_name = args
        
        if not self.available or queue_id >= len(self.queues):
            # CPU fallback
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
                'processed_on': f'GPU_Queue_{queue_id}'
            }
            
        except Exception as e:
            print(f"GPU processing failed for {metric_name} on queue {queue_id}: {e}")
            return {
                'metric_name': metric_name,
                'mean': float(np.mean(data)),
                'max': float(np.max(data)),
                'queue_id': queue_id,
                'processed_on': 'CPU_Fallback'
            }
    
    def process_metrics_parallel(self, metric_data_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Process multiple metrics in parallel on different GPU cores"""
        
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
        
        print(f"Processing {len(metric_data_dict)} metrics in parallel on {self.max_parallel_jobs} GPU queues")
        
        # Prepare arguments for parallel processing
        metric_args = []
        queue_id = 0
        for metric_name, data in metric_data_dict.items():
            metric_args.append((queue_id % self.max_parallel_jobs, data.flatten(), metric_name))
            queue_id += 1
        
        # Process metrics in parallel using ThreadPoolExecutor
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            # Submit all metrics for parallel processing
            future_to_metric = {
                executor.submit(self._process_single_metric_gpu, args): args[2] 
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
                    #print(f"✓ {metric_name} processed on {result['processed_on']}")
                    
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
        
        print(f"Parallel processing complete: {gpu_processed} on GPU, {cpu_processed} on CPU")
        
        return results
    
    def get_gpu_utilization_info(self) -> Dict[str, Any]:
        """Get information about GPU utilization setup"""
        return {
            'available': self.available,
            'device_name': self.device.name if self.device else None,
            'compute_units': self.device.max_compute_units if self.device else 0,
            'max_parallel_jobs': self.max_parallel_jobs,
            'command_queues': len(self.queues),
            'max_work_group_size': self.device.max_work_group_size if self.device else 0
        }

# Global instance for reuse
_parallel_gpu_processor = None
_initialization_logged = False

def get_parallel_gpu_processor() -> ParallelGPUProcessor:
    """Get the singleton parallel GPU processor"""
    global _parallel_gpu_processor
    if _parallel_gpu_processor is None:
        _parallel_gpu_processor = ParallelGPUProcessor()
    return _parallel_gpu_processor
