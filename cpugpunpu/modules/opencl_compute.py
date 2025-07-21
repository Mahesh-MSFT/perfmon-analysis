# OpenCL GPU acceleration utilities for Intel Arc Graphics
# Provides real GPU acceleration using PyOpenCL

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from typing import Optional, Any
import time

class OpenCLAccelerator:
    """OpenCL GPU acceleration manager for Intel Arc Graphics"""
    _initialization_logged = False  # Class variable to track if we've logged initialization
    
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.available = False
        self._initialize_opencl()
    
    def _initialize_opencl(self):
        """Initialize OpenCL context and command queue"""
        try:
            # Get the first available OpenCL platform and device
            platforms = cl.get_platforms()
            if not platforms:
                print("No OpenCL platforms found")
                return
            
            # Look for Intel GPU first, then any GPU
            self.device = None
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                for device in devices:
                    if 'Intel' in device.name and 'Arc' in device.name:
                        self.device = device
                        break
                if self.device:
                    break
            
            # If no Intel Arc, use any GPU
            if not self.device:
                for platform in platforms:
                    devices = platform.get_devices(cl.device_type.GPU)
                    if devices:
                        self.device = devices[0]
                        break
            
            if not self.device:
                print("No GPU devices found")
                return
            
            # Create context and command queue
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self.available = True
            
            # Only print initialization message once per session
            if not OpenCLAccelerator._initialization_logged:
                print(f"OpenCL GPU initialized: {self.device.name}")
                print(f"GPU Memory: {self.device.global_mem_size / (1024**3):.2f} GB")
                print(f"Processing strategy: GPU-accelerated (OpenCL)")
                OpenCLAccelerator._initialization_logged = True
            
        except Exception as e:
            print(f"OpenCL initialization failed: {e}")
            self.available = False
    
    def accelerated_mean(self, data: np.ndarray, use_gpu_threshold: int = 10000) -> float:
        """Calculate mean using GPU acceleration when beneficial"""
        if not self.available or len(data) < use_gpu_threshold:
            return float(np.mean(data))
        
        try:
            # Transfer data to GPU
            data_gpu = cl_array.to_device(self.queue, data.astype(np.float32))
            
            # Calculate mean on GPU
            result = cl_array.sum(data_gpu).get() / len(data)
            
            return float(result)
            
        except Exception as e:
            print(f"GPU mean calculation failed: {e}, falling back to CPU")
            return float(np.mean(data))
    
    def accelerated_max(self, data: np.ndarray, use_gpu_threshold: int = 10000) -> float:
        """Calculate max using GPU acceleration when beneficial"""
        if not self.available or len(data) < use_gpu_threshold:
            return float(np.max(data))
        
        try:
            # Transfer data to GPU
            data_gpu = cl_array.to_device(self.queue, data.astype(np.float32))
            
            # Calculate max on GPU
            result = cl_array.max(data_gpu).get()
            
            return float(result)
            
        except Exception as e:
            print(f"GPU max calculation failed: {e}, falling back to CPU")
            return float(np.max(data))
    
    def accelerated_percentage_change(self, data: np.ndarray, use_gpu_threshold: int = 10000) -> np.ndarray:
        """Calculate percentage change using GPU acceleration when beneficial"""
        if not self.available or len(data) < use_gpu_threshold:
            return np.diff(data) / data[:-1] * 100
        
        try:
            # Transfer data to GPU
            data_gpu = cl_array.to_device(self.queue, data.astype(np.float32))
            
            # Calculate percentage change on GPU
            diff_gpu = data_gpu[1:] - data_gpu[:-1]
            pct_change_gpu = diff_gpu / data_gpu[:-1] * 100
            
            return pct_change_gpu.get()
            
        except Exception as e:
            print(f"GPU percentage change calculation failed: {e}, falling back to CPU")
            return np.diff(data) / data[:-1] * 100
    
    def accelerated_statistics(self, data: np.ndarray, use_gpu_threshold: int = 10000) -> dict:
        """Calculate multiple statistics using GPU acceleration"""
        if not self.available or len(data) < use_gpu_threshold:
            return {
                'mean': float(np.mean(data)),
                'max': float(np.max(data)),
                'min': float(np.min(data)),
                'std': float(np.std(data))
            }
        
        try:
            # Transfer data to GPU
            data_gpu = cl_array.to_device(self.queue, data.astype(np.float32))
            
            # Calculate statistics on GPU
            mean_val = cl_array.sum(data_gpu).get() / len(data)
            max_val = cl_array.max(data_gpu).get()
            min_val = cl_array.min(data_gpu).get()
            
            # Standard deviation requires more complex calculation
            mean_gpu = cl_array.empty_like(data_gpu)
            mean_gpu.fill(mean_val)
            diff_gpu = data_gpu - mean_gpu
            var_gpu = cl_array.sum(diff_gpu * diff_gpu).get() / len(data)
            std_val = np.sqrt(var_gpu)
            
            return {
                'mean': float(mean_val),
                'max': float(max_val),
                'min': float(min_val),
                'std': float(std_val)
            }
            
        except Exception as e:
            print(f"GPU statistics calculation failed: {e}, falling back to CPU")
            return {
                'mean': float(np.mean(data)),
                'max': float(np.max(data)),
                'min': float(np.min(data)),
                'std': float(np.std(data))
            }
    
    def benchmark_performance(self, data_size: int = 100000) -> dict:
        """Benchmark GPU vs CPU performance"""
        data = np.random.randn(data_size).astype(np.float32)
        
        # CPU benchmark
        start_time = time.time()
        cpu_mean = np.mean(data)
        cpu_time = time.time() - start_time
        
        # GPU benchmark
        start_time = time.time()
        gpu_mean = self.accelerated_mean(data, use_gpu_threshold=1)  # Force GPU
        gpu_time = time.time() - start_time
        
        return {
            'data_size': data_size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': cpu_time / gpu_time if gpu_time > 0 else 0,
            'cpu_result': cpu_mean,
            'gpu_result': gpu_mean,
            'results_match': np.isclose(cpu_mean, gpu_mean)
        }
