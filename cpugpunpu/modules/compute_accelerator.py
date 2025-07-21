# GPU acceleration utilities for perfmon3.py
# Provides GPU-accelerated operations with CPU fallback

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any
from modules.hardware_detector import get_hardware_detector

class GPUAccelerator:
    """GPU acceleration manager with automatic fallback to CPU"""
    
    def __init__(self):
        self.hardware = get_hardware_detector()
        self.gpu_available = self._check_gpu_availability()
        self.cupy_available = False
        self.cudf_available = False
        self.opencl_available = False
        self.opencl_accelerator = None
        
        if self.gpu_available:
            self._initialize_gpu_libraries()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        if not self.hardware.profile.gpu:
            return False
        
        # Check if any GPU libraries are available
        gpu_libs = self.hardware.profile.gpu_libraries_available or {}
        return any(gpu_libs.values())
    
    def _initialize_gpu_libraries(self):
        """Initialize available GPU libraries"""
        gpu_libs = self.hardware.profile.gpu_libraries_available or {}
        
        # Try to import CuPy for GPU arrays
        if gpu_libs.get('cupy', False):
            try:
                import cupy as cp
                self.cupy = cp
                self.cupy_available = True
                print("Processing strategy: GPU-accelerated (CuPy available)")
            except ImportError:
                print("Processing strategy: CuPy not available, using CPU fallback")
        
        # Try to import cuDF for GPU dataframes
        if gpu_libs.get('cudf', False):
            try:
                import cudf
                self.cudf = cudf
                self.cudf_available = True
                print("Processing strategy: GPU-accelerated (cuDF available)")
            except ImportError:
                print("Processing strategy: cuDF not available, using CPU fallback")
        
        # Try to initialize OpenCL acceleration
        if gpu_libs.get('opencl', False):
            try:
                from modules.opencl_compute import OpenCLAccelerator
                self.opencl_accelerator = OpenCLAccelerator()
                self.opencl_available = self.opencl_accelerator.available
                if self.opencl_available:
                    print("Processing strategy: GPU-accelerated (OpenCL available)")
            except ImportError:
                print("Processing strategy: OpenCL not available, using CPU fallback")
        
        # Fall back to CPU if no GPU libraries available
        if not self.cupy_available and not self.cudf_available and not self.opencl_available:
            print("Processing strategy: CPU-based (no GPU libraries available)")
    
    def calculate_mean(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """GPU-accelerated mean calculation with CPU fallback"""
        if self.cupy_available:
            try:
                # Processing strategy: GPU-accelerated mean calculation
                print("Processing strategy: GPU-accelerated mean calculation (CuPy)")
                gpu_data = self.cupy.asarray(data)
                result = self.cupy.asnumpy(self.cupy.nanmean(gpu_data, axis=axis))
                return result
            except Exception as e:
                print(f"GPU mean calculation failed, falling back to CPU: {e}")
        
        if self.opencl_available:
            try:
                # Processing strategy: OpenCL GPU-accelerated mean calculation
                result = self.opencl_accelerator.accelerated_mean(data.flatten())
                return np.array(result)
            except Exception as e:
                print(f"OpenCL GPU mean calculation failed, falling back to CPU: {e}")
        
        # Processing strategy: CPU-based mean calculation
        print("Processing strategy: CPU-based mean calculation")
        return np.nanmean(data, axis=axis)
    
    def calculate_max(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """GPU-accelerated max calculation with CPU fallback"""
        if self.cupy_available:
            try:
                # Processing strategy: GPU-accelerated max calculation
                print("Processing strategy: GPU-accelerated max calculation (CuPy)")
                gpu_data = self.cupy.asarray(data)
                result = self.cupy.asnumpy(self.cupy.nanmax(gpu_data, axis=axis))
                return result
            except Exception as e:
                print(f"GPU max calculation failed, falling back to CPU: {e}")
        
        if self.opencl_available:
            try:
                # Processing strategy: OpenCL GPU-accelerated max calculation
                result = self.opencl_accelerator.accelerated_max(data.flatten())
                return np.array(result)
            except Exception as e:
                print(f"OpenCL GPU max calculation failed, falling back to CPU: {e}")
        
        # Processing strategy: CPU-based max calculation
        print("Processing strategy: CPU-based max calculation")
        return np.nanmax(data, axis=axis)
    
    def calculate_percentage_change(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated percentage change calculation with CPU fallback"""
        if self.cupy_available:
            try:
                # Processing strategy: GPU-accelerated percentage change
                print("Processing strategy: GPU-accelerated percentage change calculation (CuPy)")
                gpu_data = self.cupy.asarray(data)
                with self.cupy.errstate(divide='ignore', invalid='ignore'):
                    pct_changes = self.cupy.abs(self.cupy.diff(gpu_data) / gpu_data[:-1]) * 100
                result = self.cupy.asnumpy(pct_changes)
                return result
            except Exception as e:
                print(f"GPU percentage change calculation failed, falling back to CPU: {e}")
        
        if self.opencl_available:
            try:
                # Processing strategy: OpenCL GPU-accelerated percentage change
                result = self.opencl_accelerator.accelerated_percentage_change(data)
                return result
            except Exception as e:
                print(f"OpenCL GPU percentage change calculation failed, falling back to CPU: {e}")
        
        # Processing strategy: CPU-based percentage change
        print("Processing strategy: CPU-based percentage change calculation")
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_changes = np.abs(np.diff(data) / data[:-1]) * 100
        return pct_changes
    
    def process_dataframe(self, df: pd.DataFrame, operation: str) -> pd.DataFrame:
        """GPU-accelerated dataframe operations with CPU fallback"""
        if self.cudf_available and len(df) > 50000:
            try:
                # Processing strategy: GPU-accelerated dataframe operations
                print(f"Processing strategy: GPU-accelerated dataframe {operation}")
                gpu_df = self.cudf.DataFrame(df)
                
                if operation == 'mean':
                    result = gpu_df.mean()
                elif operation == 'max':
                    result = gpu_df.max()
                else:
                    result = gpu_df
                
                return result.to_pandas() if hasattr(result, 'to_pandas') else result
            except Exception as e:
                print(f"GPU dataframe operation failed, falling back to CPU: {e}")
        
        # Processing strategy: CPU-based dataframe operations
        print(f"Processing strategy: CPU-based dataframe {operation}")
        if operation == 'mean':
            return df.mean()
        elif operation == 'max':
            return df.max()
        else:
            return df
    
    def get_acceleration_info(self) -> dict:
        """Get information about available acceleration"""
        return {
            'gpu_available': self.gpu_available,
            'cupy_available': self.cupy_available,
            'cudf_available': self.cudf_available,
            'gpu_name': self.hardware.profile.gpu.name if self.hardware.profile.gpu else None,
            'gpu_memory_gb': self.hardware.profile.gpu.memory_gb if self.hardware.profile.gpu else 0
        }

# Global GPU accelerator instance
_gpu_accelerator = None

def get_compute_accelerator() -> GPUAccelerator:
    """Get singleton GPU accelerator instance"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator

def calculate_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculate mean using compute acceleration"""
    return get_compute_accelerator().calculate_mean(data, axis)

def calculate_max(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculate maximum using compute acceleration"""
    return get_compute_accelerator().calculate_max(data, axis)

def calculate_percentage_change(data: np.ndarray) -> np.ndarray:
    """Calculate percentage change using compute acceleration"""
    return get_compute_accelerator().calculate_percentage_change(data)

# Backward compatibility aliases
def get_gpu_accelerator() -> GPUAccelerator:
    """Backward-compatible wrapper for get_compute_accelerator"""
    return get_compute_accelerator()

def gpu_accelerated_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Backward-compatible wrapper for calculate_mean"""
    return calculate_mean(data, axis)

def gpu_accelerated_max(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Backward-compatible wrapper for calculate_max"""
    return calculate_max(data, axis)

def gpu_accelerated_percentage_change(data: np.ndarray) -> np.ndarray:
    """Backward-compatible wrapper for calculate_percentage_change"""
    return calculate_percentage_change(data)
