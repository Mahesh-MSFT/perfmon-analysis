# Hardware detection and capability analysis for CPU, GPU, and NPU
import os
import sys
import psutil
import platform
import subprocess
import json
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class CPUCapabilities:
    """CPU hardware capabilities - only what we actually use"""
    cores: int
    threads: int
    
@dataclass
class GPUCapabilities:
    """GPU hardware capabilities - only what we actually use"""
    name: str
    memory_gb: float
    compute_units: int
    shared_memory: bool
    opencl_available: bool = False
    
@dataclass
class HardwareProfile:
    """Complete hardware profile - streamlined"""
    cpu: CPUCapabilities
    gpu: Optional[GPUCapabilities]
    total_memory_gb: float
    available_memory_gb: float
    
class HardwareDetector:
    """Detects and analyzes available hardware for optimal workload distribution"""
    
    def __init__(self):
        self.profile = None
        self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect CPU and GPU capabilities - streamlined"""
        # Get memory info (fast operation)
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        available_memory_gb = memory_info.available / (1024**3)
        
        # Detect CPU (very fast - no parallel overhead needed)
        cpu_caps = self._detect_cpu_capabilities()
        
        # Detect GPU (only if OpenCL import succeeds - fail fast)
        gpu_caps = self._detect_gpu_capabilities()
        
        self.profile = HardwareProfile(
            cpu=cpu_caps,
            gpu=gpu_caps,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb
        )
    
    def _detect_cpu_capabilities(self) -> CPUCapabilities:
        """Detect CPU capabilities - only cores and threads"""
        return CPUCapabilities(
            cores=psutil.cpu_count(logical=False) or os.cpu_count(),
            threads=psutil.cpu_count(logical=True) or os.cpu_count()
        )
    
    def _detect_gpu_capabilities(self) -> Optional[GPUCapabilities]:
        """Detect GPU capabilities - fast OpenCL-only detection"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            
            # Look for Intel GPU specifically
            for platform in platforms:
                if 'Intel' in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    if devices:
                        device = devices[0]
                        
                        # Get only the essential info we actually use
                        memory_gb = 16.0  # Default fallback
                        try:
                            memory_gb = device.global_mem_size / (1024**3)
                        except:
                            pass
                        
                        return GPUCapabilities(
                            name=device.name.strip(),
                            memory_gb=memory_gb,
                            compute_units=device.max_compute_units,
                            shared_memory=True,  # Intel GPUs are shared memory
                            opencl_available=True
                        )
            
            return None  # No Intel GPU found
            
        except ImportError:
            return None  # OpenCL not available
        except Exception:
            return None  # Any other error
    
    def get_optimal_worker_allocation(self, workload_type: str, data_size_gb: float) -> Dict[str, int]:
        """
        Determine optimal worker allocation based on hardware capabilities and workload characteristics
        
        Args:
            workload_type: 'io_bound', 'cpu_bound', 'compute_bound', 'mixed'
            data_size_gb: Size of data to process
            
        Returns:
            Dict with worker counts for each hardware type
        """
        if not self.profile:
            return {'cpu': os.cpu_count(), 'gpu': 0}
        
        allocation = {'cpu': 0, 'gpu': 0}
        
        if workload_type == 'io_bound':
            # I/O bound tasks (like BLG conversion) benefit from more CPU workers
            # but should leave resources for other acceleration
            base_cpu_workers = min(self.profile.cpu.threads, 8)  # Cap at 8 for I/O
            
            if self.profile.gpu:
                allocation['cpu'] = int(base_cpu_workers * 0.75)  # Leave 25% for GPU tasks
            else:
                allocation['cpu'] = base_cpu_workers
                
        elif workload_type == 'cpu_bound':
            # CPU-intensive tasks
            allocation['cpu'] = max(1, int(self.profile.cpu.threads * 0.8))
            
        elif workload_type == 'compute_bound':
            # Computational tasks that can benefit from GPU acceleration
            if self.profile.gpu and data_size_gb > 0.1:  # Only use GPU for larger datasets
                allocation['gpu'] = 1  # GPU processes can handle multiple streams
                allocation['cpu'] = max(1, int(self.profile.cpu.threads * 0.5))
            else:
                allocation['cpu'] = max(1, int(self.profile.cpu.threads * 0.8))
                
        elif workload_type == 'mixed':
            # Mixed workloads - balance across available hardware
            allocation['cpu'] = max(1, int(self.profile.cpu.threads * 0.6))
            if self.profile.gpu:
                allocation['gpu'] = 1
        
        return allocation
    
    def print_hardware_summary(self) -> None:
        """Print detailed hardware summary"""
        if not self.profile:
            print("Hardware detection not completed")
            return
        
        print("=" * 60)
        print("HARDWARE ACCELERATION PROFILE")
        print("=" * 60)
        
        # CPU Information
        print(f"CPU: {self.profile.cpu.cores} cores, {self.profile.cpu.threads} threads")
        
        # GPU Information
        if self.profile.gpu:
            print(f"GPU: {self.profile.gpu.name}")
            print(f"Memory: {self.profile.gpu.memory_gb:.1f} GB ({'Shared' if self.profile.gpu.shared_memory else 'Dedicated'})")
            print(f"Compute Units: {self.profile.gpu.compute_units}")
            print(f"OpenCL Available: {'Yes' if self.profile.gpu.opencl_available else 'No'}")
            
            # Initialize GPU processor if OpenCL is available
            if self.profile.gpu.opencl_available:
                print(f"GPU Libraries: opencl")
                print()
                try:
                    from modules.gpu_processor import initialize_gpu_once
                    initialize_gpu_once()
                except Exception as e:
                    print(f"GPU initialization failed: {e}")
        else:
            print("GPU: Not detected or not supported")
        
        # Memory Information
        print(f"\nSystem Memory: {self.profile.total_memory_gb:.1f} GB total, {self.profile.available_memory_gb:.1f} GB available")
        
        print("=" * 60)

# Global hardware detector instance
_hardware_detector = None

def get_hardware_detector() -> HardwareDetector:
    """Get singleton hardware detector instance"""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector

def detect_hardware_capabilities() -> HardwareProfile:
    """Convenience function to get hardware profile"""
    return get_hardware_detector().profile

def get_optimal_workers(workload_type: str, data_size_gb: float = 1.0) -> Dict[str, int]:
    """Convenience function to get optimal worker allocation"""
    return get_hardware_detector().get_optimal_worker_allocation(workload_type, data_size_gb)

def print_hardware_info() -> None:
    """Convenience function to print hardware information"""
    get_hardware_detector().print_hardware_summary()
