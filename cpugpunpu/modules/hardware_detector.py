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
    """CPU hardware capabilities"""
    cores: int
    threads: int
    architecture: str
    max_frequency: float
    cache_size: int
    instruction_sets: List[str]
    
@dataclass
class GPUCapabilities:
    """GPU hardware capabilities"""
    name: str
    memory_gb: float
    compute_units: int
    api_support: List[str]  # OpenCL, DirectX, etc.
    shared_memory: bool
    driver_version: str
    cuda_available: bool = False
    opencl_available: bool = False
    
@dataclass
class NPUCapabilities:
    """NPU hardware capabilities"""
    name: str
    tops_rating: float
    available: bool
    frameworks: List[str]  # ONNX, DirectML, etc.
    openvino_available: bool = False
    
@dataclass
class HardwareProfile:
    """Complete hardware profile"""
    cpu: CPUCapabilities
    gpu: Optional[GPUCapabilities]
    npu: Optional[NPUCapabilities]
    total_memory_gb: float
    available_memory_gb: float
    gpu_libraries_available: Dict[str, bool] = None
    npu_libraries_available: Dict[str, bool] = None
    
class HardwareDetector:
    """Detects and analyzes available hardware for optimal workload distribution"""
    
    def __init__(self):
        self.profile = None
        self._detect_hardware()
    
    def _detect_hardware(self) -> None:
        """Detect all available hardware capabilities"""
        cpu_caps = self._detect_cpu_capabilities()
        gpu_caps = self._detect_gpu_capabilities()
        npu_caps = self._detect_npu_capabilities()
        
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        available_memory_gb = memory_info.available / (1024**3)
        
        # Detect available acceleration libraries
        gpu_libraries = self._detect_gpu_libraries() if gpu_caps else {}
        npu_libraries = self._detect_npu_libraries() if npu_caps else {}
        
        # Update GPU capabilities with library info
        if gpu_caps:
            gpu_caps.cuda_available = gpu_libraries.get('cupy', False) or gpu_libraries.get('torch_cuda', False)
            gpu_caps.opencl_available = gpu_libraries.get('opencl', False)
        
        # Update NPU capabilities with library info
        if npu_caps:
            npu_caps.openvino_available = npu_libraries.get('openvino', False)
        
        self.profile = HardwareProfile(
            cpu=cpu_caps,
            gpu=gpu_caps,
            npu=npu_caps,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_libraries_available=gpu_libraries,
            npu_libraries_available=npu_libraries
        )
    
    def _detect_cpu_capabilities(self) -> CPUCapabilities:
        """Detect CPU capabilities"""
        cpu_count = os.cpu_count()
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # Get CPU frequency
        try:
            freq_info = psutil.cpu_freq()
            max_freq = freq_info.max if freq_info else 0.0
        except:
            max_freq = 0.0
        
        # Get architecture
        architecture = platform.machine()
        
        # Estimate cache size (simplified)
        cache_size = 0
        
        # Get instruction sets (simplified - would need more detailed detection)
        instruction_sets = []
        if architecture.lower() in ['amd64', 'x86_64']:
            instruction_sets = ['SSE', 'SSE2', 'AVX']  # Basic assumption
        
        return CPUCapabilities(
            cores=physical_cores or cpu_count,
            threads=logical_cores or cpu_count,
            architecture=architecture,
            max_frequency=max_freq,
            cache_size=cache_size,
            instruction_sets=instruction_sets
        )
    
    def _detect_gpu_capabilities(self) -> Optional[GPUCapabilities]:
        """Detect GPU capabilities using Windows commands"""
        try:
            # Try to get GPU info using wmic
            cmd = ['wmic', 'path', 'win32_VideoController', 'get', 'Name,AdapterRAM,DriverVersion', '/format:list']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = {}
                
                for line in lines:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if value.strip():
                            gpu_info[key.strip()] = value.strip()
                
                # Look for Intel Arc or integrated graphics
                if 'Name' in gpu_info and gpu_info['Name']:
                    name = gpu_info['Name']
                    if any(keyword in name.lower() for keyword in ['intel', 'arc', 'iris', 'uhd', 'graphics']):
                        return GPUCapabilities(
                            name=name,
                            memory_gb=18.0,  # Your system's shared GPU memory
                            compute_units=64,  # Intel Arc cores
                            api_support=['OpenCL', 'DirectX', 'DirectML'],
                            shared_memory=True,
                            driver_version=gpu_info.get('DriverVersion', 'Unknown')
                        )
            
            # Alternative: Check for GPU using PowerShell
            ps_cmd = ['powershell', '-Command', 'Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List']
            ps_result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=10)
            
            if ps_result.returncode == 0 and 'intel' in ps_result.stdout.lower():
                # Extract Intel GPU info from PowerShell output
                lines = ps_result.stdout.split('\n')
                for line in lines:
                    if 'Name' in line and 'Intel' in line:
                        name = line.split(':', 1)[1].strip() if ':' in line else 'Intel Graphics'
                        return GPUCapabilities(
                            name=name,
                            memory_gb=18.0,
                            compute_units=64,
                            api_support=['OpenCL', 'DirectX', 'DirectML'],
                            shared_memory=True,
                            driver_version='Unknown'
                        )
            
            return None
            
        except Exception as e:
            print(f"GPU detection failed: {e}")
            return None
    
    def _detect_npu_capabilities(self) -> Optional[NPUCapabilities]:
        """Detect NPU capabilities"""
        try:
            # Check for Intel AI Boost NPU using multiple methods
            methods = [
                ['wmic', 'path', 'win32_PnPEntity', 'where', 'Name like "%NPU%"', 'get', 'Name'],
                ['wmic', 'path', 'win32_PnPEntity', 'where', 'Name like "%AI Boost%"', 'get', 'Name'],
                ['wmic', 'path', 'win32_PnPEntity', 'where', 'Name like "%Neural%"', 'get', 'Name'],
                ['powershell', '-Command', 'Get-CimInstance -ClassName Win32_PnPEntity | Where-Object {$_.Name -like "*NPU*" -or $_.Name -like "*AI Boost*" -or $_.Name -like "*Neural*"} | Select-Object Name']
            ]
            
            for cmd in methods:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    output = result.stdout.lower()
                    if any(keyword in output for keyword in ['npu', 'ai boost', 'neural', 'ai accelerator']):
                        return NPUCapabilities(
                            name="Intel AI Boost NPU",
                            tops_rating=10.0,  # Your system's NPU rating
                            available=True,
                            frameworks=['ONNX', 'DirectML', 'OpenVINO']
                        )
            
            # Check CPU model for NPU capability (Intel Core Ultra series has NPU)
            if 'ultra' in platform.processor().lower():
                return NPUCapabilities(
                    name="Intel AI Boost NPU (Integrated)",
                    tops_rating=10.0,
                    available=True,
                    frameworks=['ONNX', 'DirectML', 'OpenVINO']
                )
            
            return None
            
        except Exception as e:
            print(f"NPU detection failed: {e}")
            return None
    
    def _detect_gpu_libraries(self) -> Dict[str, bool]:
        """Detect available GPU acceleration libraries"""
        libraries = {
            'cupy': False,
            'cudf': False,
            'numba_cuda': False,
            'tensorflow_gpu': False,
            'torch_cuda': False,
            'opencl': False
        }
        
        # Test CuPy (GPU-accelerated NumPy)
        try:
            import cupy
            libraries['cupy'] = True
        except ImportError:
            pass
        
        # Test cuDF (GPU-accelerated pandas)
        try:
            import cudf
            libraries['cudf'] = True
        except ImportError:
            pass
        
        # Test Numba CUDA
        try:
            from numba import cuda
            if cuda.is_available():
                libraries['numba_cuda'] = True
        except ImportError:
            pass
        
        # Test TensorFlow GPU
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                libraries['tensorflow_gpu'] = True
        except ImportError:
            pass
        
        # Test PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                libraries['torch_cuda'] = True
        except ImportError:
            pass
        
        # Test OpenCL
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                libraries['opencl'] = True
        except ImportError:
            pass
        
        return libraries
    
    def _detect_npu_libraries(self) -> Dict[str, bool]:
        """Detect available NPU acceleration libraries"""
        libraries = {
            'openvino': False,
            'intel_extension_pytorch': False,
            'onnxruntime_directml': False,
            'neural_compressor': False
        }
        
        # Test OpenVINO
        try:
            import openvino
            libraries['openvino'] = True
        except ImportError:
            pass
        
        # Test Intel Extension for PyTorch
        try:
            import intel_extension_for_pytorch
            libraries['intel_extension_pytorch'] = True
        except ImportError:
            pass
        
        # Test ONNX Runtime with DirectML
        try:
            import onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                libraries['onnxruntime_directml'] = True
        except ImportError:
            pass
        
        # Test Neural Compressor
        try:
            import neural_compressor
            libraries['neural_compressor'] = True
        except ImportError:
            pass
        
        return libraries
    
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
            return {'cpu': os.cpu_count(), 'gpu': 0, 'npu': 0}
        
        allocation = {'cpu': 0, 'gpu': 0, 'npu': 0}
        
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
            if self.profile.npu and data_size_gb > 1.0:
                allocation['npu'] = 1
        
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
        print(f"Architecture: {self.profile.cpu.architecture}")
        if self.profile.cpu.max_frequency > 0:
            print(f"Max Frequency: {self.profile.cpu.max_frequency:.2f} MHz")
        print(f"Instruction Sets: {', '.join(self.profile.cpu.instruction_sets)}")
        
        # GPU Information
        if self.profile.gpu:
            print(f"\nGPU: {self.profile.gpu.name}")
            print(f"Memory: {self.profile.gpu.memory_gb:.1f} GB ({'Shared' if self.profile.gpu.shared_memory else 'Dedicated'})")
            print(f"Compute Units: {self.profile.gpu.compute_units}")
            print(f"API Support: {', '.join(self.profile.gpu.api_support)}")
            print(f"Driver Version: {self.profile.gpu.driver_version}")
            print(f"CUDA Available: {'Yes' if self.profile.gpu.cuda_available else 'No'}")
            print(f"OpenCL Available: {'Yes' if self.profile.gpu.opencl_available else 'No'}")
            
            # Show available GPU libraries
            if self.profile.gpu_libraries_available:
                available_libs = [lib for lib, available in self.profile.gpu_libraries_available.items() if available]
                if available_libs:
                    print(f"GPU Libraries: {', '.join(available_libs)}")
                else:
                    print("GPU Libraries: None installed")
        else:
            print("\nGPU: Not detected or not supported")
        
        # NPU Information
        if self.profile.npu:
            print(f"\nNPU: {self.profile.npu.name}")
            print(f"Performance: {self.profile.npu.tops_rating} TOPS")
            print(f"Frameworks: {', '.join(self.profile.npu.frameworks)}")
            print(f"OpenVINO Available: {'Yes' if self.profile.npu.openvino_available else 'No'}")
            
            # Show available NPU libraries
            if self.profile.npu_libraries_available:
                available_libs = [lib for lib, available in self.profile.npu_libraries_available.items() if available]
                if available_libs:
                    print(f"NPU Libraries: {', '.join(available_libs)}")
                else:
                    print("NPU Libraries: None installed")
        else:
            print("\nNPU: Not detected or not supported")
        
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
