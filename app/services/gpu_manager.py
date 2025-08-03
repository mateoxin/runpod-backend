"""
GPU Manager Service
Manages GPU allocation and monitoring for LoRA training and generation
"""

import asyncio
import logging
import psutil
import subprocess
from typing import Dict, List, Optional, Set
from datetime import datetime

from app.core.models import GPUInfo, GPUStatus

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and allocation."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.allocated_gpus: Dict[str, str] = {}  # gpu_id -> process_id
        self.gpu_info: Dict[str, GPUInfo] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize GPU manager."""
        await self.refresh_gpu_info()
        logger.info(f"GPU Manager initialized with {len(self.gpu_info)} GPUs")
        
    async def refresh_gpu_info(self) -> None:
        """Refresh GPU information using nvidia-smi."""
        try:
            # Query GPU information
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning("nvidia-smi command failed, assuming no GPUs available")
                return
                
            self.gpu_info.clear()
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_id = parts[0]
                    name = parts[1]
                    memory_total = int(parts[2]) * 1024 * 1024  # Convert MB to bytes
                    memory_used = int(parts[3]) * 1024 * 1024
                    memory_free = int(parts[4]) * 1024 * 1024
                    utilization = float(parts[5])
                    temperature = float(parts[6]) if parts[6] != '[Not Supported]' else None
                    
                    self.gpu_info[gpu_id] = GPUInfo(
                        id=gpu_id,
                        name=name,
                        memory_total=memory_total,
                        memory_used=memory_used,
                        memory_free=memory_free,
                        utilization=utilization,
                        temperature=temperature,
                        is_available=gpu_id not in self.allocated_gpus
                    )
                    
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi command timed out")
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, running without GPU support")
        except Exception as e:
            logger.error(f"Failed to refresh GPU info: {e}")
            
    async def allocate_gpu(self, process_id: str) -> Optional[str]:
        """Allocate a GPU for a process."""
        async with self._lock:
            # Check if we have reached max concurrent jobs
            if len(self.allocated_gpus) >= self.max_concurrent:
                logger.warning(f"Max concurrent jobs ({self.max_concurrent}) reached")
                return None
                
            # Refresh GPU info
            await self.refresh_gpu_info()
            
            # Find best available GPU (most free memory)
            best_gpu = None
            max_free_memory = 0
            
            for gpu_id, gpu_info in self.gpu_info.items():
                if gpu_id not in self.allocated_gpus and gpu_info.memory_free > max_free_memory:
                    best_gpu = gpu_id
                    max_free_memory = gpu_info.memory_free
                    
            if best_gpu:
                self.allocated_gpus[best_gpu] = process_id
                logger.info(f"Allocated GPU {best_gpu} to process {process_id}")
                return best_gpu
            else:
                logger.warning("No available GPUs")
                return None
                
    async def release_gpu(self, process_id: str) -> None:
        """Release GPU allocated to a process."""
        async with self._lock:
            gpu_to_release = None
            for gpu_id, allocated_process_id in self.allocated_gpus.items():
                if allocated_process_id == process_id:
                    gpu_to_release = gpu_id
                    break
                    
            if gpu_to_release:
                del self.allocated_gpus[gpu_to_release]
                logger.info(f"Released GPU {gpu_to_release} from process {process_id}")
                
    async def get_gpu_status(self) -> GPUStatus:
        """Get current GPU status."""
        await self.refresh_gpu_info()
        
        total_gpus = len(self.gpu_info)
        busy_gpus = len(self.allocated_gpus)
        available_gpus = total_gpus - busy_gpus
        
        # Update availability in gpu_info
        for gpu_id, gpu_info in self.gpu_info.items():
            gpu_info.is_available = gpu_id not in self.allocated_gpus
            
        return GPUStatus(
            total_gpus=total_gpus,
            available_gpus=available_gpus,
            busy_gpus=busy_gpus,
            gpus=list(self.gpu_info.values())
        )
        
    def get_status(self) -> Dict[str, any]:
        """Get manager status."""
        return {
            "total_gpus": len(self.gpu_info),
            "allocated_gpus": len(self.allocated_gpus),
            "max_concurrent": self.max_concurrent,
            "allocations": dict(self.allocated_gpus)
        }
        
    async def cleanup(self) -> None:
        """Cleanup GPU manager."""
        async with self._lock:
            self.allocated_gpus.clear()
            logger.info("GPU manager cleaned up") 