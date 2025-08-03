"""
RunPod Serverless Adapter
Converts REST API calls to RunPod Serverless handler format
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from app.core.models import ApiResponse, TrainRequest, GenerateRequest
from app.core.config import get_settings

# Conditional import - only load RunPod handler in production mode
settings = get_settings()
if not settings.mock_mode:
    from app.rp_handler import async_handler
else:
    async_handler = None

logger = logging.getLogger(__name__)


class RunPodAdapter:
    """Adapter to convert REST API calls to RunPod Serverless format"""
    
    def __init__(self):
        settings = get_settings()
        if settings.mock_mode:
            # In mock mode, use the direct services instead of RunPod handler
            self.handler = None
            self.mock_mode = True
        else:
            self.handler = async_handler
            self.mock_mode = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Convert health check to RunPod format"""
        if self.mock_mode:
            # In mock mode, return mock health response
            return {
                "success": True,
                "data": {
                    "status": "healthy",
                    "services": {
                        "process_manager": "healthy",
                        "storage": "healthy", 
                        "gpu_manager": {
                            "total_gpus": 2,
                            "allocated_gpus": 1,
                            "available_gpus": 1,
                            "gpu_memory": "24GB",
                            "gpu_type": "RTX 4090 (Mock)"
                        }
                    }
                },
                "message": "API is healthy (Mock Mode)"
            }
        
        event = {"input": {"type": "health"}}
        return await self.handler(event)
    
    async def start_training(self, request: TrainRequest) -> Dict[str, Any]:
        """Convert training request to RunPod format"""
        if self.mock_mode:
            # In mock mode, just return success with fake process ID
            return {
                "success": True,
                "data": {"process_id": f"training_{asyncio.get_event_loop().time():.0f}"},
                "message": "Training process started successfully (Mock Mode)"
            }
            
        event = {
            "input": {
                "type": "train",
                "config": request.config
            }
        }
        return await self.handler(event)
    
    async def start_generation(self, request: GenerateRequest) -> Dict[str, Any]:
        """Convert generation request to RunPod format"""
        event = {
            "input": {
                "type": "generate", 
                "config": request.config
            }
        }
        return await self.handler(event)
    
    async def get_processes(self) -> Dict[str, Any]:
        """Get all processes"""
        event = {"input": {"type": "processes"}}
        return await self.handler(event)
    
    async def get_process(self, process_id: str) -> Dict[str, Any]:
        """Get specific process"""
        event = {
            "input": {
                "type": "process_status",
                "process_id": process_id
            }
        }
        return await self.handler(event)
    
    async def cancel_process(self, process_id: str) -> Dict[str, Any]:
        """Cancel process"""
        event = {
            "input": {
                "type": "cancel",
                "process_id": process_id
            }
        }
        return await self.handler(event)
    
    async def get_lora_models(self) -> Dict[str, Any]:
        """Get LoRA models"""
        event = {"input": {"type": "lora"}}
        return await self.handler(event)
    
    async def get_download_url(self, process_id: str) -> Dict[str, Any]:
        """Get download URL for process results"""
        # This would need to be implemented in the handler
        event = {
            "input": {
                "type": "download",
                "process_id": process_id
            }
        }
        return await self.handler(event) 