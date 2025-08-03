"""
Mock Services for Local Testing
Provides fake implementations without external dependencies
"""

import asyncio
import uuid
import yaml
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from app.core.models import Process, ProcessStatus, ProcessType, LoRAModel, LoRAMetadata


class MockProcessManager:
    """Mock Process Manager without Redis dependency"""
    
    def __init__(self):
        self.processes: Dict[str, Process] = {}
        self.running_counter = 0
        
        # Add some sample processes
        self._create_sample_processes()
        
    def _create_sample_processes(self):
        """Create sample processes for testing"""
        now = datetime.now(timezone.utc)
        
        # Completed training process
        training_process = Process(
            id="training_123",
            name="Sample LoRA Training",
            type=ProcessType.TRAINING,
            status=ProcessStatus.COMPLETED,
            config={
                "job": "extension",
                "config": {
                    "name": "sample_training",
                    "trigger_word": "test_style"
                }
            },
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(minutes=30),
            output_path="/workspace/output/training_123",
            progress=100
        )
        
        # Running generation process
        generation_process = Process(
            id="generation_456",
            name="Landscape Generation",
            type=ProcessType.GENERATION,
            status=ProcessStatus.RUNNING,
            config={
                "job": "generate",
                "config": {
                    "prompts": ["beautiful landscape with test_style"]
                }
            },
            created_at=now - timedelta(minutes=15),
            updated_at=now - timedelta(minutes=1),
            output_path="/workspace/output/generation_456",
            progress=25
        )
        
        # Queued process
        queued_process = Process(
            id="training_789",
            name="Character Training",
            type=ProcessType.TRAINING,
            status=ProcessStatus.PENDING,
            config={
                "job": "extension",
                "config": {
                    "name": "character_training"
                }
            },
            created_at=now - timedelta(minutes=5),
            updated_at=now - timedelta(minutes=5),
            output_path=None,
            progress=0
        )
        
        self.processes = {
            training_process.id: training_process,
            generation_process.id: generation_process,
            queued_process.id: queued_process
        }
    
    async def get_all_processes(self) -> List[Process]:
        """Get all processes"""
        # Simulate some progress on running processes
        for process in self.processes.values():
            if process.status == ProcessStatus.RUNNING:
                process.progress = min(100, process.progress + 5)
                process.updated_at = datetime.now(timezone.utc)
                if process.progress >= 100:
                    process.status = ProcessStatus.COMPLETED
        
        return list(self.processes.values())
    
    async def get_process(self, process_id: str) -> Optional[Process]:
        """Get specific process"""
        return self.processes.get(process_id)
    
    async def start_training(self, config: str) -> str:
        """Start training process"""
        process_id = f"training_{uuid.uuid4().hex[:8]}"
        
        # Parse config to extract name
        try:
            config_data = yaml.safe_load(config)
            name = config_data.get("config", {}).get("name", "unnamed_training")
        except:
            name = "training"
        
        # Parse YAML config to dict
        try:
            config_dict = yaml.safe_load(config)
        except:
            config_dict = {"config": {"name": name}}
            
        process = Process(
            id=process_id,
            name=name,
            type=ProcessType.TRAINING,
            status=ProcessStatus.PENDING,
            config=config_dict,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            output_path=f"/workspace/output/{process_id}",
            progress=0
        )
        
        self.processes[process_id] = process
        
        # Simulate async processing
        asyncio.create_task(self._simulate_training(process_id))
        
        return process_id
    
    async def start_generation(self, config: str) -> str:
        """Start generation process"""
        process_id = f"generation_{uuid.uuid4().hex[:8]}"
        
        # Parse YAML config to dict
        try:
            config_dict = yaml.safe_load(config)
            name = config_dict.get("config", {}).get("name", "Generation Process")
        except:
            config_dict = {"config": {"name": "generation"}}
            name = "Generation Process"
            
        process = Process(
            id=process_id,
            name=name,
            type=ProcessType.GENERATION,
            status=ProcessStatus.PENDING,
            config=config_dict,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            output_path=f"/workspace/output/{process_id}",
            progress=0
        )
        
        self.processes[process_id] = process
        
        # Simulate async processing
        asyncio.create_task(self._simulate_generation(process_id))
        
        return process_id
    
    async def cancel_process(self, process_id: str) -> bool:
        """Cancel process"""
        if process_id in self.processes:
            process = self.processes[process_id]
            if process.status in [ProcessStatus.PENDING, ProcessStatus.RUNNING]:
                process.status = ProcessStatus.CANCELLED
                process.updated_at = datetime.now(timezone.utc)
                return True
        return False
    
    async def _simulate_training(self, process_id: str):
        """Simulate training progress"""
        await asyncio.sleep(2)  # Initial delay
        
        if process_id not in self.processes:
            return
            
        process = self.processes[process_id]
        process.status = ProcessStatus.RUNNING
        
        # Simulate progress updates
        for progress in range(0, 101, 10):
            await asyncio.sleep(3)  # Simulate work
            if process_id not in self.processes or process.status == ProcessStatus.CANCELLED:
                return
                
            process.progress = progress
            process.updated_at = datetime.now(timezone.utc)
            
            if progress >= 100:
                process.status = ProcessStatus.COMPLETED
    
    async def _simulate_generation(self, process_id: str):
        """Simulate generation progress"""
        await asyncio.sleep(1)  # Initial delay
        
        if process_id not in self.processes:
            return
            
        process = self.processes[process_id]
        process.status = ProcessStatus.RUNNING
        
        # Simulate progress updates
        for progress in range(0, 101, 20):
            await asyncio.sleep(2)  # Simulate work
            if process_id not in self.processes or process.status == ProcessStatus.CANCELLED:
                return
                
            process.progress = progress
            process.updated_at = datetime.now(timezone.utc)
            
            if progress >= 100:
                process.status = ProcessStatus.COMPLETED


class MockStorageService:
    """Mock Storage Service without S3 dependency"""
    
    def __init__(self):
        self.bucket_name = "mock-bucket"
    
    async def health_check(self) -> str:
        """Health check"""
        return "healthy"
    
    async def upload_file(self, local_path: str, s3_key: str) -> str:
        """Mock file upload"""
        return f"https://mock-storage.com/{self.bucket_name}/{s3_key}"
    
    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """Mock file download"""
        return True
    
    async def get_download_url(self, process_id: str) -> str:
        """Mock download URL"""
        return f"https://mock-storage.com/{self.bucket_name}/results/{process_id}.zip?expires=1234567890"
    
    async def list_files(self, prefix: str = "") -> List[dict]:
        """Mock file listing"""
        return [
            {
                'key': f'{prefix}sample_model_1.safetensors',
                'size': 1048576,  # 1MB
                'last_modified': datetime.now(timezone.utc) - timedelta(days=1),
                'etag': '"abc123"'
            },
            {
                'key': f'{prefix}sample_model_2.safetensors', 
                'size': 2097152,  # 2MB
                'last_modified': datetime.now(timezone.utc) - timedelta(hours=6),
                'etag': '"def456"'
            }
        ]


class MockLoRAService:
    """Mock LoRA Service"""
    
    def __init__(self, storage_service: MockStorageService):
        self.storage_service = storage_service
        self.models_cache = {}
        self._create_sample_models()
    
    def _create_sample_models(self):
        """Create sample LoRA models"""
        now = datetime.now(timezone.utc)
        
        models = [
            LoRAModel(
                id="lora_001",
                name="Anime Style LoRA",
                path="/workspace/models/anime_style.safetensors",
                created_at=now - timedelta(days=2),
                size=1048576,  # 1MB
                metadata=LoRAMetadata(
                    trigger_word="anime_style",
                    model_type="LoRA",
                    steps=1000,
                    learning_rate=0.0001,
                    base_model="flux.1-dev"
                )
            ),
            LoRAModel(
                id="lora_002", 
                name="Portrait Photography",
                path="/workspace/models/portrait_photo.safetensors",
                created_at=now - timedelta(days=1),
                size=2097152,  # 2MB
                metadata=LoRAMetadata(
                    trigger_word="portrait_photo",
                    model_type="LoRA",
                    steps=800,
                    learning_rate=0.00015,
                    base_model="flux.1-dev"
                )
            ),
            LoRAModel(
                id="lora_003",
                name="Landscape Art",
                path="/workspace/models/landscape_art.safetensors", 
                created_at=now - timedelta(hours=12),
                size=1572864,  # 1.5MB
                metadata=LoRAMetadata(
                    trigger_word="landscape_art",
                    model_type="LoRA",
                    steps=1200,
                    learning_rate=0.0001,
                    base_model="flux.1-dev"
                )
            )
        ]
        
        for model in models:
            self.models_cache[model.id] = model
    
    async def get_available_models(self) -> List[LoRAModel]:
        """Get all available LoRA models"""
        return list(self.models_cache.values())
    
    async def get_model_by_id(self, model_id: str) -> Optional[LoRAModel]:
        """Get specific model"""
        return self.models_cache.get(model_id)


class MockGPUManager:
    """Mock GPU Manager"""
    
    def __init__(self):
        self.total_gpus = 2
        self.allocated_gpus = 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get GPU status"""
        return {
            "total_gpus": self.total_gpus,
            "allocated_gpus": self.allocated_gpus,
            "available_gpus": self.total_gpus - self.allocated_gpus,
            "gpu_memory": "24GB",
            "gpu_type": "RTX 4090 (Mock)"
        }
    
    async def allocate_gpu(self) -> Optional[str]:
        """Mock GPU allocation"""
        if self.allocated_gpus < self.total_gpus:
            self.allocated_gpus += 1
            return f"cuda:{self.allocated_gpus - 1}"
        return None
    
    async def release_gpu(self, gpu_id: str) -> bool:
        """Mock GPU release"""
        if self.allocated_gpus > 0:
            self.allocated_gpus -= 1
            return True
        return False 