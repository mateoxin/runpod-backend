"""
LoRA Service
Manages LoRA model discovery and metadata
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid

from app.core.models import LoRAModel, LoRAMetadata
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class LoRAService:
    """Service for managing LoRA models."""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.local_models_path = "/workspace/ai-toolkit/output"
        self.models_cache: Dict[str, LoRAModel] = {}
        self._last_scan = None
        self._scan_interval = 300  # 5 minutes
        
    async def get_available_models(self) -> List[LoRAModel]:
        """Get all available LoRA models."""
        # Check if we need to rescan
        now = datetime.now(timezone.utc)
        if (self._last_scan is None or 
            (now - self._last_scan).total_seconds() > self._scan_interval):
            await self._scan_models()
            self._last_scan = now
            
        return list(self.models_cache.values())
        
    async def _scan_models(self) -> None:
        """Scan for available LoRA models."""
        try:
            # Scan local directory
            await self._scan_local_models()
            
            # Scan storage
            await self._scan_storage_models()
            
            logger.info(f"Found {len(self.models_cache)} LoRA models")
            
        except Exception as e:
            logger.error(f"Failed to scan models: {e}")
            
    async def _scan_local_models(self) -> None:
        """Scan local models directory."""
        if not os.path.exists(self.local_models_path):
            return
            
        try:
            for root, dirs, files in os.walk(self.local_models_path):
                for file in files:
                    if file.endswith('.safetensors'):
                        await self._process_local_model(root, file)
                        
        except Exception as e:
            logger.error(f"Failed to scan local models: {e}")
            
    async def _process_local_model(self, root: str, filename: str) -> None:
        """Process a local LoRA model file."""
        try:
            file_path = os.path.join(root, filename)
            
            # Get file stats
            stat = os.stat(file_path)
            
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Try to load metadata
            metadata = await self._load_model_metadata(root, filename)
            
            # Create model object
            model = LoRAModel(
                id=model_id,
                name=os.path.splitext(filename)[0],
                path=file_path,
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                size=stat.st_size,
                metadata=metadata
            )
            
            self.models_cache[model_id] = model
            
        except Exception as e:
            logger.error(f"Failed to process local model {filename}: {e}")
            
    async def _load_model_metadata(self, root: str, filename: str) -> Optional[LoRAMetadata]:
        """Load metadata for a LoRA model."""
        try:
            # Look for associated metadata files
            base_name = os.path.splitext(filename)[0]
            
            # Check for JSON metadata file
            json_path = os.path.join(root, f"{base_name}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                return LoRAMetadata(
                    steps=data.get('steps'),
                    trigger_word=data.get('trigger_word'),
                    model_type=data.get('model_type'),
                    training_config=data.get('training_config')
                )
                
            # Check for YAML config file
            yaml_path = os.path.join(root, f"{base_name}_config.yaml")
            if os.path.exists(yaml_path):
                # Parse YAML to extract metadata
                import yaml
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Extract relevant metadata from config
                metadata = LoRAMetadata()
                
                if 'config' in config and 'process' in config['config']:
                    process_config = config['config']['process'][0]
                    
                    # Get trigger word
                    if 'trigger_word' in process_config:
                        metadata.trigger_word = process_config['trigger_word']
                        
                    # Get training steps
                    if 'train' in process_config and 'steps' in process_config['train']:
                        metadata.steps = process_config['train']['steps']
                        
                    # Get model type
                    if 'model' in process_config and 'name_or_path' in process_config['model']:
                        model_path = process_config['model']['name_or_path']
                        if 'FLUX' in model_path:
                            metadata.model_type = 'FLUX'
                        elif 'SDXL' in model_path:
                            metadata.model_type = 'SDXL'
                            
                    metadata.training_config = config
                    
                return metadata
                
        except Exception as e:
            logger.debug(f"Failed to load metadata for {filename}: {e}")
            
        return None
        
    async def _scan_storage_models(self) -> None:
        """Scan storage for LoRA models."""
        try:
            # List files in storage with lora prefix
            files = await self.storage_service.list_files("lora/")
            
            for file_info in files:
                if file_info['key'].endswith('.safetensors'):
                    await self._process_storage_model(file_info)
                    
        except Exception as e:
            logger.error(f"Failed to scan storage models: {e}")
            
    async def _process_storage_model(self, file_info: Dict[str, Any]) -> None:
        """Process a LoRA model in storage."""
        try:
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Extract name from key
            key = file_info['key']
            name = os.path.splitext(os.path.basename(key))[0]
            
            # Create model object
            model = LoRAModel(
                id=model_id,
                name=name,
                path=f"s3://{self.storage_service.settings.s3_bucket}/{key}",
                created_at=file_info['last_modified'],
                size=file_info['size'],
                metadata=None  # Could be enhanced to load metadata from storage
            )
            
            self.models_cache[model_id] = model
            
        except Exception as e:
            logger.error(f"Failed to process storage model {file_info['key']}: {e}")
            
    async def get_model_by_id(self, model_id: str) -> Optional[LoRAModel]:
        """Get a specific model by ID."""
        return self.models_cache.get(model_id)
        
    async def refresh_cache(self) -> None:
        """Force refresh of models cache."""
        self.models_cache.clear()
        await self._scan_models()
        self._last_scan = datetime.now(timezone.utc)
        
    async def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about available models."""
        models = await self.get_available_models()
        
        total_size = sum(model.size for model in models)
        model_types = {}
        
        for model in models:
            if model.metadata and model.metadata.model_type:
                model_type = model.metadata.model_type
                model_types[model_type] = model_types.get(model_type, 0) + 1
                
        return {
            "total_models": len(models),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "model_types": model_types,
            "local_models": len([m for m in models if not m.path.startswith('s3://')]),
            "storage_models": len([m for m in models if m.path.startswith('s3://')])
        } 