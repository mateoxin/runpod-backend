#!/usr/bin/env python3
"""
🔧 Pydantic Models for Data Validation
Ensures type safety and validation for all API inputs/outputs
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
import yaml
import base64
import os

# Configuration constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_FILES_PER_UPLOAD = 100
MAX_CONFIG_SIZE = 10 * 1024 * 1024  # 10MB for YAML config
ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
ALLOWED_CAPTION_EXTENSIONS = ['.txt']
ALLOWED_LORA_EXTENSIONS = ['.safetensors', '.ckpt', '.pt', '.pth']

class FileUpload(BaseModel):
    """Validates individual file upload"""
    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., description="Base64 encoded file content")
    content_type: Optional[str] = "application/octet-stream"
    
    @validator('filename')
    def validate_filename(cls, v):
        """Ensure filename is safe and has valid extension"""
        # Remove any path traversal attempts
        v = os.path.basename(v)
        if not v or v.startswith('.'):
            raise ValueError("Invalid filename")
        
        # Check for valid extensions
        ext = os.path.splitext(v)[1].lower()
        valid_extensions = ALLOWED_IMAGE_EXTENSIONS + ALLOWED_CAPTION_EXTENSIONS + ['.yaml', '.yml', '.json']
        if ext and ext not in valid_extensions:
            raise ValueError(f"File extension {ext} not allowed")
        
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """Validate base64 content and check size"""
        try:
            # Add padding if missing
            missing_padding = len(v) % 4
            if missing_padding:
                v += '=' * (4 - missing_padding)
            
            # Decode to check validity and size
            decoded = base64.b64decode(v)
            if len(decoded) > MAX_FILE_SIZE:
                raise ValueError(f"File too large: {len(decoded)} bytes (max: {MAX_FILE_SIZE})")
            
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 content: {str(e)}")

class TrainingConfig(BaseModel):
    """Validates training configuration"""
    type: Literal["train", "train_with_yaml"]
    config: Optional[str] = Field(None, description="YAML configuration as string")
    yaml_config: Optional[str] = Field(None, description="Alternative field for YAML config")
    
    model_config = ConfigDict(extra='allow')
    
    @validator('config', 'yaml_config', pre=True)
    def validate_yaml_config(cls, v):
        """Validate YAML configuration"""
        if v is None:
            return v
            
        if len(v) > MAX_CONFIG_SIZE:
            raise ValueError(f"Config too large: {len(v)} bytes (max: {MAX_CONFIG_SIZE})")
        
        try:
            # Parse YAML to validate structure
            config_data = yaml.safe_load(v)
            
            # Basic structure validation
            if not isinstance(config_data, dict):
                raise ValueError("Config must be a YAML dictionary")
            
            # Check for required fields based on config type
            if 'config' in config_data:
                process_config = config_data['config']
                if 'process' not in process_config:
                    raise ValueError("Missing 'process' in config")
                
                # Validate process list
                if not isinstance(process_config['process'], list):
                    raise ValueError("'process' must be a list")
                
                for process in process_config['process']:
                    if 'type' not in process:
                        raise ValueError("Each process must have a 'type'")
            
            return v
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML config: {e}")

class GenerationConfig(BaseModel):
    """Validates generation configuration"""
    type: Literal["generate"]
    config: Optional[str] = Field(None, description="YAML configuration")
    prompt: Optional[str] = Field(None, max_length=2000)
    model_path: Optional[str] = Field(None, description="Path or S3 URI to LoRA model")
    
    model_config = ConfigDict(extra='allow')
    
    @validator('config', 'prompt')
    def validate_has_config_or_prompt(cls, v, values):
        """Ensure either config or prompt is provided"""
        if not v and not values.get('prompt') and not values.get('config'):
            raise ValueError("Either 'config' or 'prompt' must be provided")
        return v

class UploadTrainingData(BaseModel):
    """Validates training data upload request"""
    type: Literal["upload_training_data"]
    training_name: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$")
    trigger_word: Optional[str] = Field("", max_length=100)
    cleanup_existing: Optional[bool] = True
    files: List[FileUpload] = Field(..., min_items=1, max_items=MAX_FILES_PER_UPLOAD)
    
    @validator('files')
    def validate_files(cls, v):
        """Validate files have proper pairs of images and captions"""
        image_count = sum(1 for f in v if any(f.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS))
        caption_count = sum(1 for f in v if any(f.filename.lower().endswith(ext) for ext in ALLOWED_CAPTION_EXTENSIONS))
        
        if image_count == 0:
            raise ValueError("At least one image file is required")
        
        # Warning, not error - captions are optional but recommended
        if caption_count < image_count:
            print(f"Warning: {image_count} images but only {caption_count} captions")
        
        return v

class ProcessStatusRequest(BaseModel):
    """Validates process status request"""
    type: Literal["process_status", "cancel", "download"]
    process_id: str = Field(..., min_length=1, pattern="^[a-zA-Z0-9_-]+$")

class BulkDownloadRequest(BaseModel):
    """Validates bulk download request"""
    type: Literal["bulk_download"]
    process_ids: List[str] = Field(..., min_items=1, max_items=50)
    include_images: Optional[bool] = True
    include_loras: Optional[bool] = True

class DownloadFileRequest(BaseModel):
    """Validates direct file download request"""
    type: Literal["download_file"]
    file_path: str = Field(..., min_length=1)
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Ensure file path is within workspace"""
        # Normalize path to prevent traversal
        v = os.path.normpath(v)
        
        # Must be absolute path in workspace
        if not v.startswith('/workspace/'):
            if v.startswith('workspace/'):
                v = '/' + v
            elif not v.startswith('/'):
                v = '/workspace/' + v
            else:
                raise ValueError("File must be in /workspace directory")
        
        # Prevent path traversal
        if '..' in v:
            raise ValueError("Path traversal not allowed")
        
        return v

class ListFilesRequest(BaseModel):
    """Validates list files request"""
    type: Literal["list_files"]
    path: Optional[str] = Field("", description="Optional path filter")

class HealthCheckRequest(BaseModel):
    """Validates health check request"""
    type: Literal["health"]

class ProcessListRequest(BaseModel):
    """Validates process list request"""  
    type: Literal["processes"]

class LoRAModelsRequest(BaseModel):
    """Validates LoRA models list request"""
    type: Literal["lora", "list_models"]

# Union type for all requests
RequestTypes = Union[
    TrainingConfig,
    GenerationConfig,
    UploadTrainingData,
    ProcessStatusRequest,
    BulkDownloadRequest,
    DownloadFileRequest,
    ListFilesRequest,
    HealthCheckRequest,
    ProcessListRequest,
    LoRAModelsRequest
]

def validate_request(job_input: Dict[str, Any]) -> RequestTypes:
    """
    Validates incoming request and returns appropriate model instance
    """
    job_type = job_input.get("type")
    
    # Auto-detect generation from prompt
    if not job_type and job_input.get("prompt"):
        job_input["type"] = "generate"
        job_type = "generate"
    
    if not job_type:
        raise ValueError("Missing 'type' field in request")
    
    # Map type to model class
    type_mapping = {
        "train": TrainingConfig,
        "train_with_yaml": TrainingConfig,
        "generate": GenerationConfig,
        "upload_training_data": UploadTrainingData,
        "process_status": ProcessStatusRequest,
        "cancel": ProcessStatusRequest,
        "download": ProcessStatusRequest,
        "bulk_download": BulkDownloadRequest,
        "download_file": DownloadFileRequest,
        "list_files": ListFilesRequest,
        "health": HealthCheckRequest,
        "processes": ProcessListRequest,
        "lora": LoRAModelsRequest,
        "list_models": LoRAModelsRequest
    }
    
    model_class = type_mapping.get(job_type)
    if not model_class:
        raise ValueError(f"Unknown job type: {job_type}")
    
    # Validate with appropriate model
    return model_class(**job_input)

# Process state models
class ProcessInfo(BaseModel):
    """Process information model"""
    id: str
    type: Literal["training", "generation"]
    status: Literal["starting", "running", "completed", "failed", "cancelled"]
    config: Dict[str, Any]
    created_at: str
    updated_at: str
    output_path: Optional[str] = None
    error: Optional[str] = None
    s3_path: Optional[str] = None
    worker_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

from typing import Union
