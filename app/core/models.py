"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from enum import Enum

from pydantic import BaseModel, Field


# Generic type for API responses
T = TypeVar('T')


class ProcessStatus(str, Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessType(str, Enum):
    """Process type enumeration."""
    TRAINING = "training"
    GENERATION = "generation"


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response model."""
    success: bool
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class TrainRequest(BaseModel):
    """Training request model."""
    config: str = Field(..., description="YAML configuration for training")


class GenerateRequest(BaseModel):
    """Generation request model."""
    config: str = Field(..., description="YAML configuration for generation")


# 📁 FILE UPLOAD MODELS
class UploadedFile(BaseModel):
    """Uploaded file information."""
    filename: str
    path: str
    size: int
    content_type: str
    uploaded_at: datetime


class TrainingDataUploadResponse(BaseModel):
    """Response for training data upload."""
    uploaded_files: List[UploadedFile]
    training_folder: str
    total_images: int
    total_captions: int
    message: str


class BulkDownloadRequest(BaseModel):
    """Request for bulk download."""
    process_ids: List[str] = Field(..., description="List of process IDs to download")
    include_images: bool = Field(default=True, description="Include generated images")
    include_loras: bool = Field(default=True, description="Include LoRA models")


class DownloadItem(BaseModel):
    """Individual download item."""
    filename: str
    url: str
    size: Optional[int] = None
    type: str  # 'image', 'lora', 'other'


class BulkDownloadResponse(BaseModel):
    """Response for bulk download."""
    download_items: List[DownloadItem]
    zip_url: Optional[str] = None
    total_files: int
    total_size: int


class Process(BaseModel):
    """Process model."""
    id: str
    name: str
    type: ProcessType
    status: ProcessStatus
    progress: float = Field(ge=0, le=100, description="Progress percentage")
    created_at: datetime
    updated_at: datetime
    elapsed_time: Optional[int] = Field(None, description="Elapsed time in seconds")
    gpu_id: Optional[str] = Field(None, description="Assigned GPU ID")
    step: Optional[int] = Field(None, description="Current step")
    total_steps: Optional[int] = Field(None, description="Total steps")
    eta: Optional[int] = Field(None, description="Estimated time remaining in seconds")
    config: Optional[Dict[str, Any]] = Field(None, description="Process configuration")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_path: Optional[str] = Field(None, description="Output file path")


class LoRAMetadata(BaseModel):
    """LoRA model metadata."""
    steps: Optional[int] = None
    trigger_word: Optional[str] = None
    model_type: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None


class LoRAModel(BaseModel):
    """LoRA model representation."""
    id: str
    name: str
    path: str
    created_at: datetime
    size: int = Field(..., description="File size in bytes")
    metadata: Optional[LoRAMetadata] = None


class ProcessesResponse(BaseModel):
    """Response for processes endpoint."""
    processes: List[Process]


class LoRAResponse(BaseModel):
    """Response for LoRA models endpoint."""
    models: List[LoRAModel]


class GPUInfo(BaseModel):
    """GPU information."""
    id: str
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: float
    temperature: Optional[float] = None
    is_available: bool


class GPUStatus(BaseModel):
    """GPU status information."""
    total_gpus: int
    available_gpus: int
    busy_gpus: int
    gpus: List[GPUInfo]


class HealthStatus(BaseModel):
    """Health check status."""
    status: str
    services: Dict[str, Any]
    timestamp: datetime


class JobConfig(BaseModel):
    """Job configuration base model."""
    name: str
    device: str = "cuda:0"
    output_folder: str


class TrainingJobConfig(JobConfig):
    """Training job configuration."""
    trigger_word: str
    training_folder: str
    network: Dict[str, Any]
    save: Dict[str, Any]
    datasets: List[Dict[str, Any]]
    train: Dict[str, Any]
    model: Dict[str, Any]
    sample: Dict[str, Any]


class GenerationJobConfig(JobConfig):
    """Generation job configuration."""
    generate: Dict[str, Any]
    model: Dict[str, Any] 