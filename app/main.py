"""
LoRA Dashboard FastAPI Backend
Serverless Training & Generation Suite
"""

import asyncio
import logging
import os
import shutil
import uuid
import yaml
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.models import (
    ApiResponse,
    GenerateRequest,
    TrainRequest,
    Process,
    LoRAModel,
    ProcessesResponse,
    LoRAResponse,
    UploadedFile,
    TrainingDataUploadResponse,
    BulkDownloadRequest,
    BulkDownloadResponse,
)
from app.services.gpu_manager import GPUManager
from app.services.process_manager import ProcessManager
from app.services.storage_service import StorageService
from app.services.lora_service import LoRAService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
gpu_manager: GPUManager = None
process_manager: ProcessManager = None
storage_service: StorageService = None
lora_service: LoRAService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services."""
    global gpu_manager, process_manager, storage_service, lora_service
    
    try:
        settings = get_settings()
        
        if settings.mock_mode:
            logger.info("🧪 Starting in MOCK MODE - using fake services for testing")
            from app.services.mock_services import (
                MockProcessManager, MockStorageService, 
                MockGPUManager, MockLoRAService
            )
            
            # Initialize mock services
            storage_service = MockStorageService()
            lora_service = MockLoRAService(storage_service)
            gpu_manager = MockGPUManager()
            process_manager = MockProcessManager()
            
            logger.info("✅ Mock services initialized - ready for testing!")
        else:
            logger.info("🚀 Starting in PRODUCTION MODE - using real services")
            
            # Initialize real services
            storage_service = StorageService()
            lora_service = LoRAService(storage_service)
            gpu_manager = GPUManager(max_concurrent=settings.max_concurrent_jobs)
            process_manager = ProcessManager(
                gpu_manager=gpu_manager,
                storage_service=storage_service,
                redis_url=settings.redis_url
            )
            
            await process_manager.initialize()
            logger.info("✅ Real services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up services...")
        if process_manager:
            await process_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="LoRA Dashboard API",
    description="Serverless Training & Generation Suite",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_process_manager() -> ProcessManager:
    if process_manager is None:
        raise HTTPException(status_code=500, detail="Process manager not initialized")
    return process_manager


def get_lora_service() -> LoRAService:
    if lora_service is None:
        raise HTTPException(status_code=500, detail="LoRA service not initialized")
    return lora_service


def get_storage_service() -> StorageService:
    if storage_service is None:
        raise HTTPException(status_code=500, detail="Storage service not initialized")
    return storage_service


# Initialize adapter conditionally
from app.core.config import get_settings
settings = get_settings()

if not settings.mock_mode:
    from app.adapters.runpod_adapter import RunPodAdapter
    adapter = RunPodAdapter()
else:
    adapter = None  # Will use direct services in mock mode

# Health endpoint
@app.get("/api/health", response_model=ApiResponse)
async def health_check() -> ApiResponse:
    """Health check endpoint."""
    try:
        if settings.mock_mode:
            # Mock mode - return mock health status
            return ApiResponse(
                success=True,
                data={
                    "status": "healthy",
                    "services": {
                        "process_manager": "healthy",
                        "storage": "healthy", 
                        "gpu_manager": gpu_manager.get_status() if gpu_manager else {"status": "mock"}
                    }
                },
                message="API is healthy (Mock Mode)"
            )
        else:
            result = await adapter.health_check()
            return ApiResponse(**result)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ApiResponse(
            success=False,
            error=f"Health check failed: {str(e)}"
        )


# Training endpoint
@app.post("/api/train", response_model=ApiResponse)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> ApiResponse:
    """Start a new training process."""
    try:
        if settings.mock_mode:
            # Mock mode - use mock service directly
            process_id = await process_manager.start_training(request.config)
            return ApiResponse(
                success=True,
                data={"process_id": process_id},
                message="Training process started successfully (Mock Mode)"
            )
        else:
            result = await adapter.start_training(request)
            return ApiResponse(**result)
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


# Generation endpoint
@app.post("/api/generate", response_model=ApiResponse)
async def start_generation(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
) -> ApiResponse:
    """Start a new generation process."""
    try:
        if settings.mock_mode:
            # Mock mode - use mock service directly
            process_id = await process_manager.start_generation(request.config)
            return ApiResponse(
                success=True,
                data={"process_id": process_id},
                message="Generation started successfully (Mock Mode)"
            )
        else:
            result = await adapter.start_generation(request)
            return ApiResponse(**result)
    except Exception as e:
        logger.error(f"Failed to start generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")


# Processes endpoint
@app.get("/api/processes", response_model=ProcessesResponse)
async def get_processes() -> ProcessesResponse:
    """Get all processes with their current status."""
    try:
        if settings.mock_mode:
            # Mock mode - use mock service directly
            processes = await process_manager.get_all_processes()
            return ProcessesResponse(processes=processes)
        else:
            result = await adapter.get_processes()
            return ProcessesResponse(processes=result.get("processes", []))
    except Exception as e:
        logger.error(f"Failed to get processes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get processes: {str(e)}")


# Process detail endpoint
@app.get("/api/processes/{process_id}", response_model=ApiResponse[Process])
async def get_process(process_id: str) -> ApiResponse[Process]:
    """Get specific process details."""
    try:
        result = await adapter.get_process(process_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
            
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get process {process_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get process: {str(e)}")


# Cancel process endpoint
@app.delete("/api/processes/{process_id}", response_model=ApiResponse)
async def cancel_process(process_id: str) -> ApiResponse:
    """Cancel a running process."""
    try:
        result = await adapter.cancel_process(process_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
            
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel process {process_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel process: {str(e)}")


# LoRA models endpoint
@app.get("/api/lora", response_model=LoRAResponse)
async def get_lora_models() -> LoRAResponse:
    """Get available LoRA models."""
    try:
        if settings.mock_mode:
            # Mock mode - use mock service directly
            models = await lora_service.get_available_models()
            return LoRAResponse(models=models)
        else:
            result = await adapter.get_lora_models()
            return LoRAResponse(models=result.get("models", []))
    except Exception as e:
        logger.error(f"Failed to get LoRA models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LoRA models: {str(e)}")


# Download endpoint
@app.get("/api/download/{process_id}", response_model=ApiResponse[Dict[str, str]])
async def get_download_url(process_id: str) -> ApiResponse[Dict[str, str]]:
    """Get presigned download URL for process results."""
    try:
        result = await adapter.get_download_url(process_id)
        if result.get("error"):
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=400, detail=result["error"])
        
        return ApiResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get download URL for {process_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get download URL: {str(e)}")


# 📁 FILE UPLOAD ENDPOINTS

@app.post("/api/upload/training-data", response_model=ApiResponse[TrainingDataUploadResponse])
async def upload_training_data(
    files: List[UploadFile] = File(...),
    training_name: str = Form(...),
    trigger_word: str = Form(...),
    cleanup_existing: bool = Form(default=True)
) -> ApiResponse[TrainingDataUploadResponse]:
    """Upload training images and captions for LoRA training."""
    try:
        settings = get_settings()
        
        # Create unique training folder
        training_id = str(uuid.uuid4())[:8]
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in training_name)
        training_folder = os.path.join(settings.workspace_path, "training_data", f"{safe_name}_{training_id}")
        
        # Clean up existing training data if requested
        if cleanup_existing:
            base_training_path = os.path.join(settings.workspace_path, "training_data")
            if os.path.exists(base_training_path):
                # Remove old training folders
                for item in os.listdir(base_training_path):
                    old_path = os.path.join(base_training_path, item)
                    if os.path.isdir(old_path):
                        shutil.rmtree(old_path)
                        logger.info(f"Cleaned up old training data: {old_path}")
        
        # Create training folder
        os.makedirs(training_folder, exist_ok=True)
        
        uploaded_files = []
        image_count = 0
        caption_count = 0
        
        # Process uploaded files
        for file in files:
            if not file.filename:
                continue
                
            # Save file to training folder
            file_path = os.path.join(training_folder, file.filename)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            file_info = UploadedFile(
                filename=file.filename,
                path=file_path,
                size=len(content),
                content_type=file.content_type or "application/octet-stream",
                uploaded_at=datetime.now()
            )
            uploaded_files.append(file_info)
            
            # Count file types
            if file.content_type and file.content_type.startswith('image/'):
                image_count += 1
            elif file.filename.endswith('.txt'):
                caption_count += 1
        
        # Create trigger word info file
        trigger_file = os.path.join(training_folder, "_training_info.txt")
        with open(trigger_file, "w") as f:
            f.write(f"Training Name: {training_name}\n")
            f.write(f"Trigger Word: {trigger_word}\n")
            f.write(f"Upload Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Images: {image_count}\n")
            f.write(f"Total Captions: {caption_count}\n")
        
        response_data = TrainingDataUploadResponse(
            uploaded_files=uploaded_files,
            training_folder=training_folder,
            total_images=image_count,
            total_captions=caption_count,
            message=f"Successfully uploaded {len(uploaded_files)} files to {training_folder}"
        )
        
        logger.info(f"Training data uploaded: {training_folder} ({image_count} images, {caption_count} captions)")
        
        return ApiResponse(
            success=True,
            data=response_data,
            message="Training data uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to upload training data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload training data: {str(e)}")


@app.post("/api/download/bulk", response_model=ApiResponse[BulkDownloadResponse])
async def bulk_download(request: BulkDownloadRequest) -> ApiResponse[BulkDownloadResponse]:
    """Create bulk download URLs for multiple processes."""
    try:
        storage_service = get_storage_service()
        download_items = []
        total_size = 0
        
        for process_id in request.process_ids:
            # List files for this process
            files = await storage_service.list_files(f"results/{process_id}/")
            
            for file_info in files:
                file_type = "other"
                if any(ext in file_info['key'].lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                    if request.include_images:
                        file_type = "image"
                    else:
                        continue
                elif any(ext in file_info['key'].lower() for ext in ['.safetensors', '.ckpt', '.pt']):
                    if request.include_loras:
                        file_type = "lora"
                    else:
                        continue
                
                # Generate download URL
                download_url = await storage_service.get_download_url(process_id)
                if download_url:
                    download_items.append({
                        "filename": os.path.basename(file_info['key']),
                        "url": download_url,
                        "size": file_info.get('size', 0),
                        "type": file_type
                    })
                    total_size += file_info.get('size', 0)
        
        response_data = BulkDownloadResponse(
            download_items=download_items,
            zip_url=None,  # TODO: Implement zip creation if needed
            total_files=len(download_items),
            total_size=total_size
        )
        
        return ApiResponse(
            success=True,
            data=response_data,
            message=f"Generated {len(download_items)} download URLs"
        )
        
    except Exception as e:
        logger.error(f"Failed to create bulk download: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create bulk download: {str(e)}")


# 📝 LOGGING ENDPOINTS

@app.get("/api/logs/stats", response_model=ApiResponse[Dict])
async def get_log_stats() -> ApiResponse[Dict]:
    """Get logging statistics"""
    try:
        from app.core.logger import get_logger
        logger_instance = get_logger()
        stats = logger_instance.get_log_stats()
        
        return ApiResponse(
            success=True,
            data=stats,
            message="Log statistics retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get log stats: {str(e)}")

@app.get("/api/logs/tail/{log_type}")
async def tail_logs(log_type: str, lines: int = 100) -> ApiResponse[Dict]:
    """Get last N lines from log file"""
    try:
        from app.core.logger import get_logger
        logger_instance = get_logger()
        
        log_files = {
            "app": "app.log",
            "requests": "requests.log",
            "errors": "errors.log"
        }
        
        if log_type not in log_files:
            raise HTTPException(status_code=400, detail=f"Invalid log type. Available: {list(log_files.keys())}")
        
        log_file = logger_instance.log_dir / log_files[log_type]
        
        if not log_file.exists():
            return ApiResponse(
                success=True,
                data={"lines": [], "message": f"Log file {log_type} not found"},
                message="No logs available"
            )
        
        # Read last N lines
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
            return ApiResponse(
                success=True,
                data={
                    "lines": [line.strip() for line in last_lines],
                    "total_lines": len(all_lines),
                    "returned_lines": len(last_lines),
                    "log_type": log_type
                },
                message=f"Retrieved last {len(last_lines)} lines from {log_type} log"
            )
        except UnicodeDecodeError:
            # Try with different encoding
            with open(log_file, 'r', encoding='latin-1') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
            return ApiResponse(
                success=True,
                data={
                    "lines": [line.strip() for line in last_lines],
                    "total_lines": len(all_lines),
                    "returned_lines": len(last_lines),
                    "log_type": log_type,
                    "encoding": "latin-1"
                },
                message=f"Retrieved last {len(last_lines)} lines from {log_type} log"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to tail logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tail logs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    ) 