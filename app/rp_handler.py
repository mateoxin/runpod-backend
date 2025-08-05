#!/usr/bin/env python3
"""
🚀 ULTRA-FAST RUNPOD HANDLER - LoRA Dashboard Backend
Minimal handler with runtime setup for heavy dependencies
Deploy time: ~30 seconds instead of 20 minutes!
Based on successful runpod-fastbackend/ approach
"""

import runpod
import asyncio
import logging
import os
import sys
import subprocess
import time
from datetime import datetime
from typing import Dict, Any

# Global flag to track if environment is setup
ENVIRONMENT_READY = False
SETUP_LOCK = False

def log(message, level="INFO"):
    """Unified logging to stdout and stderr for RunPod visibility"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {message}"
    
    # Write to both stdout and stderr for maximum visibility
    print(log_msg)
    sys.stderr.write(f"{log_msg}\n")
    sys.stderr.flush()
    sys.stdout.flush()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup heavy dependencies at runtime (wykorzystuje RunPod cache)"""
    global ENVIRONMENT_READY, SETUP_LOCK
    
    if ENVIRONMENT_READY:
        log("Environment already ready", "INFO")
        return True
        
    if SETUP_LOCK:
        log("Environment setup in progress...", "WARN")
        # Wait for setup to complete
        while SETUP_LOCK and not ENVIRONMENT_READY:
            time.sleep(1)
        return ENVIRONMENT_READY
    
    SETUP_LOCK = True
    
    try:
        log("🚀 Setting up environment at runtime...", "INFO")
        
        # Step 1: Install Redis and AWS dependencies
        log("📦 Installing Redis and AWS dependencies...", "INFO")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "redis>=5.0.1", "boto3>=1.34.0", "asyncio-subprocess>=0.1.0"
        ], capture_output=False, text=True)
        
        if result.returncode != 0:
            log(f"❌ Redis/AWS install failed with code {result.returncode}", "ERROR")
            return False
        
        log("✅ Redis and AWS dependencies installed successfully", "INFO")
        
        # Step 2: Setup directories
        log("📁 Creating workspace directories...", "INFO")
        os.makedirs("/workspace", exist_ok=True)
        os.makedirs("/workspace/training_data", exist_ok=True)
        os.makedirs("/workspace/models", exist_ok=True)
        os.makedirs("/workspace/logs", exist_ok=True)
        
        # Step 3: Setup HuggingFace token (FROM ENVIRONMENT)
        # Get HF token from RunPod environment variables
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token and hf_token != "":
            log("🤗 Setting up HuggingFace token...", "INFO")
            try:
                subprocess.run([
                    "huggingface-cli", "login", "--token", hf_token
                ], capture_output=True, text=True, timeout=30)
                log("✅ HuggingFace token configured", "INFO")
            except subprocess.TimeoutExpired:
                log("⚠️ HuggingFace login timeout, continuing...", "WARN")
            except Exception as e:
                log(f"⚠️ HuggingFace login failed: {e}", "WARN")
        else:
            log("⚠️ No HuggingFace token provided", "WARN")
        
        # Step 4: Install AI-Toolkit
        log("🤖 Installing AI-Toolkit (ostris/ai-toolkit)...", "INFO")
        toolkit_path = "/workspace/ai-toolkit"
        
        if not os.path.exists(toolkit_path):
            try:
                log("📥 Cloning AI-Toolkit repository...", "INFO")
                result = subprocess.run([
                    "git", "clone", 
                    "https://github.com/ostris/ai-toolkit.git",
                    toolkit_path
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    log(f"❌ AI-Toolkit clone failed: {result.stderr}", "ERROR")
                    return False
                
                log("✅ AI-Toolkit repository cloned successfully", "INFO")
                
                # Install AI-Toolkit dependencies
                log("📦 Installing AI-Toolkit dependencies...", "INFO")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", 
                    f"{toolkit_path}/requirements.txt"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    log(f"⚠️ AI-Toolkit dependencies install warning: {result.stderr}", "WARN")
                    log("📄 Continuing with manual torch installation...", "INFO")
                    
                    # Install essential packages manually
                    essential_packages = [
                        "torch>=2.0.0",
                        "torchvision>=0.15.0",
                        "diffusers>=0.21.0",
                        "transformers>=4.25.0",
                        "accelerate>=0.20.0",
                        "xformers",
                        "safetensors",
                        "Pillow",
                        "numpy",
                        "tqdm"
                    ]
                    
                    for package in essential_packages:
                        log(f"📦 Installing {package}...", "INFO")
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True, timeout=120)
                        
                        if result.returncode == 0:
                            log(f"✅ {package} installed successfully", "INFO")
                        else:
                            log(f"⚠️ {package} install failed, continuing...", "WARN")
                else:
                    log("✅ AI-Toolkit dependencies installed successfully", "INFO")
                    
            except subprocess.TimeoutExpired:
                log("❌ AI-Toolkit installation timeout", "ERROR")
                return False
            except Exception as e:
                log(f"❌ AI-Toolkit installation failed: {e}", "ERROR")
                return False
        else:
            log("✅ AI-Toolkit already available", "INFO")
        
        log("✅ Environment setup completed successfully!", "INFO")
        ENVIRONMENT_READY = True
        return True
        
    except Exception as e:
        log(f"❌ Setup error: {e}", "ERROR")
        return False
    finally:
        SETUP_LOCK = False

def lazy_import_services():
    """Import services only after environment is ready"""
    try:
        # Import enhanced logger
        from app.core.logger import get_logger
        enhanced_logger = get_logger()
        
        # Import services
        from app.services.gpu_manager import GPUManager
        from app.services.process_manager import ProcessManager
        from app.services.storage_service import StorageService
        from app.services.lora_service import LoRAService
        from app.core.config import get_settings
        
        return {
            'enhanced_logger': enhanced_logger,
            'GPUManager': GPUManager,
            'ProcessManager': ProcessManager,
            'StorageService': StorageService,
            'LoRAService': LoRAService,
            'get_settings': get_settings
        }
    except ImportError as e:
        log(f"Failed to import services: {e}", "ERROR")
        return None

# Global service instances (initialized on first use)
_services_initialized = False
_services = None
_gpu_manager = None
_process_manager = None
_storage_service = None
_lora_service = None
_enhanced_logger = None

async def initialize_services():
    """Initialize services once with runtime setup"""
    global _services_initialized, _services, _gpu_manager, _process_manager, _storage_service, _lora_service, _enhanced_logger
    
    if _services_initialized:
        return
    
    try:
        # First setup environment
        if not setup_environment():
            raise Exception("Failed to setup runtime environment")
        
        # Then import services
        _services = lazy_import_services()
        if not _services:
            raise Exception("Failed to import services")
        
        log("🚀 Initializing LoRA Dashboard services...", "INFO")
        
        _enhanced_logger = _services['enhanced_logger']
        settings = _services['get_settings']() if _services['get_settings'] else None
        
        # Initialize services
        _storage_service = _services['StorageService']() if _services['StorageService'] else None
        _lora_service = _services['LoRAService'](_storage_service) if _services['LoRAService'] and _storage_service else None
        _gpu_manager = _services['GPUManager'](max_concurrent=10) if _services['GPUManager'] else None
        _process_manager = _services['ProcessManager'](
            gpu_manager=_gpu_manager,
            storage_service=_storage_service,
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        ) if _services['ProcessManager'] else None
        
        if _process_manager:
            await _process_manager.initialize()
        
        _services_initialized = True
        log("✅ Services initialized successfully", "INFO")
        
    except Exception as e:
        log(f"❌ Failed to initialize services: {e}", "ERROR")
        raise

async def async_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod Serverless handler function
    
    Expected input format:
    {
        "input": {
            "type": "train" | "generate" | "health" | "processes" | "lora",
            "config": "YAML configuration string",
            "process_id": "optional process ID for status check"
        }
    }
    """
    request_id = None
    try:
        # Initialize services if needed
        await initialize_services()
        
        job_input = event.get("input", {})
        job_type = job_input.get("type")
        
        # Log incoming request
        request_id = None
        if _enhanced_logger:
            request_id = _enhanced_logger.log_request(
                request_type=job_type or "unknown",
                request_data=job_input,
                endpoint="runpod_serverless"
            )
        
        log(f"📨 Processing job type: {job_type} | Request ID: {request_id}", "INFO")
        
        if job_type == "health":
            response = await handle_health_check()
        elif job_type == "train":
            response = await handle_training(job_input)
        elif job_type == "generate":
            response = await handle_generation(job_input)
        elif job_type == "processes":
            response = await handle_get_processes(job_input)
        elif job_type == "process_status":
            response = await handle_process_status(job_input)
        elif job_type == "lora":
            response = await handle_get_lora_models()
        elif job_type == "cancel":
            response = await handle_cancel_process(job_input)
        elif job_type == "download":
            response = await handle_download_url(job_input)
        elif job_type == "upload_training_data":
            response = await handle_upload_training_data(job_input, request_id)
        elif job_type == "bulk_download":
            response = await handle_bulk_download(job_input)
        else:
            response = {
                "error": f"Unknown job type: {job_type}",
                "supported_types": ["health", "train", "generate", "processes", "process_status", "lora", "cancel", "download", "upload_training_data", "bulk_download"]
            }
        
        # Log successful response
        if _enhanced_logger and request_id:
            _enhanced_logger.log_response(
                request_id=request_id,
                response_data=response,
                status_code=200 if not response.get("error") else 400
            )
        
        return response
            
    except Exception as e:
        log(f"💥 Handler error: {e}", "ERROR")
        error_response = {"error": str(e)}
        
        # Log error response
        if _enhanced_logger and request_id:
            _enhanced_logger.log_error(e, {"event": event}, request_id)
            _enhanced_logger.log_response(
                request_id=request_id,
                response_data=error_response,
                status_code=500,
                error=str(e)
            )
        
        return error_response

async def handle_health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        services_status = {}
        
        if _process_manager:
            services_status["process_manager"] = "healthy"
        if _storage_service:
            services_status["storage"] = await _storage_service.health_check()
        if _gpu_manager:
            services_status["gpu_manager"] = _gpu_manager.get_status()
            
        return {
            "status": "healthy",
            "services": services_status,
            "worker_id": os.environ.get("RUNPOD_WORKER_ID", "local"),
            "environment": "serverless"
        }
    except Exception as e:
        return {"error": f"Health check failed: {str(e)}"}

async def handle_training(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle training request"""
    try:
        config = job_input.get("config")
        if not config:
            return {"error": "Missing 'config' parameter"}
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        process_id = await _process_manager.start_training(config)
        log(f"✅ Training started with process_id: {process_id}", "INFO")
        
        return {"process_id": process_id}
    except Exception as e:
        log(f"❌ Training error: {e}", "ERROR")
        return {"error": f"Failed to start training: {str(e)}"}

async def handle_generation(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle generation request"""
    try:
        config = job_input.get("config")
        if not config:
            return {"error": "Missing 'config' parameter"}
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        process_id = await _process_manager.start_generation(config)
        log(f"✅ Generation started with process_id: {process_id}", "INFO")
        
        return {"process_id": process_id}
    except Exception as e:
        log(f"❌ Generation error: {e}", "ERROR")
        return {"error": f"Failed to start generation: {str(e)}"}

async def handle_get_processes(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get processes request"""
    try:
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        processes = await _process_manager.get_all_processes()
        return {"processes": processes}
    except Exception as e:
        log(f"❌ Get processes error: {e}", "ERROR")
        return {"error": f"Failed to get processes: {str(e)}"}

async def handle_process_status(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle process status request"""
    try:
        process_id = job_input.get("process_id")
        if not process_id:
            return {"error": "Missing 'process_id' parameter"}
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        process = await _process_manager.get_process(process_id)
        if not process:
            return {"error": "Process not found"}
            
        return process
    except Exception as e:
        log(f"❌ Process status error: {e}", "ERROR")
        return {"error": f"Failed to get process status: {str(e)}"}

async def handle_get_lora_models() -> Dict[str, Any]:
    """Handle get LoRA models request"""
    try:
        if not _lora_service:
            return {"error": "LoRA service not initialized"}
        
        models = await _lora_service.get_available_models()
        return {"models": models}
    except Exception as e:
        log(f"❌ Get LoRA models error: {e}", "ERROR")
        return {"error": f"Failed to get LoRA models: {str(e)}"}

async def handle_cancel_process(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle cancel process request"""
    try:
        process_id = job_input.get("process_id")
        if not process_id:
            return {"error": "Missing 'process_id' parameter"}
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        success = await _process_manager.cancel_process(process_id)
        if not success:
            return {"error": "Process not found or cannot be cancelled"}
            
        log(f"✅ Process {process_id} cancelled successfully", "INFO")
        return {"message": "Process cancelled successfully"}
    except Exception as e:
        log(f"❌ Cancel process error: {e}", "ERROR")
        return {"error": f"Failed to cancel process: {str(e)}"}

async def handle_download_url(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle download URL request"""
    try:
        process_id = job_input.get("process_id")
        if not process_id:
            return {"error": "Missing 'process_id' parameter"}
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        # Check if process exists and is completed
        process = await _process_manager.get_process(process_id)
        if not process:
            return {"error": "Process not found"}
            
        if process.status != "completed":
            return {"error": "Process not completed"}
        
        if not _storage_service:
            return {"error": "Storage service not initialized"}
            
        # Get download URL
        url = await _storage_service.get_download_url(process_id)
        
        return {"url": url}
    except Exception as e:
        log(f"❌ Download URL error: {e}", "ERROR")
        return {"error": f"Failed to get download URL: {str(e)}"}

async def handle_upload_training_data(job_input: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
    """Handle training data upload request"""
    try:
        import os
        import shutil
        import uuid
        from datetime import datetime
        
        # Get upload parameters from job input
        training_name = job_input.get("training_name", f"training_{int(datetime.now().timestamp())}")
        trigger_word = job_input.get("trigger_word", "")
        cleanup_existing = job_input.get("cleanup_existing", True)
        files_data = job_input.get("files", [])
        
        if not files_data:
            return {"error": "No files provided"}
        
        # Create unique training folder
        training_id = str(uuid.uuid4())[:8]
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in training_name)
        workspace_path = os.environ.get("WORKSPACE_PATH", "/workspace")
        training_folder = os.path.join(workspace_path, "training_data", f"{safe_name}_{training_id}")
        
        # Clean up existing training data if requested
        if cleanup_existing:
            base_training_path = os.path.join(workspace_path, "training_data")
            if os.path.exists(base_training_path):
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
        
        # Process files (expecting base64 encoded files)
        for file_info in files_data:
            filename = file_info.get("filename")
            content = file_info.get("content")  # base64 encoded
            content_type = file_info.get("content_type", "application/octet-stream")
            
            if not filename or not content:
                continue
                
            # Save file to training folder
            file_path = os.path.join(training_folder, filename)
            
            # Decode base64 content and save
            import base64
            try:
                file_content = base64.b64decode(content)
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                file_data = {
                    "filename": filename,
                    "path": file_path,
                    "size": len(file_content),
                    "content_type": content_type,
                    "uploaded_at": datetime.now().isoformat()
                }
                uploaded_files.append(file_data)
                
                # Log file operation
                if _enhanced_logger:
                    _enhanced_logger.log_file_operation(
                        operation="upload",
                        file_info=file_data,
                        request_id=request_id
                    )
                
                # Count file types
                if content_type and content_type.startswith('image/'):
                    image_count += 1
                elif filename.endswith('.txt'):
                    caption_count += 1
                    
            except Exception as e:
                log(f"❌ Failed to process file {filename}: {e}", "ERROR")
                if _enhanced_logger:
                    _enhanced_logger.log_error(e, {"filename": filename}, request_id)
                continue
        
        # Create trigger word info file
        trigger_file = os.path.join(training_folder, "_training_info.txt")
        with open(trigger_file, "w") as f:
            f.write(f"Training Name: {training_name}\n")
            f.write(f"Trigger Word: {trigger_word}\n")
            f.write(f"Upload Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Images: {image_count}\n")
            f.write(f"Total Captions: {caption_count}\n")
        
        response_data = {
            "uploaded_files": uploaded_files,
            "training_folder": training_folder,
            "total_images": image_count,
            "total_captions": caption_count,
            "message": f"Successfully uploaded {len(uploaded_files)} files to {training_folder}"
        }
        
        log(f"✅ Training data uploaded: {training_folder} ({image_count} images, {caption_count} captions)", "INFO")
        
        return response_data
        
    except Exception as e:
        log(f"❌ Upload training data error: {e}", "ERROR")
        return {"error": f"Failed to upload training data: {str(e)}"}

async def handle_bulk_download(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle bulk download request"""
    try:
        process_ids = job_input.get("process_ids", [])
        include_images = job_input.get("include_images", True)
        include_loras = job_input.get("include_loras", True)
        
        if not process_ids:
            return {"error": "No process IDs provided"}
        
        if not _storage_service:
            return {"error": "Storage service not initialized"}
        
        download_items = []
        total_size = 0
        
        for process_id in process_ids:
            try:
                # List files for this process
                files = await _storage_service.list_files(f"results/{process_id}/")
                
                for file_info in files:
                    file_type = "other"
                    if any(ext in file_info.get('key', '').lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                        if include_images:
                            file_type = "image"
                        else:
                            continue
                    elif any(ext in file_info.get('key', '').lower() for ext in ['.safetensors', '.ckpt', '.pt']):
                        if include_loras:
                            file_type = "lora"
                        else:
                            continue
                    
                    # Generate download URL
                    download_url = await _storage_service.get_download_url(process_id)
                    if download_url:
                        download_items.append({
                            "filename": os.path.basename(file_info.get('key', '')),
                            "url": download_url,
                            "size": file_info.get('size', 0),
                            "type": file_type
                        })
                        total_size += file_info.get('size', 0)
            except Exception as e:
                log(f"⚠️ Failed to process files for process {process_id}: {e}", "WARN")
                continue
        
        response_data = {
            "download_items": download_items,
            "zip_url": None,  # TODO: Implement zip creation if needed
            "total_files": len(download_items),
            "total_size": total_size
        }
        
        return response_data
        
    except Exception as e:
        log(f"❌ Bulk download error: {e}", "ERROR")
        return {"error": f"Failed to create bulk download: {str(e)}"}

# Sync wrapper for RunPod
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for async handler"""
    try:
        # Run async handler in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_handler(event))
        loop.close()
        return result
    except Exception as e:
        log(f"💥 Handler wrapper error: {e}", "ERROR")
        return {"error": str(e)}

# Start RunPod Serverless
if __name__ == "__main__":
    log("🚀 Starting LoRA Dashboard RunPod Serverless Handler", "INFO")
    runpod.serverless.start({"handler": handler}) 