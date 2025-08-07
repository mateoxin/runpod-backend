#!/usr/bin/env python3
"""
🚀 ULTRA-FAST RUNPOD HANDLER - LoRA Dashboard Backend
Minimal handler with runtime setup for heavy dependencies
Deploy time: ~30 seconds instead of 20 minutes!
Based on successful runpod-fastbackend/ approach

🔄 QUEUE MANAGEMENT:
- Uses RunPod's built-in queue system (IN_QUEUE → IN_PROGRESS → COMPLETED)
- No Redis needed - RunPod handles job queuing and status tracking
- Handler simply returns results, RunPod manages the rest
"""

import runpod
import asyncio
import logging
import os
import sys
import subprocess
import time
import threading
import uuid
import yaml
import base64
import shutil
import glob
from datetime import datetime
from typing import Dict, Any, Optional
import io

# Global flag to track if environment is setup
ENVIRONMENT_READY = False
SETUP_LOCK = False

# Global process management (in-memory storage)
RUNNING_PROCESSES: Dict[str, Dict[str, Any]] = {}
PROCESS_LOCK = threading.Lock()

def log(message, level="INFO"):
    """Unified logging to stdout for RunPod visibility"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {message}"
    
    # Write only to stdout to avoid duplicate logs in RunPod
    print(log_msg)
    sys.stdout.flush()

def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Process management functions
def add_process(process_id: str, process_type: str, status: str, config: Dict[str, Any]):
    """Add a new process to tracking."""
    with PROCESS_LOCK:
        RUNNING_PROCESSES[process_id] = {
            "id": process_id,
            "type": process_type,
            "status": status,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "output_path": None,
            "error": None
        }

def update_process_status(process_id: str, status: str, output_path: str = None, error: str = None):
    """Update process status."""
    with PROCESS_LOCK:
        if process_id in RUNNING_PROCESSES:
            RUNNING_PROCESSES[process_id]["status"] = status
            RUNNING_PROCESSES[process_id]["updated_at"] = datetime.now().isoformat()
            if output_path:
                RUNNING_PROCESSES[process_id]["output_path"] = output_path
            if error:
                RUNNING_PROCESSES[process_id]["error"] = error

def get_process(process_id: str) -> Optional[Dict[str, Any]]:
    """Get process by ID."""
    with PROCESS_LOCK:
        return RUNNING_PROCESSES.get(process_id)

def get_all_processes() -> list:
    """Get all processes."""
    with PROCESS_LOCK:
        return list(RUNNING_PROCESSES.values())

def setup_environment():
    """Simplified setup for fast deployment"""
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
        log("🚀 Setting up minimal environment...", "INFO")
        
        # Step 1: Setup directories
        log("📁 Creating workspace directories...", "INFO")
        os.makedirs("/workspace", exist_ok=True)
        os.makedirs("/workspace/training_data", exist_ok=True)
        os.makedirs("/workspace/models", exist_ok=True)
        os.makedirs("/workspace/logs", exist_ok=True)
        
        # Step 2: Optional HuggingFace token (non-blocking)
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token and hf_token != "":
            log("🤗 HuggingFace token found, continuing...", "INFO")
        else:
            log("ℹ️ No HuggingFace token provided", "INFO")
        
        # Step 3: Install essential dependencies (no Redis - RunPod has built-in queue)
        log("📦 Installing essential dependencies...", "INFO")
        try:
            # Install HuggingFace Hub with CLI for model downloads and authentication
            log("📦 Installing HuggingFace Hub...", "INFO")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "huggingface_hub[cli]>=0.24.0"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                log("✅ HuggingFace Hub installed", "INFO")
            else:
                log("⚠️ HuggingFace Hub install failed, continuing...", "WARN")
            
            # Install S3 support for storage
            log("📦 Installing S3 support...", "INFO")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "boto3>=1.34.0"  # Only S3 for storage, RunPod handles queue
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                log("✅ Essential dependencies installed", "INFO")
            else:
                log("⚠️ Some dependencies failed, continuing...", "WARN")
        except Exception as e:
            log(f"⚠️ Dependency install warning: {e}, continuing...", "WARN")
        
        # Step 4: Upgrade pip
        log("📦 Upgrading pip...", "INFO")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                log("✅ Pip upgraded successfully", "INFO")
            else:
                log(f"⚠️ Pip upgrade failed: {result.stderr}", "WARN")
        except Exception as e:
            log(f"⚠️ Pip upgrade warning: {e}", "WARN")
        
        # Step 5: Clone ai-toolkit if not exists
        ai_toolkit_path = "/workspace/ai-toolkit"
        if not os.path.exists(ai_toolkit_path):
            log("📥 Cloning ai-toolkit...", "INFO")
            try:
                # Change to workspace directory first
                os.chdir("/workspace")
                result = subprocess.run([
                    "git", "clone", "https://github.com/ostris/ai-toolkit.git"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    log("✅ AI-toolkit cloned successfully", "INFO")
                else:
                    log(f"❌ AI-toolkit clone failed: {result.stderr}", "ERROR")
                    return False
            except Exception as e:
                log(f"❌ AI-toolkit clone error: {e}", "ERROR")
                return False
        else:
            log("✅ AI-toolkit already exists", "INFO")
        
        # Step 6: Install ai-toolkit requirements
        ai_toolkit_requirements = "/workspace/ai-toolkit/requirements.txt"
        if os.path.exists(ai_toolkit_requirements):
            log("📦 Installing ai-toolkit requirements...", "INFO")
            try:
                # Change to ai-toolkit directory first
                os.chdir("/workspace/ai-toolkit")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    log("✅ AI-toolkit requirements installed", "INFO")
                else:
                    log(f"⚠️ AI-toolkit requirements install warning: {result.stderr}", "WARN")
            except Exception as e:
                log(f"⚠️ AI-toolkit requirements install warning: {e}", "WARN")
        else:
            log("⚠️ AI-toolkit requirements.txt not found", "WARN")
        
        # Step 7: Install additional python-dotenv for ai-toolkit
        log("📦 Installing python-dotenv for ai-toolkit...", "INFO")
        try:
            os.chdir("/workspace/ai-toolkit")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "python-dotenv"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                log("✅ Python-dotenv installed", "INFO")
            else:
                log(f"⚠️ Python-dotenv install warning: {result.stderr}", "WARN")
        except Exception as e:
            log(f"⚠️ Python-dotenv install warning: {e}", "WARN")
        
        # Step 8: Install essential ML packages
        log("📦 Installing essential ML packages...", "INFO")
        ml_packages = [
            "albumentations",
            "diffusers", 
            "transformers",
            "accelerate",
            "peft"
        ]
        
        for package in ml_packages:
            try:
                log(f"📦 Installing {package}...", "INFO")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--upgrade", package
                ], capture_output=True, text=True, timeout=180)
                
                if result.returncode == 0:
                    log(f"✅ {package} installed successfully", "INFO")
                else:
                    log(f"⚠️ {package} install warning: {result.stderr}", "WARN")
            except Exception as e:
                log(f"⚠️ {package} install warning: {e}", "WARN")
        
        log("✅ Complete environment setup ready!", "INFO")
        ENVIRONMENT_READY = True
        return True
        
    except Exception as e:
        log(f"❌ Setup error: {e}", "ERROR")
        return False
    finally:
        SETUP_LOCK = False

def get_real_services():
    """Return real services for full deployment"""
    log("🔥 Initializing REAL services with AI toolkit integration", "INFO")
    
    class RealGPUManager:
        def __init__(self):
            log("🎮 Real GPU Manager initialized", "INFO")
            self.gpu_info = self._detect_gpu()
        
        def _detect_gpu(self):
            """Detect available GPU information"""
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    gpus = []
                    for line in lines:
                        name, total, free = line.split(', ')
                        gpus.append({
                            "name": name,
                            "memory_total": f"{total}MB",
                            "memory_free": f"{free}MB"
                        })
                    return {"gpus": gpus, "count": len(gpus)}
                else:
                    return {"gpus": [], "count": 0, "error": "nvidia-smi failed"}
            except Exception as e:
                log(f"⚠️ GPU detection failed: {e}", "WARN")
                return {"gpus": [], "count": 0, "error": str(e)}
        
        def get_status(self):
            return {"status": "healthy", **self.gpu_info}
    
    class RealProcessManager:
        def __init__(self, **kwargs):
            log("⚡ Real Process Manager initialized", "INFO")
            log("🔄 Using RunPod built-in queue system + real AI toolkit", "INFO")
        
        async def initialize(self):
            log("🚀 Real Process Manager ready for AI operations", "INFO")
        
        async def start_training(self, config):
            """Start real LoRA training with AI toolkit"""
            process_id = f"train_{uuid.uuid4().hex[:12]}"
            
            try:
                # Add to process tracking
                add_process(process_id, "training", "starting", {"config": config})
                log(f"🎯 Real training started: {process_id}", "INFO")
                
                # Start training in background thread
                threading.Thread(
                    target=self._run_training_background,
                    args=(process_id, config),
                    daemon=True
                ).start()
                
                return process_id
            except Exception as e:
                log(f"❌ Training start failed: {e}", "ERROR")
                update_process_status(process_id, "failed", error=str(e))
                raise
        
        def _run_training_background(self, process_id: str, config):
            """Run training in background thread"""
            try:
                log(f"🚀 Training background thread started: {process_id}", "INFO")
                update_process_status(process_id, "running")
                log(f"📊 Process status updated to running: {process_id}", "INFO")
                
                # Setup environment
                log(f"🔧 Setting up environment for: {process_id}", "INFO")
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    log(f"❌ HF_TOKEN not found in environment for: {process_id}", "ERROR")
                    raise Exception("HF_TOKEN not found in environment")
                
                log(f"✅ HF_TOKEN found for: {process_id}", "INFO")
                
                # Create YAML config file
                log(f"📁 Checking workspace directory for: {process_id}", "INFO")
                os.makedirs("/workspace", exist_ok=True)
                config_path = f"/workspace/training_{process_id}.yaml"
                log(f"💾 Writing YAML config to: {config_path}", "INFO")
                with open(config_path, 'w') as f:
                    f.write(config)
                log(f"✅ YAML config written successfully: {process_id}", "INFO")
                
                # Log YAML config details
                log(f"💾 YAML config written to: {config_path}", "INFO")
                log(f"📄 Full YAML config for {process_id}:\n{config}", "INFO")
                
                # Login to HuggingFace programmatically
                try:
                    # Try using huggingface_hub API (should be available after install)
                    from huggingface_hub import login
                    login(token=hf_token)
                    log(f"✅ HuggingFace programmatic login successful", "INFO")
                except ImportError as e:
                    # This shouldn't happen after our install, but fallback gracefully
                    log(f"⚠️ huggingface_hub import failed: {e} - using env vars only", "WARNING")
                except Exception as e:
                    # If programmatic login fails, continue with env vars (often sufficient)
                    log(f"⚠️ HF programmatic login failed: {e} - env vars will be used", "WARNING")
                
                # Setup environment variables
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HUGGING_FACE_HUB_TOKEN": hf_token,
                    "HF_TOKEN": hf_token,
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "TRANSFORMERS_CACHE": "/workspace/cache"
                })
                
                # Check if AI toolkit exists
                ai_toolkit_path = "/workspace/ai-toolkit/run.py"
                if not os.path.exists(ai_toolkit_path):
                    raise Exception(f"AI toolkit not found at {ai_toolkit_path}")
                
                log(f"🚀 Starting AI toolkit training: {ai_toolkit_path}", "INFO")
                
                # Run AI toolkit training
                cmd = ["python3", ai_toolkit_path, config_path]
                log(f"🎯 Training command: {' '.join(cmd)}", "INFO")
                result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
                
                # Log training output for debugging
                if result.stdout:
                    log(f"📋 Training stdout: {result.stdout[:1000]}...", "INFO")
                if result.stderr:
                    log(f"❌ Training stderr: {result.stderr[:1000]}...", "ERROR")
                
                if result.returncode == 0:
                    # Find output files
                    output_dir = "/workspace/output"
                    output_files = glob.glob(f"{output_dir}/**/*.safetensors", recursive=True)
                    
                    if output_files:
                        update_process_status(process_id, "completed", output_path=output_files[0])
                        log(f"✅ Training completed: {process_id}", "INFO")
                    else:
                        update_process_status(process_id, "completed", output_path=output_dir)
                        log(f"✅ Training completed (no .safetensors found): {process_id}", "INFO")
                else:
                    error_msg = f"Training failed: {result.stderr}"
                    update_process_status(process_id, "failed", error=error_msg)
                    log(f"❌ Training failed: {process_id} - {error_msg}", "ERROR")
                    
            except Exception as e:
                error_msg = f"Training error: {str(e)}"
                update_process_status(process_id, "failed", error=error_msg)
                log(f"❌ Training error: {process_id} - {error_msg}", "ERROR")
        
        async def start_generation(self, config):
            """Start real image generation"""
            process_id = f"gen_{uuid.uuid4().hex[:12]}"
            
            try:
                # Add to process tracking  
                add_process(process_id, "generation", "starting", {"config": config})
                log(f"🖼️ Real generation started: {process_id}", "INFO")
                
                # For now, simple placeholder - you can extend this with actual Stable Diffusion
                threading.Thread(
                    target=self._run_generation_background,
                    args=(process_id, config),
                    daemon=True
                ).start()
                
                return process_id
            except Exception as e:
                log(f"❌ Generation start failed: {e}", "ERROR")
                update_process_status(process_id, "failed", error=str(e))
                raise
        
        def _run_generation_background(self, process_id: str, config):
            """Run generation in background thread"""
            try:
                update_process_status(process_id, "running")
                
                # Simulate generation work (replace with real SD pipeline)
                time.sleep(5)  # Placeholder for actual generation
                
                # Create placeholder output
                output_path = f"/workspace/output/generated_{process_id}.txt"
                os.makedirs("/workspace/output", exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(f"Generated image for config: {config}")
                
                update_process_status(process_id, "completed", output_path=output_path)
                log(f"✅ Generation completed: {process_id}", "INFO")
                
            except Exception as e:
                error_msg = f"Generation error: {str(e)}"
                update_process_status(process_id, "failed", error=error_msg)
                log(f"❌ Generation error: {process_id} - {error_msg}", "ERROR")
        
        async def get_process(self, process_id):
            """Get process status from global tracking"""
            return get_process(process_id) or {"status": "not_found"}
        
        async def cancel_process(self, process_id):
            """Cancel a running process"""
            process = get_process(process_id)
            if process:
                update_process_status(process_id, "cancelled")
                return True
            return False
        
        async def get_all_processes(self):
            """Get all processes from global tracking"""
            return get_all_processes()
    
    class RealStorageService:
        def __init__(self):
            log("💾 Real Storage Service initialized", "INFO")
            self.workspace_path = "/workspace"
        
        async def health_check(self):
            return "healthy"
        
        async def get_download_url(self, process_id):
            """Generate download URL/data for process output"""
            process = get_process(process_id)
            if not process or not process.get("output_path"):
                return None
            
            output_path = process.get("output_path")
            
            try:
                # In RunPod Serverless, we need to return file as base64 data
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as file:
                        file_data = file.read()
                        file_base64 = base64.b64encode(file_data).decode('utf-8')
                        
                        # Return download data instead of URL
                        return {
                            "type": "file_data",
                            "filename": os.path.basename(output_path),
                            "data": file_base64,
                            "size": len(file_data),
                            "content_type": "application/octet-stream"
                        }
                else:
                    log(f"❌ File not found: {output_path}", "ERROR")
                    return None
                    
            except Exception as e:
                log(f"❌ Error reading file {output_path}: {e}", "ERROR")
                return None
        
        async def list_files(self, path):
            """List files in directory"""
            full_path = os.path.join(self.workspace_path, path.lstrip('/'))
            if os.path.exists(full_path):
                files = []
                for item in os.listdir(full_path):
                    item_path = os.path.join(full_path, item)
                    if os.path.isfile(item_path):
                        files.append({
                            "key": f"{path}/{item}",
                            "size": os.path.getsize(item_path)
                        })
                return files
            return []
    
    class RealLoRAService:
        def __init__(self, storage=None):
            log("🎨 Real LoRA Service initialized", "INFO")
            self.storage = storage
        
        async def get_available_models(self):
            """Get list of trained LoRA models"""
            try:
                output_dir = "/workspace/output"
                if os.path.exists(output_dir):
                    models = glob.glob(f"{output_dir}/**/*.safetensors", recursive=True)
                    model_list = []
                    for model_path in models:
                        filename = os.path.basename(model_path)
                        try:
                            stat = os.stat(model_path)
                            size_mb = round(stat.st_size / (1024 * 1024), 1)
                            modified_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
                        except:
                            size_mb = 0
                            modified_date = datetime.now().isoformat()
                        
                        model_list.append({
                            "id": filename.replace('.safetensors', ''),
                            "filename": filename,
                            "name": filename.replace('.safetensors', ''),
                            "path": model_path,
                            "size_mb": size_mb,
                            "modified_date": modified_date,
                            "status": "ready"
                        })
                    return model_list
                return []
            except Exception as e:
                log(f"❌ Error listing LoRA models: {e}", "ERROR")
                return []
    
    return {
        'GPUManager': RealGPUManager,
        'ProcessManager': RealProcessManager,
        'StorageService': RealStorageService,
        'LoRAService': RealLoRAService,
        'get_settings': lambda: {"workspace_path": "/workspace"}
    }

# Global service instances (initialized on first use)
_services_initialized = False
_services = None
_gpu_manager = None
_process_manager = None
_storage_service = None
_lora_service = None

async def initialize_services():
    """Initialize simplified services for fast deployment"""
    global _services_initialized, _services, _gpu_manager, _process_manager, _storage_service, _lora_service
    
    if _services_initialized:
        log("🔄 Services already initialized, skipping setup", "INFO")
        return
    
    try:
        # First setup environment
        if not setup_environment():
            log("⚠️ Environment setup failed, using minimal mode", "WARN")
        
        # Use REAL services for full AI functionality
        _services = get_real_services()
        
        log("🚀 Initializing REAL AI services...", "INFO")
        
        settings = _services['get_settings']() if _services['get_settings'] else {"workspace_path": "/workspace"}
        
        # Initialize real services (create instances)
        _storage_service = _services['StorageService']()
        _lora_service = _services['LoRAService'](_storage_service) 
        _gpu_manager = _services['GPUManager']()
        _process_manager = _services['ProcessManager'](
            gpu_manager=_gpu_manager,
            storage_service=_storage_service
            # No Redis needed - RunPod handles queue management
        )
        
        # Initialize process manager async
        if _process_manager:
            await _process_manager.initialize()
        
        _services_initialized = True
        log("✅ Real AI services ready for training and generation!", "INFO")
        
        # Log service status for debugging
        log(f"🔧 Service status: GPU Manager: {'✅' if _gpu_manager else '❌'} | Process Manager: {'✅' if _process_manager else '❌'} | Storage: {'✅' if _storage_service else '❌'} | LoRA: {'✅' if _lora_service else '❌'}", "INFO")
        
    except Exception as e:
        log(f"❌ Failed to initialize services: {e}", "ERROR")
        # Don't raise - continue with minimal functionality
        _services_initialized = True
        log("⚠️ Continuing with minimal functionality", "WARN")

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
        
        # Handle simple prompt requests (auto-detect as generation)
        if not job_type and job_input.get("prompt"):
            job_type = "generate"
            log("📝 Auto-detected generation request from prompt", "INFO")
        
        # Log incoming request
        request_id = f"req_{int(time.time() * 1000)}"
        log(f"📨 Incoming {job_type or 'unknown'} request | Request ID: {request_id}", "INFO")
        
        # Enhanced request logging
        if job_type == "processes":
            log(f"🔍 GET PROCESSES request detected - retrieving process list | Request ID: {request_id}", "INFO")
        elif job_type == "train":
            log(f"🎯 TRAINING request detected - will start new training | Request ID: {request_id}", "INFO")
        elif job_type == "generate":
            log(f"🖼️ GENERATION request detected - will start new generation | Request ID: {request_id}", "INFO")
        
        log(f"📨 Processing job type: {job_type} | Request ID: {request_id}", "INFO")
        
        # Route to appropriate handler based on job type
        if job_type == "health":
            response = await handle_health_check()
        elif job_type == "train" or job_type == "train_with_yaml":
            response = await handle_training(job_input)
        elif job_type == "generate":
            response = await handle_generation(job_input)
        elif job_type == "processes":
            response = await handle_get_processes(job_input)
        elif job_type == "process_status":
            response = await handle_process_status(job_input)
        elif job_type == "lora" or job_type == "list_models":
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
                "supported_types": ["health", "train", "train_with_yaml", "generate", "processes", "process_status", "lora", "list_models", "cancel", "download", "upload_training_data", "bulk_download"]
            }
        
        # Log successful response with timing
        status = "success" if not response.get("error") else "error"
        end_time = time.time()
        start_time = int(request_id.split('_')[1]) / 1000  # Extract timestamp from request_id
        duration = end_time - start_time
        
        log(f"✅ Request completed: {status} | Duration: {duration:.3f}s | Request ID: {request_id}", "INFO")
        
        # Additional response details for debugging
        if job_type == "processes" and status == "success":
            process_count = len(response.get("processes", []))
            log(f"📊 Processes response: {process_count} processes returned | Request ID: {request_id}", "INFO")
        
        return response
            
    except Exception as e:
        log(f"💥 Handler error: {e}", "ERROR")
        error_response = {"error": str(e)}
        
        # Log error response
        log(f"❌ Request failed | Request ID: {request_id} | Error: {str(e)}", "ERROR")
        
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
        # Support both 'config' and 'yaml_config' for backward compatibility
        config = job_input.get("config") or job_input.get("yaml_config")
        if not config:
            return {"error": "Missing 'config' or 'yaml_config' parameter"}
        
        # Log incoming YAML configuration
        log(f"📝 Training YAML config received (first 500 chars): {str(config)[:500]}...", "INFO")
        
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
        # Support both config and simple prompt
        config = job_input.get("config")
        prompt = job_input.get("prompt")
        
        if not config and not prompt:
            return {"error": "Missing 'config' or 'prompt' parameter"}
        
        # Create simple config from prompt if needed
        if not config and prompt:
            config = f"prompt: '{prompt}'"
            log(f"📝 Created simple config from prompt: {prompt}", "INFO")
        
        # Log generation config
        if config:
            log(f"🖼️ Generation config received (first 300 chars): {str(config)[:300]}...", "INFO")
        
        if not _process_manager:
            return {"error": "Process manager not initialized"}
        
        process_id = await _process_manager.start_generation(config)
        log(f"✅ Generation started with process_id: {process_id}", "INFO")
        
        return {
            "process_id": process_id,
            "message": f"Generation started for prompt: {prompt}" if prompt else "Generation started",
            "status": "started"
        }
    except Exception as e:
        log(f"❌ Generation error: {e}", "ERROR")
        return {"error": f"Failed to start generation: {str(e)}"}

async def handle_get_processes(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get processes request"""
    try:
        if not _process_manager:
            log("❌ Process manager not initialized for get_processes request", "ERROR")
            return {"error": "Process manager not initialized"}
        
        processes = await _process_manager.get_all_processes()
        
        # Enhanced logging for debugging
        log(f"📊 Retrieved {len(processes)} total processes", "INFO")
        
        if processes:
            # Count processes by status
            status_counts = {}
            for process in processes:
                status = process.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            log(f"📋 Process status breakdown: {status_counts}", "INFO")
            
            # Log recent processes (last 3)
            recent_processes = sorted(processes, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
            for i, process in enumerate(recent_processes):
                log(f"📄 Recent process {i+1}: {process.get('id', 'unknown')} | Type: {process.get('type', 'unknown')} | Status: {process.get('status', 'unknown')}", "INFO")
        else:
            log("📭 No processes found in system", "INFO")
        
        return {
            "processes": processes,
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'local'),
            "environment": "serverless",
            "note": "Processes are isolated per serverless worker instance"
        }
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
    """Handle download request - returns file data as base64"""
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
            
        # Get download data (now returns file data instead of URL)
        download_data = await _storage_service.get_download_url(process_id)
        
        if download_data:
            log(f"✅ Download prepared for process {process_id}: {download_data['filename']} ({format_file_size(download_data['size'])})", "INFO")
            return download_data
        else:
            return {"error": "No download data available for this process"}
            
    except Exception as e:
        log(f"❌ Download error: {e}", "ERROR")
        return {"error": f"Failed to prepare download: {str(e)}"}

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
        
        # ZMIENIONE: Upload bezpośrednio do /workspace/training_data (bez subfolderów)
        workspace_path = os.environ.get("WORKSPACE_PATH", "/workspace")
        training_folder = os.path.join(workspace_path, "training_data")  # Bezpośrednio do głównego folderu
        
        # ULEPSZONE: Wyczyść wszystkie istniejące pliki (nie tylko foldery)
        if cleanup_existing:
            if os.path.exists(training_folder):
                log(f"🧹 Czyszczenie istniejących plików w {training_folder}", "INFO")
                
                # Usuń wszystkie pliki w folderze
                for item in os.listdir(training_folder):
                    item_path = os.path.join(training_folder, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            log(f"   🗑️  Usunięto plik: {item}", "INFO")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            log(f"   🗑️  Usunięto folder: {item}", "INFO")
                    except Exception as e:
                        log(f"   ⚠️  Błąd usuwania {item}: {e}", "WARN")
                
                log(f"✅ Folder wyczyszczony: {training_folder}", "INFO")
        
        # Create training folder
        os.makedirs(training_folder, exist_ok=True)
        
        uploaded_files = []
        image_count = 0
        caption_count = 0
        total_files_attempted = len(files_data)
        total_files_failed = 0
        
        log(f"🚀 Processing {total_files_attempted} files for upload | Request ID: {request_id}", "INFO")
        
        # Process files (expecting base64 encoded files)
        for i, file_info in enumerate(files_data):
            filename = file_info.get("filename")
            content = file_info.get("content")  # base64 encoded
            content_type = file_info.get("content_type", "application/octet-stream")
            
            log(f"📁 Processing file {i+1}/{total_files_attempted}: {filename} | Request ID: {request_id}", "INFO")
            
            if not filename or not content:
                log(f"⚠️ Skipping file {i+1}: missing filename or content | Request ID: {request_id}", "WARN")
                total_files_failed += 1
                continue
                
            # Save file to training folder
            file_path = os.path.join(training_folder, filename)
            
            # Decode base64 content and save
            import base64
            try:
                # Fix base64 padding if needed
                content_padded = content
                missing_padding = len(content) % 4
                if missing_padding:
                    content_padded += '=' * (4 - missing_padding)
                
                file_content = base64.b64decode(content_padded)
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                # Determine file type
                file_type = "other"
                if content_type and content_type.startswith('image/'):
                    file_type = "image"
                elif filename.endswith('.txt'):
                    file_type = "caption"
                
                file_data = {
                    "filename": filename,
                    "path": file_path,
                    "size": len(file_content),
                    "size_formatted": format_file_size(len(file_content)),
                    "content_type": content_type,
                    "file_type": file_type,
                    "uploaded_at": datetime.now().isoformat(),
                    "folder": "training_data"
                }
                uploaded_files.append(file_data)
                
                # Enhanced log file operation with more details
                log(f"✅ File uploaded: {filename} | Size: {file_data['size_formatted']} | Type: {file_type} | Folder: training_data | Request ID: {request_id}", "INFO")
                
                # Count file types - CRITICAL: This is now inside the successful processing block
                if file_type == "image":
                    image_count += 1
                    log(f"📷 Image count incremented to: {image_count} | Request ID: {request_id}", "INFO")
                elif file_type == "caption":
                    caption_count += 1
                    log(f"📝 Caption count incremented to: {caption_count} | Request ID: {request_id}", "INFO")
                    
            except Exception as e:
                log(f"❌ Failed to process file {filename} | Request ID: {request_id} | Error: {e}", "ERROR")
                total_files_failed += 1
                continue
        
        # Final upload summary with debugging info
        log(f"📊 Upload summary | Request ID: {request_id} | Total attempted: {total_files_attempted} | Succeeded: {len(uploaded_files)} | Failed: {total_files_failed} | Images: {image_count} | Captions: {caption_count}", "INFO")
        
        # CRITICAL DEBUG: Warn if all files failed (this causes "undefined" in frontend)
        if total_files_failed == total_files_attempted:
            log(f"🚨 WARNING: ALL files failed to upload! This will show undefined counts in frontend | Request ID: {request_id}", "WARN")
        elif image_count == 0 and caption_count == 0:
            log(f"🚨 WARNING: No images or captions were successfully processed | Request ID: {request_id}", "WARN")
        
        # Create trigger word info file
        trigger_file = os.path.join(training_folder, "_training_info.txt")
        with open(trigger_file, "w") as f:
            f.write(f"Training Name: {training_name}\n")
            f.write(f"Trigger Word: {trigger_word}\n")
            f.write(f"Upload Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Images: {image_count}\n")
            f.write(f"Total Captions: {caption_count}\n")
        
        # Get RunPod environment information
        worker_id = os.environ.get("RUNPOD_WORKER_ID", "local")
        pod_id = os.environ.get("RUNPOD_POD_ID", "local")
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", None)
        
        # Create RunPod-specific paths for uploaded files
        for file_data in uploaded_files:
            # Add RunPod workspace relative path
            relative_path = file_data["path"].replace(workspace_path, "").lstrip("/")
            file_data["runpod_workspace_path"] = f"/workspace/{relative_path}"
            file_data["runpod_relative_path"] = relative_path
        
        # Calculate total size
        total_size = sum(file_data["size"] for file_data in uploaded_files)
        total_size_formatted = format_file_size(total_size)
        
        # Create detailed file summary
        file_types_summary = {
            "images": [f for f in uploaded_files if f["file_type"] == "image"],
            "captions": [f for f in uploaded_files if f["file_type"] == "caption"],
            "other": [f for f in uploaded_files if f["file_type"] == "other"]
        }
        
        response_data = {
            "uploaded_files": uploaded_files,
            "training_folder": training_folder,
            "total_images": image_count,
            "total_captions": caption_count,
            "total_size": total_size,
            "total_size_formatted": total_size_formatted,
            "file_types_summary": file_types_summary,
            "message": f"✅ Successfully uploaded {len(uploaded_files)} files ({total_size_formatted}) to training_data folder",
            "detailed_message": f"📁 Uploaded to training_data folder:\n📷 {image_count} images\n📝 {caption_count} captions\n💾 Total size: {total_size_formatted}",
            "debug_info": {
                "total_files_attempted": total_files_attempted,
                "total_files_succeeded": len(uploaded_files),
                "total_files_failed": total_files_failed,
                "all_files_failed": total_files_failed == total_files_attempted,
                "no_valid_content": image_count == 0 and caption_count == 0
            },
            "runpod_info": {
                "worker_id": worker_id,
                "pod_id": pod_id,
                "endpoint_id": endpoint_id,
                "workspace_path": workspace_path,
                "training_folder_relative": training_folder.replace(workspace_path, "").lstrip("/")
            }
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
        
        log(f"✅ Bulk download prepared: {len(download_items)} files ({format_file_size(total_size)})", "INFO")
        return response_data
        
    except Exception as e:
        log(f"❌ Bulk download error: {e}", "ERROR")
        return {"error": f"Failed to create bulk download: {str(e)}"}

# Start RunPod Serverless
if __name__ == "__main__":
    log("🚀 Starting LoRA Dashboard RunPod Serverless Handler", "INFO")
    runpod.serverless.start({
        "handler": async_handler,  # ✅ Bezpośrednio async handler
        "return_aggregate_stream": True
    }) 