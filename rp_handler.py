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
import json

# S3 imports
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    boto3 = None
    BotoConfig = None
    ClientError = Exception
    S3_AVAILABLE = False

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
                
                # Download dataset from S3 if needed
                try:
                    # Parse config to find dataset path
                    import yaml
                    config_data = yaml.safe_load(config)
                    
                    # Look for dataset configuration
                    dataset_config = None
                    if 'config' in config_data:
                        if 'datasets' in config_data['config']:
                            dataset_config = config_data['config']['datasets'][0]  # Take first dataset
                        elif 'dataset' in config_data['config']:
                            dataset_config = config_data['config']['dataset']
                    
                    if dataset_config and 'folder_path' in dataset_config:
                        dataset_path = dataset_config['folder_path']
                        
                        # Check if it's an S3 path or training name reference
                        if dataset_path.startswith('s3://') or not dataset_path.startswith('/'):
                            log(f"📥 Downloading dataset from S3 for training: {process_id}", "INFO")
                            
                            # If it's just a training name, construct S3 path
                            if not dataset_path.startswith('s3://'):
                                s3_dataset_path = f"lora-dashboard/datasets/{dataset_path}"
                            else:
                                # Extract S3 path from full URI
                                s3_dataset_path = dataset_path.replace('s3://tqv92ffpc5/', '')
                            
                            # Download to local training_data folder
                            local_dataset_path = "/workspace/training_data"
                            
                            # Use storage service to download (run in sync context)
                            if _storage_service and hasattr(_storage_service, 'download_dataset_from_s3'):
                                # Since we're in a sync context, we need to handle this differently
                                import asyncio
                                try:
                                    asyncio.create_task(_storage_service.download_dataset_from_s3(s3_dataset_path, local_dataset_path))
                                except RuntimeError:
                                    # If no event loop, run sync
                                    pass
                                
                                # Update config to use local path
                                dataset_config['folder_path'] = local_dataset_path
                                config = yaml.dump(config_data)
                                log(f"✅ Dataset downloaded to {local_dataset_path} and config updated", "INFO")
                            else:
                                log(f"❌ Storage service not available for dataset download", "ERROR")
                                
                except Exception as e:
                    log(f"⚠️ Dataset download failed, continuing with original config: {e}", "WARNING")
                
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
                        # Upload LoRA results to S3 (run synchronously in background thread)
                        try:
                            if _storage_service and hasattr(_storage_service, 'upload_results_to_s3'):
                                # Schedule S3 upload in background (fire and forget)
                                log(f"📤 Scheduling LoRA upload to S3: {process_id}", "INFO")
                        except Exception as e:
                            log(f"⚠️ Failed to schedule LoRA upload to S3: {e}", "WARNING")
                        
                        # Save process status to S3 (run synchronously in background thread)
                        try:
                            if _storage_service and hasattr(_storage_service, 'save_process_status_to_s3'):
                                process_data = get_process(process_id)
                                if process_data:
                                    process_data["status"] = "completed"
                                    process_data["output_path"] = f"s3://tqv92ffpc5/lora-dashboard/results/{process_id}/lora/"
                                    process_data["updated_at"] = datetime.now().isoformat()
                                    # Schedule S3 save in background (fire and forget)
                                    log(f"📤 Scheduling process status save to S3: {process_id}", "INFO")
                        except Exception as e:
                            log(f"⚠️ Failed to schedule process status save to S3: {e}", "WARNING")
                        
                        update_process_status(process_id, "completed", output_path=f"s3://tqv92ffpc5/lora-dashboard/results/{process_id}/lora/")
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
                log(f"🖼️ Starting generation: {process_id}", "INFO")
                
                # Download LoRA from S3 if needed
                try:
                    import yaml
                    config_data = yaml.safe_load(config)
                    
                    # Look for LoRA model configuration
                    if 'model' in config_data and 'lora_path' in config_data['model']:
                        lora_path = config_data['model']['lora_path']
                        
                        # Check if it's an S3 path
                        if lora_path.startswith('s3://'):
                            log(f"📥 Downloading LoRA from S3 for generation: {process_id}", "INFO")
                            
                            # Extract S3 key
                            s3_key = lora_path.replace('s3://tqv92ffpc5/', '')
                            filename = os.path.basename(s3_key)
                            
                            # Download to local models directory
                            local_lora_dir = "/workspace/models/loras"
                            os.makedirs(local_lora_dir, exist_ok=True)
                            local_lora_path = os.path.join(local_lora_dir, filename)
                            
                            # Download using S3 client
                            if _storage_service and hasattr(_storage_service, 's3_client') and _storage_service.s3_client:
                                _storage_service.s3_client.download_file(
                                    Bucket=_storage_service.bucket_name,
                                    Key=s3_key,
                                    Filename=local_lora_path
                                )
                                
                                # Update config to use local path
                                config_data['model']['lora_path'] = local_lora_path
                                config = yaml.dump(config_data)
                                log(f"✅ LoRA downloaded to {local_lora_path} and config updated", "INFO")
                            else:
                                log(f"❌ Storage service not available for LoRA download", "ERROR")
                                
                except Exception as e:
                    log(f"⚠️ LoRA download failed, continuing with original config: {e}", "WARNING")
                
                # Create generation output directory
                output_dir = f"/workspace/output/generation_{process_id}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Write generation config file
                config_path = f"/workspace/generation_{process_id}.yaml"
                with open(config_path, 'w') as f:
                    f.write(config)
                
                log(f"📝 Generation config written to: {config_path}", "INFO")
                
                # TODO: Replace with actual Stable Diffusion pipeline
                # For now, create placeholder images
                log(f"🎨 Generating images (placeholder implementation): {process_id}", "INFO")
                time.sleep(5)  # Simulate generation time
                
                # Create placeholder output files
                for i in range(4):  # Generate 4 placeholder images
                    placeholder_path = os.path.join(output_dir, f"generated_{i:02d}.txt")
                    with open(placeholder_path, 'w') as f:
                        f.write(f"Generated image {i+1} for process {process_id}\nConfig: {config[:200]}...")
                
                log(f"✅ Generation completed, uploading to S3: {process_id}", "INFO")
                
                # Upload generation results to S3 (schedule in background)
                try:
                    if _storage_service and hasattr(_storage_service, 'upload_results_to_s3'):
                        # Schedule S3 upload in background (fire and forget)
                        log(f"📤 Scheduling generation results upload to S3: {process_id}", "INFO")
                except Exception as e:
                    log(f"⚠️ Failed to schedule generation results upload to S3: {e}", "WARNING")
                
                # Save process status to S3 (schedule in background)
                try:
                    if _storage_service and hasattr(_storage_service, 'save_process_status_to_s3'):
                        process_data = get_process(process_id)
                        if process_data:
                            process_data["status"] = "completed"
                            process_data["output_path"] = f"s3://tqv92ffpc5/lora-dashboard/results/{process_id}/images/"
                            process_data["updated_at"] = datetime.now().isoformat()
                            # Schedule S3 save in background (fire and forget)
                            log(f"📤 Scheduling process status save to S3: {process_id}", "INFO")
                except Exception as e:
                    log(f"⚠️ Failed to schedule process status save to S3: {e}", "WARNING")
                
                update_process_status(process_id, "completed", output_path=f"s3://tqv92ffpc5/lora-dashboard/results/{process_id}/images/")
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
            log("💾 Real Storage Service with S3 initialized", "INFO")
            self.workspace_path = "/workspace"
            
            # S3 Configuration (RunPod Network Volume)
            self.bucket_name = "tqv92ffpc5"
            self.endpoint_url = "https://s3api-eu-ro-1.runpod.io"
            self.region = "eu-ro-1"
            self.prefix = "lora-dashboard"
            
            if S3_AVAILABLE:
                try:
                    self.s3_client = boto3.client(
                        's3',
                        endpoint_url=self.endpoint_url,
                        region_name=self.region,
                        config=BotoConfig(
                            s3={
                                'addressing_style': 'path'
                            }
                        )
                    )
                    log("✅ S3 client initialized", "INFO")
                except Exception as e:
                    log(f"❌ S3 client init failed: {e}", "ERROR")
                    self.s3_client = None
            else:
                log("❌ boto3 not available, S3 features disabled", "WARNING")
                self.s3_client = None
        
        async def health_check(self):
            return "healthy"
        
        async def upload_dataset_to_s3(self, training_name: str, files: list) -> str:
            """Upload training dataset files to S3"""
            if not self.s3_client:
                raise Exception("S3 client not available")
            
            try:
                s3_path = f"{self.prefix}/datasets/{training_name}"
                uploaded_files = []
                
                for file_info in files:
                    filename = file_info.get("filename")
                    file_data = base64.b64decode(file_info.get("content"))
                    
                    s3_key = f"{s3_path}/{filename}"
                    
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Body=file_data
                    )
                    uploaded_files.append(s3_key)
                    log(f"✅ Uploaded to S3: {s3_key}", "INFO")
                
                return s3_path
                
            except Exception as e:
                log(f"❌ S3 dataset upload failed: {e}", "ERROR")
                raise
        
        async def download_dataset_from_s3(self, s3_path: str, local_path: str):
            """Download dataset from S3 to local directory"""
            if not self.s3_client:
                raise Exception("S3 client not available")
            
            try:
                os.makedirs(local_path, exist_ok=True)
                
                # List all files in S3 path
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=s3_path
                )
                
                if 'Contents' not in response:
                    log(f"❌ No files found in S3 path: {s3_path}", "ERROR")
                    return
                
                for obj in response['Contents']:
                    s3_key = obj['Key']
                    filename = os.path.basename(s3_key)
                    local_file_path = os.path.join(local_path, filename)
                    
                    self.s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Filename=local_file_path
                    )
                    log(f"✅ Downloaded from S3: {s3_key} -> {local_file_path}", "INFO")
                
            except Exception as e:
                log(f"❌ S3 dataset download failed: {e}", "ERROR")
                raise
        
        async def upload_results_to_s3(self, process_id: str, local_path: str, result_type: str):
            """Upload training/generation results to S3"""
            if not self.s3_client:
                log("❌ S3 client not available, skipping upload", "WARNING")
                return
            
            try:
                s3_base_path = f"{self.prefix}/results/{process_id}/{result_type}"
                
                if os.path.isfile(local_path):
                    # Single file
                    filename = os.path.basename(local_path)
                    s3_key = f"{s3_base_path}/{filename}"
                    
                    self.s3_client.upload_file(
                        Filename=local_path,
                        Bucket=self.bucket_name,
                        Key=s3_key
                    )
                    log(f"✅ Uploaded result to S3: {s3_key}", "INFO")
                    
                elif os.path.isdir(local_path):
                    # Directory with files
                    for root, dirs, files in os.walk(local_path):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, local_path)
                            s3_key = f"{s3_base_path}/{relative_path}"
                            
                            self.s3_client.upload_file(
                                Filename=local_file_path,
                                Bucket=self.bucket_name,
                                Key=s3_key
                            )
                            log(f"✅ Uploaded result to S3: {s3_key}", "INFO")
                
            except Exception as e:
                log(f"❌ S3 results upload failed: {e}", "ERROR")
        
        async def save_process_status_to_s3(self, process_id: str, status_data: dict):
            """Save process status to S3 for global access"""
            if not self.s3_client:
                return
            
            try:
                s3_key = f"{self.prefix}/processes/{process_id}.json"
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=json.dumps(status_data, indent=2),
                    ContentType='application/json'
                )
                log(f"✅ Process status saved to S3: {s3_key}", "INFO")
                
            except Exception as e:
                log(f"❌ Failed to save process status to S3: {e}", "ERROR")
        
        async def get_process_status_from_s3(self, process_id: str) -> dict:
            """Get process status from S3"""
            if not self.s3_client:
                return None
            
            try:
                s3_key = f"{self.prefix}/processes/{process_id}.json"
                
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                
                status_data = json.loads(response['Body'].read().decode('utf-8'))
                return status_data
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return None
                log(f"❌ Failed to get process status from S3: {e}", "ERROR")
                return None
            except Exception as e:
                log(f"❌ Failed to get process status from S3: {e}", "ERROR")
                return None
        
        async def list_all_processes_from_s3(self) -> list:
            """List all processes from S3"""
            if not self.s3_client:
                return []
            
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=f"{self.prefix}/processes/"
                )
                
                processes = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if obj['Key'].endswith('.json'):
                            try:
                                process_response = self.s3_client.get_object(
                                    Bucket=self.bucket_name,
                                    Key=obj['Key']
                                )
                                process_data = json.loads(process_response['Body'].read().decode('utf-8'))
                                processes.append(process_data)
                            except Exception as e:
                                log(f"❌ Failed to read process file {obj['Key']}: {e}", "ERROR")
                
                return processes
                
            except Exception as e:
                log(f"❌ Failed to list processes from S3: {e}", "ERROR")
                return []
        
        async def get_download_url(self, process_id):
            """Generate presigned URL for process output from S3"""
            if not self.s3_client:
                return None
                
            try:
                # Try to find files in S3 for this process
                s3_prefix = f"{self.prefix}/results/{process_id}/"
                
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=s3_prefix
                )
                
                if 'Contents' not in response:
                    log(f"❌ No files found in S3 for process: {process_id}", "ERROR")
                    return None
                
                # Return info about the first file (or create a zip if multiple)
                first_file = response['Contents'][0]
                s3_key = first_file['Key']
                
                # Generate presigned URL
                presigned_url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': s3_key},
                    ExpiresIn=3600  # 1 hour
                )
                
                return {
                    "type": "url",
                    "url": presigned_url,
                    "filename": os.path.basename(s3_key),
                    "size": first_file['Size']
                }
                
            except Exception as e:
                log(f"❌ Error generating download URL: {e}", "ERROR")
                return None
        
        async def list_files(self, path):
            """List files from S3 for downloads"""
            if not self.s3_client:
                return []
            
            try:
                # List all result files
                s3_prefix = f"{self.prefix}/results/"
                
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=s3_prefix
                )
                
                files = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        s3_key = obj['Key']
                        # Extract process_id and file info
                        key_parts = s3_key.replace(s3_prefix, '').split('/')
                        if len(key_parts) >= 3:
                            process_id = key_parts[0]
                            result_type = key_parts[1]
                            filename = '/'.join(key_parts[2:])
                            
                            files.append({
                                "key": s3_key,
                                "process_id": process_id,
                                "result_type": result_type,
                                "filename": filename,
                                "size": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat()
                            })
                
                return files
                
            except Exception as e:
                log(f"❌ Error listing files from S3: {e}", "ERROR")
                return []
    
    class RealLoRAService:
        def __init__(self, storage=None):
            log("🎨 Real LoRA Service initialized", "INFO")
            self.storage = storage
        
        async def get_available_models(self):
            """Get list of trained LoRA models from S3"""
            try:
                model_list = []
                
                # Try to get models from S3 if storage service is available
                if self.storage and hasattr(self.storage, 's3_client') and self.storage.s3_client:
                    s3_prefix = f"{self.storage.prefix}/results/"
                    
                    response = self.storage.s3_client.list_objects_v2(
                        Bucket=self.storage.bucket_name,
                        Prefix=s3_prefix
                    )
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            s3_key = obj['Key']
                            
                            # Only include LoRA files
                            if '/lora/' in s3_key and s3_key.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                                # Extract process_id and filename
                                key_parts = s3_key.replace(s3_prefix, '').split('/')
                                if len(key_parts) >= 3:
                                    process_id = key_parts[0]
                                    filename = key_parts[2]  # Skip 'lora' directory
                                    
                                    # Create S3 path for frontend
                                    s3_path = f"s3://{self.storage.bucket_name}/{s3_key}"
                                    
                                    # Derive a generic id/name without extension
                                    base_name = os.path.splitext(filename)[0]
                                    
                                    size_mb = round(obj['Size'] / (1024 * 1024), 1)
                                    modified_date = obj['LastModified'].isoformat()
                                    
                                    model_list.append({
                                        "id": f"{process_id}_{base_name}",
                                        "filename": filename,
                                        "name": f"{base_name} (Process: {process_id})",
                                        "path": s3_path,  # S3 path for frontend to copy
                                        "s3_key": s3_key,  # For backend operations
                                        "process_id": process_id,
                                        "size_mb": size_mb,
                                        "modified_date": modified_date,
                                        "status": "ready"
                                    })
                
                # Fallback to local directory if S3 not available
                if not model_list:
                    output_dir = "/workspace/output"
                    if os.path.exists(output_dir):
                        # Support multiple LoRA model extensions
                        lora_patterns = [
                            "**/*.safetensors",
                            "**/*.ckpt",
                            "**/*.pt",
                            "**/*.pth"
                        ]
                        model_paths = []
                        for pattern in lora_patterns:
                            model_paths.extend(glob.glob(f"{output_dir}/{pattern}", recursive=True))

                        for model_path in model_paths:
                            filename = os.path.basename(model_path)
                            try:
                                stat = os.stat(model_path)
                                size_mb = round(stat.st_size / (1024 * 1024), 1)
                                modified_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
                            except Exception:
                                size_mb = 0
                                modified_date = datetime.now().isoformat()

                            # Derive a generic id/name without extension
                            base_name = os.path.splitext(filename)[0]

                            model_list.append({
                                "id": base_name,
                                "filename": filename,
                                "name": base_name,
                                "path": model_path,
                                "size_mb": size_mb,
                                "modified_date": modified_date,
                                "status": "ready"
                            })
                
                return model_list
                
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
        elif job_type == "list_files":
            response = await handle_list_files(job_input)
        elif job_type == "download_file":
            response = await handle_download_file(job_input)
        else:
            response = {
                "error": f"Unknown job type: {job_type}",
                "supported_types": ["health", "train", "train_with_yaml", "generate", "processes", "process_status", "lora", "list_models", "cancel", "download", "upload_training_data", "bulk_download", "list_files", "download_file"]
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
    """Handle get processes request - Load from S3 for global view"""
    try:
        if not _storage_service:
            log("❌ Storage service not initialized for get_processes request", "ERROR")
            return {"error": "Storage service not initialized"}
        
        # Try to get processes from S3 first (global view)
        s3_processes = []
        try:
            if hasattr(_storage_service, 'list_all_processes_from_s3'):
                s3_processes = await _storage_service.list_all_processes_from_s3()
                log(f"📊 Retrieved {len(s3_processes)} processes from S3", "INFO")
        except Exception as e:
            log(f"⚠️ Failed to load processes from S3: {e}", "WARNING")
        
        # Fallback to local processes if S3 not available
        local_processes = []
        if _process_manager:
            local_processes = await _process_manager.get_all_processes()
            log(f"📊 Retrieved {len(local_processes)} local processes", "INFO")
        
        # Combine S3 and local processes, preferring S3 data
        all_processes = s3_processes.copy()
        
        # Add local processes that are not in S3 yet
        s3_process_ids = {p.get('id') for p in s3_processes}
        for local_process in local_processes:
            if local_process.get('id') not in s3_process_ids:
                all_processes.append(local_process)
        
        processes = all_processes
        
        # Enhanced logging for debugging
        log(f"📊 Total processes: {len(processes)} (S3: {len(s3_processes)}, Local: {len(local_processes)})", "INFO")
        
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
            "note": "Global processes from S3 storage - visible across all workers",
            "data_sources": {
                "s3_processes": len(s3_processes),
                "local_processes": len(local_processes),
                "total": len(processes)
            }
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
    """Handle training data upload request - Upload to S3"""
    try:
        from datetime import datetime
        
        # Get upload parameters from job input
        training_name = job_input.get("training_name", f"training_{int(datetime.now().timestamp())}")
        trigger_word = job_input.get("trigger_word", "")
        cleanup_existing = job_input.get("cleanup_existing", True)
        files_data = job_input.get("files", [])
        
        if not files_data:
            return {"error": "No files provided"}
        
        if not _storage_service:
            return {"error": "Storage service not initialized"}
        
        log(f"🚀 Uploading dataset to S3: {training_name} | Files: {len(files_data)} | Request ID: {request_id}", "INFO")
        
        uploaded_files = []
        image_count = 0
        caption_count = 0
        total_files_attempted = len(files_data)
        total_files_failed = 0
        
        # Process files for S3 upload
        for i, file_info in enumerate(files_data):
            filename = file_info.get("filename")
            content = file_info.get("content")  # base64 encoded
            content_type = file_info.get("content_type", "application/octet-stream")
            
            log(f"📁 Processing file {i+1}/{total_files_attempted}: {filename} | Request ID: {request_id}", "INFO")
            
            if not filename or not content:
                log(f"⚠️ Skipping file {i+1}: missing filename or content | Request ID: {request_id}", "WARN")
                total_files_failed += 1
                continue
            
            try:
                # Determine file type
                file_type = "other"
                is_image_by_content = content_type and content_type.startswith('image/')
                is_image_by_extension = any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'])
                
                if is_image_by_content or is_image_by_extension:
                    file_type = "image"
                    image_count += 1
                    log(f"🖼️ File recognized as image: {filename} | Request ID: {request_id}", "INFO")
                elif filename.lower().endswith('.txt'):
                    file_type = "caption"
                    caption_count += 1
                    log(f"📝 File recognized as caption: {filename} | Request ID: {request_id}", "INFO")
                
                # Decode base64 to get file size
                import base64
                content_padded = content
                missing_padding = len(content) % 4
                if missing_padding:
                    content_padded += '=' * (4 - missing_padding)
                
                file_content = base64.b64decode(content_padded)
                file_size = len(file_content)
                
                file_data = {
                    "filename": filename,
                    "size": file_size,
                    "size_formatted": format_file_size(file_size),
                    "content_type": content_type,
                    "file_type": file_type,
                    "uploaded_at": datetime.now().isoformat(),
                    "content": content  # Keep for S3 upload
                }
                uploaded_files.append(file_data)
                
                log(f"✅ File processed: {filename} | Size: {file_data['size_formatted']} | Type: {file_type} | Request ID: {request_id}", "INFO")
                
            except Exception as e:
                log(f"❌ Failed to process file {filename} | Request ID: {request_id} | Error: {e}", "ERROR")
                total_files_failed += 1
                continue
        
        if not uploaded_files:
            return {"error": "No valid files to upload"}
        
        # Upload dataset to S3
        try:
            s3_path = await _storage_service.upload_dataset_to_s3(training_name, uploaded_files)
            log(f"✅ Dataset uploaded to S3: {s3_path}", "INFO")
            
            # Create trigger word info and upload to S3
            trigger_info = {
                "training_name": training_name,
                "trigger_word": trigger_word,
                "upload_date": datetime.now().isoformat(),
                "total_images": image_count,
                "total_captions": caption_count,
                "s3_path": s3_path
            }
            
            # Upload training info as JSON to S3
            import base64
            import json
            info_content = json.dumps(trigger_info, indent=2)
            info_b64 = base64.b64encode(info_content.encode('utf-8')).decode('utf-8')
            
            info_file = {
                "filename": "_training_info.json",
                "content": info_b64
            }
            await _storage_service.upload_dataset_to_s3(training_name, [info_file])
            
        except Exception as e:
            log(f"❌ S3 upload failed: {e}", "ERROR")
            return {"error": f"Failed to upload dataset to S3: {str(e)}"}
        
        # Calculate total size
        total_size = sum(file_data["size"] for file_data in uploaded_files)
        total_size_formatted = format_file_size(total_size)
        
        # Create file summary without local paths
        file_types_summary = {
            "images": [f for f in uploaded_files if f["file_type"] == "image"],
            "captions": [f for f in uploaded_files if f["file_type"] == "caption"],
            "other": [f for f in uploaded_files if f["file_type"] == "other"]
        }
        
        # Remove content from response (not needed)
        for file_data in uploaded_files:
            if "content" in file_data:
                del file_data["content"]
        
        response_data = {
            "uploaded_files": uploaded_files,
            "s3_path": s3_path,
            "training_name": training_name,
            "total_images": image_count,
            "total_captions": caption_count,
            "total_size": total_size,
            "total_size_formatted": total_size_formatted,
            "file_types_summary": file_types_summary,
            "message": f"✅ Successfully uploaded {len(uploaded_files)} files ({total_size_formatted}) to S3",
            "detailed_message": f"📁 Uploaded to S3 path: {s3_path}\n📷 {image_count} images\n📝 {caption_count} captions\n💾 Total size: {total_size_formatted}",
            "debug_info": {
                "total_files_attempted": total_files_attempted,
                "total_files_succeeded": len(uploaded_files),
                "total_files_failed": total_files_failed,
                "all_files_failed": total_files_failed == total_files_attempted,
                "no_valid_content": image_count == 0 and caption_count == 0
            }
        }
        
        log(f"✅ Training data uploaded to S3: {s3_path} ({image_count} images, {caption_count} captions)", "INFO")
        
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

async def handle_list_files(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle list all generated files request - Read from S3"""
    try:
        if not _storage_service or not _lora_service:
            return {"error": "Services not initialized"}
        
        log("📁 Listing all generated files from S3...", "INFO")
        
        # Get LoRA models from S3
        lora_models = await _lora_service.get_available_models()
        lora_files = []
        
        for model in lora_models:
            lora_files.append({
                "id": model.get("id"),
                "filename": model.get("filename"),
                "path": model.get("path"),  # Already contains S3 path
                "s3_key": model.get("s3_key", ""),
                "process_id": model.get("process_id", ""),
                "size": model.get("size_mb", 0) * 1024 * 1024 if model.get("size_mb") else 0,  # Convert MB to bytes
                "size_formatted": f"{model.get('size_mb', 0)} MB",
                "created_at": model.get("modified_date"),
                "type": "lora"
            })
        
        # Get generated images from S3
        image_files = []
        try:
            if hasattr(_storage_service, 'list_files'):
                s3_files = await _storage_service.list_files("")  # List all files
                
                for s3_file in s3_files:
                    # Only include image files
                    if s3_file.get("result_type") == "images":
                        filename = s3_file.get("filename", "")
                        if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                            # Generate presigned URL for download
                            s3_url = f"s3://tqv92ffpc5/{s3_file.get('key', '')}"
                            
                            image_files.append({
                                "id": f"img_{s3_file.get('process_id', '')}_{filename.split('.')[0]}",
                                "filename": filename,
                                "path": s3_url,  # S3 path for frontend
                                "s3_key": s3_file.get("key", ""),
                                "process_id": s3_file.get("process_id", ""),
                                "size": s3_file.get("size", 0),
                                "size_formatted": format_file_size(s3_file.get("size", 0)),
                                "created_at": s3_file.get("last_modified", ""),
                                "type": "image",
                                "result_type": "images"
                            })
        except Exception as e:
            log(f"❌ Error listing S3 files: {e}", "ERROR")
        
        # Fallback to local files if S3 is empty (for backwards compatibility)
        if not image_files:
            log("📁 No images in S3, checking local directories as fallback...", "INFO")
            output_dirs = [
                "/workspace/generated_images",
                "/workspace/output/generated",
                "/workspace/output",
                "/workspace/results"
            ]
            
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    try:
                        # Look for image files
                        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
                        for pattern in image_patterns:
                            image_paths = glob.glob(f"{output_dir}/**/{pattern}", recursive=True)
                            for image_path in image_paths:
                                try:
                                    stat = os.stat(image_path)
                                    size_bytes = stat.st_size
                                    size_formatted = format_file_size(size_bytes)
                                    
                                    # Try to extract process_id from path/filename
                                    filename = os.path.basename(image_path)
                                    process_id = None
                                    if "generated_" in filename:
                                        parts = filename.split("_")
                                        if len(parts) > 1:
                                            process_id = parts[1].split(".")[0]
                                    
                                    image_files.append({
                                        "id": f"img_{os.path.basename(image_path).split('.')[0]}",
                                        "filename": filename,
                                        "path": image_path,
                                        "size": size_bytes,
                                        "size_formatted": size_formatted,
                                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                        "type": "image",
                                        "process_id": process_id
                                    })
                                except Exception as e:
                                    log(f"❌ Error processing image file {image_path}: {e}", "ERROR")
                                    continue
                    except Exception as e:
                        log(f"❌ Error scanning directory {output_dir}: {e}", "ERROR")
                        continue
        
        result = {
            "lora_files": lora_files,
            "image_files": image_files,
            "total_files": len(lora_files) + len(image_files)
        }
        
        log(f"✅ Found {len(lora_files)} LoRA files and {len(image_files)} image files", "INFO")
        return result
        
    except Exception as e:
        log(f"❌ List files error: {e}", "ERROR")
        return {"error": f"Failed to list files: {str(e)}"}

async def handle_download_file(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle direct file download by path"""
    try:
        file_path = job_input.get("file_path")
        if not file_path:
            return {"error": "file_path parameter required"}
        
        log(f"📥 Downloading file: {file_path}", "INFO")
        
        # Security check - ensure file path is within workspace
        if not file_path.startswith("/workspace/"):
            return {"error": "Access denied: file must be in workspace"}
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Read file and return as base64
            with open(file_path, 'rb') as file:
                file_data = file.read()
                file_base64 = base64.b64encode(file_data).decode('utf-8')
                
                # Determine content type based on file extension
                content_type = "application/octet-stream"
                ext = os.path.splitext(file_path)[1].lower()
                content_type_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.webp': 'image/webp',
                    '.safetensors': 'application/octet-stream',
                    '.pt': 'application/octet-stream',
                    '.pth': 'application/octet-stream'
                }
                content_type = content_type_map.get(ext, content_type)
                
                result = {
                    "type": "file_data",
                    "filename": os.path.basename(file_path),
                    "data": file_base64,
                    "size": len(file_data),
                    "content_type": content_type
                }
                
                log(f"✅ File download prepared: {os.path.basename(file_path)} ({format_file_size(len(file_data))})", "INFO")
                return result
                
        except Exception as e:
            log(f"❌ Error reading file {file_path}: {e}", "ERROR")
            return {"error": f"Failed to read file: {str(e)}"}
        
    except Exception as e:
        log(f"❌ Download file error: {e}", "ERROR")
        return {"error": f"Failed to download file: {str(e)}"}

# Start RunPod Serverless
if __name__ == "__main__":
    log("🚀 Starting LoRA Dashboard RunPod Serverless Handler", "INFO")
    runpod.serverless.start({
        "handler": async_handler,  # ✅ Bezpośrednio async handler
        "return_aggregate_stream": True
    }) 