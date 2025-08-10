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

✨ IMPROVEMENTS:
- Enhanced error handling with retries and timeouts
- GPU memory management and cleanup
- Pydantic validation for all inputs
- S3 synchronization for process states
- Metrics and monitoring
- Security improvements with file size limits
- Proper path handling for training/generation
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
from typing import Dict, Any, Optional, List
import io
import json
import traceback
from contextlib import asynccontextmanager

# Import our enhanced utilities and models
try:
    from models import (
        validate_request, ProcessInfo, 
        TrainingConfig, GenerationConfig, UploadTrainingData,
        ProcessStatusRequest, BulkDownloadRequest, DownloadFileRequest,
        ListFilesRequest, HealthCheckRequest, ProcessListRequest, LoRAModelsRequest
    )
    from utils import (
        log, format_file_size, track_metric, get_metrics,
        GPUManager, retry_with_backoff, execute_with_timeout,
        normalize_workspace_path, CircuitBreaker, batch_process,
        cleanup_old_files, get_memory_usage, validate_environment
    )
    ENHANCED_IMPORTS = True
except ImportError:
    # Fallback if new files not available yet
    ENHANCED_IMPORTS = False
    def log(message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        sys.stdout.flush()
    
    def format_file_size(bytes_size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

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

# Optional dotenv to load local env/config without committing secrets
try:
    from dotenv import load_dotenv
    # Load defaults from common locations; ignore if missing
    load_dotenv()  # .env in CWD if present
    # Also try explicit config.env in common locations
    for candidate_env in ("/config.env", "/workspace/config.env", os.path.join(os.path.dirname(__file__), "config.env")):
        try:
            load_dotenv(candidate_env)
        except Exception:
            pass
except Exception:
    # Safe no-op if python-dotenv is unavailable
    pass

# Global flag to track if environment is setup
ENVIRONMENT_READY = False
SETUP_LOCK = threading.Lock()

# Global process management (in-memory storage with S3 sync)
RUNNING_PROCESSES: Dict[str, Dict[str, Any]] = {}
PROCESS_LOCK = threading.Lock()

# Circuit breakers for external services
S3_CIRCUIT_BREAKER = None
AI_TOOLKIT_CIRCUIT_BREAKER = None

# Initialize circuit breakers if enhanced imports available
if ENHANCED_IMPORTS:
    S3_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    AI_TOOLKIT_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, recovery_timeout=120)

# Use enhanced logging if available, otherwise fallback is defined above

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout configurations
TRAINING_TIMEOUT = int(os.environ.get('TRAINING_TIMEOUT', 7200))  # 2 hours default
GENERATION_TIMEOUT = int(os.environ.get('GENERATION_TIMEOUT', 600))  # 10 minutes default
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))

# Enhanced process management with S3 synchronization
class ProcessManager:
    """Centralized process management with S3 sync"""
    
    @staticmethod
    def add_process(process_id: str, process_type: str, status: str, config: Dict[str, Any]):
        """Add a new process to tracking with immediate S3 sync"""
        with PROCESS_LOCK:
            process_info = {
                "id": process_id,
                "type": process_type,
                "status": status,
                "config": config,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "output_path": None,
                "error": None,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'local'),
                "job_id": os.environ.get('RUNPOD_JOB_ID', None)
            }
            RUNNING_PROCESSES[process_id] = process_info
            
            # Schedule S3 sync in background
            if _storage_service:
                threading.Thread(
                    target=lambda: asyncio.run(ProcessManager._sync_to_s3(process_id, process_info)),
                    daemon=True
                ).start()
    
    @staticmethod
    def update_process_status(process_id: str, status: str, output_path: str = None, error: str = None, metrics: Dict = None):
        """Update process status with immediate S3 sync"""
        with PROCESS_LOCK:
            if process_id in RUNNING_PROCESSES:
                RUNNING_PROCESSES[process_id]["status"] = status
                RUNNING_PROCESSES[process_id]["updated_at"] = datetime.now().isoformat()
                if output_path:
                    RUNNING_PROCESSES[process_id]["output_path"] = output_path
                if error:
                    RUNNING_PROCESSES[process_id]["error"] = error
                if metrics:
                    RUNNING_PROCESSES[process_id]["metrics"] = metrics
                
                # Schedule S3 sync
                if _storage_service:
                    process_info = RUNNING_PROCESSES[process_id].copy()
                    threading.Thread(
                        target=lambda: asyncio.run(ProcessManager._sync_to_s3(process_id, process_info)),
                        daemon=True
                    ).start()
    
    @staticmethod
    async def _sync_to_s3(process_id: str, process_info: Dict[str, Any]):
        """Sync process info to S3 (internal)"""
        try:
            if _storage_service and hasattr(_storage_service, 'save_process_status_to_s3'):
                if S3_CIRCUIT_BREAKER:
                    await S3_CIRCUIT_BREAKER.call(
                        _storage_service.save_process_status_to_s3,
                        process_id, process_info
                    )
                else:
                    await _storage_service.save_process_status_to_s3(process_id, process_info)
        except Exception as e:
            log(f"⚠️ Failed to sync process {process_id} to S3: {e}", "WARN")
    
    @staticmethod
    def get_process(process_id: str) -> Optional[Dict[str, Any]]:
        """Get process by ID"""
        with PROCESS_LOCK:
            return RUNNING_PROCESSES.get(process_id)
    
    @staticmethod
    def get_all_processes() -> List[Dict[str, Any]]:
        """Get all processes"""
        with PROCESS_LOCK:
            return list(RUNNING_PROCESSES.values())

# Backward compatibility aliases
def add_process(*args, **kwargs):
    return ProcessManager.add_process(*args, **kwargs)

def update_process_status(*args, **kwargs):
    return ProcessManager.update_process_status(*args, **kwargs)

def get_process(*args, **kwargs):
    return ProcessManager.get_process(*args, **kwargs)

def get_all_processes(*args, **kwargs):
    return ProcessManager.get_all_processes(*args, **kwargs)

@track_metric("environment_setup") if ENHANCED_IMPORTS else lambda f: f
def setup_environment():
    """Enhanced environment setup with GPU cleanup and validation"""
    global ENVIRONMENT_READY
    
    if ENVIRONMENT_READY:
        log("Environment already ready", "INFO")
        return True
    
    with SETUP_LOCK:
        # Double-check after acquiring lock
        if ENVIRONMENT_READY:
            return True
    
        try:
            log("🚀 Setting up enhanced environment...", "INFO")
            
            # Step 0: Clean up GPU memory if available
            if ENHANCED_IMPORTS and GPUManager:
                log("🧠 Cleaning GPU memory before setup...", "INFO")
                GPUManager.cleanup_gpu_memory()
                gpu_info = GPUManager.get_gpu_memory_info()
                if gpu_info.get("available"):
                    for gpu in gpu_info.get("gpus", []):
                        log(f"🎮 GPU {gpu['device']}: {gpu['name']} - Free: {gpu['free_mb']}MB", "INFO")
            
            # Step 1: Setup directories with proper permissions
            log("📁 Creating workspace directories...", "INFO")
            directories = [
                "/workspace",
                "/workspace/training_data",
                "/workspace/models",
                "/workspace/models/loras",
                "/workspace/logs",
                "/workspace/output",
                "/workspace/output/training",
                "/workspace/output/generation",
                "/workspace/cache",
                "/workspace/temp"
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                # Ensure proper permissions
                os.chmod(directory, 0o755)
            
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
                    
                    if ENHANCED_IMPORTS:
                        # Use retry logic for cloning - but sync since we're not in async
                        for attempt in range(3):
                            result = subprocess.run([
                                "git", "clone", "https://github.com/ostris/ai-toolkit.git"
                            ], capture_output=True, text=True, timeout=300)
                            if result.returncode == 0:
                                break
                            if attempt < 2:
                                log(f"⚠️ Clone attempt {attempt + 1} failed, retrying...", "WARN")
                                time.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                log(f"❌ AI-toolkit clone failed after 3 attempts: {result.stderr}", "ERROR")
                                return False
                    else:
                        result = subprocess.run([
                            "git", "clone", "https://github.com/ostris/ai-toolkit.git"
                        ], capture_output=True, text=True, timeout=300)
                        
                        if result.returncode != 0:
                            log(f"❌ AI-toolkit clone failed: {result.stderr}", "ERROR")
                            return False
                    
                    log("✅ AI-toolkit cloned successfully", "INFO")
                except Exception as e:
                    log(f"❌ AI-toolkit clone error: {e}", "ERROR")
                    return False
            else:
                log("✅ AI-toolkit already exists", "INFO")
    
            # Pin ai-toolkit to the merge commit of PR #343 for stability
            try:
                os.chdir("/workspace/ai-toolkit")
                # Ensure we have up-to-date refs when repo already exists
                subprocess.run(["git", "fetch", "--all", "--prune"], capture_output=True, text=True, timeout=180)

            # Find the merge commit for PR #343
                pr_merge_lookup = subprocess.run(
                    [
                        "git", "rev-list", "-n", "1", "--all",
                        "--grep=Merge pull request #343"
                    ],
                    capture_output=True, text=True, timeout=120
                )
                pr_merge_commit = pr_merge_lookup.stdout.strip()
    
                if pr_merge_commit:
                    # Create or reset a local branch to that commit and check it out
                    checkout_result = subprocess.run(
                        ["git", "checkout", "-B", "pr343", pr_merge_commit],
                        capture_output=True, text=True, timeout=180
                    )
                    if checkout_result.returncode == 0:
                        # Log the pinned commit for visibility
                        head_log = subprocess.run(
                            ["git", "log", "-1", "--oneline"],
                            capture_output=True, text=True, timeout=60
                        )
                        if head_log.returncode == 0:
                            log(f"📌 ai-toolkit pinned to: {head_log.stdout.strip()}", "INFO")
                        else:
                            log("⚠️ Unable to log pinned ai-toolkit commit", "WARN")
                    else:
                        log(f"⚠️ Failed to checkout ai-toolkit PR#343 commit: {checkout_result.stderr}", "WARN")
                else:
                    log("⚠️ PR #343 merge commit not found in ai-toolkit; using current default branch", "WARN")
            except Exception as e:
                log(f"⚠️ Failed to pin ai-toolkit to PR #343 commit: {e}", "WARN")
            
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
                log("🔍 Validating environment setup...", "INFO")
                validation = validate_environment()
                failed_checks = [k for k, v in validation.items() if not v]
                if failed_checks:
                    log(f"⚠️ Environment validation warnings: {failed_checks}", "WARN")
            
            # Step 10: Clean up old files on startup
            if ENHANCED_IMPORTS:
                log("🧹 Cleaning up old temporary files...", "INFO")
                try:
                    # Schedule cleanup in background thread since we're not in async context
                    threading.Thread(
                        target=lambda: asyncio.run(cleanup_old_files("/workspace/temp", retention_days=1)),
                        daemon=True
                    ).start()
                except Exception as e:
                    log(f"⚠️ Cleanup scheduling failed (non-critical): {e}", "WARN")
            
            log("✅ Complete enhanced environment setup ready!", "INFO")
            ENVIRONMENT_READY = True
            return True
            
        except Exception as e:
            log(f"❌ Setup error: {e}", "ERROR")
            log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

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
            """Start real LoRA training with AI toolkit (synchronous for Serverless)"""
            process_id = f"train_{uuid.uuid4().hex[:12]}"

            try:
                # Add to process tracking
                add_process(process_id, "training", "starting", {"config": config})
                log(f"🎯 Real training started: {process_id}", "INFO")

                # Run blocking training synchronously in executor and await completion
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._run_training_background, process_id, config)

                return process_id
            except Exception as e:
                log(f"❌ Training start failed: {e}", "ERROR")
                update_process_status(process_id, "failed", error=str(e))
                raise
        
        def _run_training_background(self, process_id: str, config):
            """Run training in background thread with enhanced error handling"""
            start_time = time.time()
            metrics = {"start_time": start_time}
            
            try:
                log(f"🚀 Training background thread started: {process_id}", "INFO")
                ProcessManager.update_process_status(process_id, "running")
                
                # Clean GPU memory before training
                if ENHANCED_IMPORTS and GPUManager:
                    log(f"🧠 Cleaning GPU memory before training: {process_id}", "INFO")
                    GPUManager.cleanup_gpu_memory()
                    
                    # Check if enough memory available
                    if not GPUManager.check_gpu_memory_available(required_mb=8192):  # 8GB minimum
                        raise Exception("Insufficient GPU memory available for training")
                
                # Parse and validate config
                try:
                    config_data = yaml.safe_load(config)

                    def resolve_and_download(ds_cfg: Dict[str, Any]) -> bool:
                        if not isinstance(ds_cfg, dict):
                            return False
                        dataset_path = ds_cfg.get('folder_path')
                        if not dataset_path:
                            return False
                        log(f"📂 Dataset path from config: {dataset_path}", "INFO")
                        needs_download = dataset_path.startswith('s3://') or not dataset_path.startswith('/')
                        if not needs_download:
                            if ENHANCED_IMPORTS:
                                ds_cfg['folder_path'] = normalize_workspace_path(dataset_path)
                            return False

                        log(f"📥 Preparing to download dataset from S3: {dataset_path}", "INFO")
                        if dataset_path.startswith('s3://'):
                            s3_parts = dataset_path.replace('s3://', '').split('/', 1)
                            bucket = s3_parts[0]
                            s3_key = s3_parts[1] if len(s3_parts) > 1 else ''
                            # If bucket matches configured bucket, strip it; otherwise use provided key as-is
                            if bucket == self.bucket_name:
                                s3_dataset_path = s3_key
                            else:
                                log(f"⚠️ Different S3 bucket specified: {bucket}", "WARN")
                                s3_dataset_path = dataset_path
                        else:
                            s3_dataset_path = f"lora-dashboard/datasets/{dataset_path}"

                        local_dataset_path = f"/workspace/training_data/{process_id}"
                        os.makedirs(local_dataset_path, exist_ok=True)

                        if _storage_service and hasattr(_storage_service, 'download_dataset_from_s3'):
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(
                                    _storage_service.download_dataset_from_s3(s3_dataset_path, local_dataset_path)
                                )
                                loop.close()
                                ds_cfg['folder_path'] = local_dataset_path
                                log(f"✅ Dataset downloaded to {local_dataset_path}", "INFO")
                                return True
                            except Exception as e:
                                log(f"❌ Dataset download failed: {e}", "ERROR")
                                return False
                        else:
                            log(f"❌ Storage service not available for dataset download", "ERROR")
                            return False

                    changed = False
                    root_cfg = config_data.get('config', {}) if isinstance(config_data, dict) else {}

                    # Top-level datasets
                    top_datasets = root_cfg.get('datasets')
                    if isinstance(top_datasets, list) and top_datasets:
                        if resolve_and_download(top_datasets[0]):
                            changed = True
                    elif isinstance(root_cfg.get('dataset'), dict):
                        if resolve_and_download(root_cfg['dataset']):
                            changed = True

                    # Process-level datasets
                    processes = root_cfg.get('process', [])
                    if isinstance(processes, list):
                        for proc in processes:
                            if isinstance(proc, dict):
                                if isinstance(proc.get('datasets'), list) and proc['datasets']:
                                    if resolve_and_download(proc['datasets'][0]):
                                        changed = True
                                elif isinstance(proc.get('dataset'), dict):
                                    if resolve_and_download(proc['dataset']):
                                        changed = True

                    if changed:
                        config = yaml.dump(config_data)

                except Exception as e:
                    log(f"⚠️ Dataset handling failed, continuing with original config: {e}", "WARNING")
                
                # Dynamically adjust sample_every to ensure samples are generated
                try:
                    config_data = yaml.safe_load(config)
                    if isinstance(config_data, dict):
                        cfg = config_data.get('config', {})
                        processes = cfg.get('process', [])
                        
                        for process in processes:
                            if isinstance(process, dict):
                                train_config = process.get('train', {})
                                sample_config = process.get('sample', {})
                                
                                steps = train_config.get('steps', 1000)
                                current_sample_every = sample_config.get('sample_every', 100)
                                
                                # Ensure sample_every allows at least 2-3 samples during training
                                optimal_sample_every = min(current_sample_every, max(10, steps // 3))
                                
                                if optimal_sample_every != current_sample_every:
                                    sample_config['sample_every'] = optimal_sample_every
                                    log(f"🔧 Adjusted sample_every from {current_sample_every} to {optimal_sample_every} (steps: {steps})", "INFO")
                        
                        config = yaml.dump(config_data)
                except Exception as e:
                    log(f"⚠️ Failed to adjust sample_every, using original config: {e}", "WARNING")
                
                # Setup environment
                log(f"🔧 Setting up environment for: {process_id}", "INFO")
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    log(f"❌ HF_TOKEN not found in environment for: {process_id}", "ERROR")
                    raise Exception("HF_TOKEN not found in environment")
                
                log(f"✅ HF_TOKEN found for: {process_id}", "INFO")
                
                # Create YAML config file in proper location
                config_dir = "/workspace/configs/training"
                os.makedirs(config_dir, exist_ok=True)
                config_path = f"{config_dir}/training_{process_id}.yaml"
                
                log(f"💾 Writing training config to: {config_path}", "INFO")
                with open(config_path, 'w') as f:
                    f.write(config)
                
                # Also save a backup copy
                backup_path = f"/workspace/logs/training_{process_id}_config.yaml"
                with open(backup_path, 'w') as f:
                    f.write(config)
                
                log(f"✅ Training config saved: {config_path}", "INFO")
                
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
                
                # Setup cache directories and cleanup before training
                cache_base = "/workspace/cache"
                cache_dirs = [
                    f"{cache_base}/hf",
                    f"{cache_base}/hub", 
                    f"{cache_base}/transformers",
                    f"{cache_base}/diffusers"
                ]
                
                for cache_dir in cache_dirs:
                    os.makedirs(cache_dir, exist_ok=True)
                
                # Check available disk space and cleanup if needed
                try:
                    import shutil
                    total, used, free = shutil.disk_usage("/workspace")
                    free_gb = free // (1024**3)
                    log(f"💾 Available disk space: {free_gb}GB", "INFO")
                    
                    # If less than 5GB free, try cleanup
                    if free_gb < 5:
                        log(f"⚠️ Low disk space ({free_gb}GB), attempting cleanup", "WARN")
                        
                        # Clean old cache files older than 1 day
                        cutoff_time = time.time() - (24 * 3600)  # 1 day ago
                        
                        for cache_dir in cache_dirs:
                            if os.path.exists(cache_dir):
                                for root, dirs, files in os.walk(cache_dir):
                                    for file in files:
                                        try:
                                            file_path = os.path.join(root, file)
                                            if os.path.getmtime(file_path) < cutoff_time:
                                                os.remove(file_path)
                                        except Exception:
                                            continue
                        
                        # Check space again
                        total, used, free = shutil.disk_usage("/workspace")
                        free_gb = free // (1024**3)
                        log(f"💾 Disk space after cleanup: {free_gb}GB", "INFO")
                        
                except Exception as e:
                    log(f"⚠️ Disk space check failed: {e}", "WARN")
                
                # Setup environment variables with comprehensive HF cache redirection
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HUGGING_FACE_HUB_TOKEN": hf_token,
                    "HF_TOKEN": hf_token,
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    # Comprehensive HF cache redirection
                    "HF_HOME": f"{cache_base}/hf",
                    "XDG_CACHE_HOME": cache_base,
                    "HUGGINGFACE_HUB_CACHE": f"{cache_base}/hub",
                    "DIFFUSERS_CACHE": f"{cache_base}/diffusers",
                    # Additional cache controls
                    "HF_HUB_CACHE": f"{cache_base}/hub",
                    "TORCH_HOME": f"{cache_base}/torch"
                })
                # Remove deprecated cache variable to avoid FutureWarning
                env.pop("TRANSFORMERS_CACHE", None)
                
                # Check if AI toolkit exists
                ai_toolkit_path = "/workspace/ai-toolkit/run.py"
                if not os.path.exists(ai_toolkit_path):
                    raise Exception(f"AI toolkit not found at {ai_toolkit_path}")
                
                log(f"🚀 Starting AI toolkit training: {ai_toolkit_path}", "INFO")
                
                # Run AI toolkit training with timeout and proper error handling
                cmd = [sys.executable, ai_toolkit_path, config_path]
                log(f"🎯 Training command: {' '.join(cmd)}", "INFO")
                
                # Use enhanced execution if available
                if ENHANCED_IMPORTS:
                    try:
                        # Run with timeout and retries
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            execute_with_timeout(cmd, timeout=TRAINING_TIMEOUT, env=env)
                        )
                        loop.close()
                    except subprocess.TimeoutExpired:
                        raise Exception(f"Training timed out after {TRAINING_TIMEOUT} seconds")
                    except Exception as e:
                        raise Exception(f"Training execution failed: {str(e)}")
                else:
                    # Fallback to simple subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
                
                # Log training output for debugging
                if result.stdout:
                    log(f"📋 Training stdout: {result.stdout[:1000]}...", "INFO")
                if result.stderr:
                    log(f"❌ Training stderr: {result.stderr[:1000]}...", "ERROR")
                
                if result.returncode == 0:
                    # Look for output files in correct locations
                    output_dirs = [
                        "/workspace/output",
                        f"/workspace/output/training_{process_id}",
                        f"/workspace/output/{process_id}",
                        "/workspace/models"
                    ]
                    
                    output_files = []
                    for output_dir in output_dirs:
                        if os.path.exists(output_dir):
                            # Look for LoRA model files
                            for ext in ['*.safetensors', '*.ckpt', '*.pt', '*.pth']:
                                output_files.extend(glob.glob(f"{output_dir}/**/{ext}", recursive=True))
                    
                    # Remove duplicates
                    output_files = list(set(output_files))
                    
                    if output_files:
                        log(f"🎆 Found {len(output_files)} output files", "INFO")
                        
                        # Create organized output structure
                        final_output_dir = f"/workspace/output/training/{process_id}/lora"
                        os.makedirs(final_output_dir, exist_ok=True)
                        
                        # Copy files to organized location
                        for output_file in output_files:
                            filename = os.path.basename(output_file)
                            dest_path = os.path.join(final_output_dir, filename)
                            shutil.copy2(output_file, dest_path)
                            log(f"📦 Copied output: {filename}", "INFO")
                        # Upload LoRA results to S3 synchronously
                        s3_output_path = None
                        if _storage_service and hasattr(_storage_service, 'upload_results_to_s3'):
                            try:
                                log(f"📤 Uploading LoRA results to S3: {process_id}", "INFO")
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(
                                    _storage_service.upload_results_to_s3(
                                        process_id, final_output_dir, 'lora'
                                    )
                                )
                                loop.close()
                                s3_output_path = f"s3://{self.bucket_name}/{self.prefix}/results/{process_id}/lora/"
                                log(f"✅ LoRA results uploaded to S3: {s3_output_path}", "INFO")
                            except Exception as e:
                                log(f"⚠️ Failed to upload LoRA to S3: {e}", "WARNING")

                        # Collect and upload training samples (images) if present
                        try:
                            samples_candidates = []
                            # Try to derive dataset name from YAML config
                            dataset_name = None
                            try:
                                if isinstance(config_data, dict):
                                    cfg = config_data.get('config') or {}
                                    dataset_name = cfg.get('name')
                            except Exception:
                                pass

                            if dataset_name:
                                samples_candidates.append(f"/workspace/output/{dataset_name}/samples")
                            # Additional generic locations
                            samples_candidates.extend([
                                f"/workspace/output/training/{process_id}/samples",
                                f"/workspace/output/{process_id}/samples",
                                "/workspace/output/samples"
                            ])

                            collected_samples = []
                            for candidate_dir in samples_candidates:
                                if os.path.isdir(candidate_dir):
                                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.gif']:
                                        collected_samples.extend(glob.glob(os.path.join(candidate_dir, '**', ext), recursive=True))

                            # Deduplicate and ensure files exist
                            collected_samples = list({p for p in collected_samples if os.path.isfile(p)})

                            if collected_samples:
                                final_samples_dir = f"/workspace/output/training/{process_id}/samples"
                                os.makedirs(final_samples_dir, exist_ok=True)
                                for src_file in collected_samples:
                                    try:
                                        shutil.copy2(src_file, os.path.join(final_samples_dir, os.path.basename(src_file)))
                                    except Exception as copy_err:
                                        log(f"⚠️ Failed to copy sample {src_file}: {copy_err}", "WARN")

                                if _storage_service and hasattr(_storage_service, 'upload_results_to_s3'):
                                    try:
                                        log(f"📤 Uploading training samples to S3: {process_id} | Files: {len(collected_samples)}", "INFO")
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(
                                            _storage_service.upload_results_to_s3(
                                                process_id, final_samples_dir, 'samples'
                                            )
                                        )
                                        loop.close()
                                        log(f"✅ Samples uploaded to S3: s3://{self.bucket_name}/{self.prefix}/results/{process_id}/samples/", "INFO")
                                    except Exception as e:
                                        log(f"⚠️ Failed to upload samples to S3: {e}", "WARNING")
                        except Exception as e:
                            log(f"⚠️ Sample collection step failed: {e}", "WARN")
                        
                        # Calculate metrics
                        duration = time.time() - start_time
                        metrics.update({
                            "duration_seconds": duration,
                            "output_files_count": len(output_files),
                            "total_output_size": sum(os.path.getsize(f) for f in output_files),
                            "success": True
                        })
                        
                        # Update process status with S3 path
                        ProcessManager.update_process_status(
                            process_id, "completed",
                            output_path=s3_output_path or final_output_dir,
                            metrics=metrics
                        )
                        log(f"✅ Training completed successfully: {process_id} (Duration: {duration:.1f}s)", "INFO")
                    else:
                        # No output files found but training succeeded
                        log(f"⚠️ Training completed but no model files found: {process_id}", "WARN")
                        ProcessManager.update_process_status(
                            process_id, "completed",
                            output_path="/workspace/output",
                            metrics={**metrics, "warning": "No model files generated"}
                        )
                else:
                    # Training command failed
                    error_msg = f"Training failed with exit code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nError output: {result.stderr[:1000]}..."
                    
                    metrics["error"] = error_msg
                    metrics["success"] = False
                    
                    ProcessManager.update_process_status(process_id, "failed", error=error_msg, metrics=metrics)
                    log(f"❌ Training failed: {process_id} - {error_msg}", "ERROR")
                    
            except Exception as e:
                error_msg = f"Training error: {str(e)}"
                metrics["error"] = error_msg
                metrics["success"] = False
                metrics["traceback"] = traceback.format_exc()
                
                ProcessManager.update_process_status(process_id, "failed", error=error_msg, metrics=metrics)
                log(f"❌ Training error: {process_id} - {error_msg}", "ERROR")
                log(f"Traceback: {traceback.format_exc()}", "ERROR")
            finally:
                # Clean up GPU memory after training
                if ENHANCED_IMPORTS and GPUManager:
                    log(f"🧠 Cleaning GPU memory after training: {process_id}", "INFO")
                    GPUManager.cleanup_gpu_memory()
                
                # Clean up temporary dataset if downloaded
                temp_dataset_path = f"/workspace/training_data/{process_id}"
                if os.path.exists(temp_dataset_path):
                    try:
                        shutil.rmtree(temp_dataset_path)
                        log(f"🧹 Cleaned up temporary dataset: {temp_dataset_path}", "INFO")
                    except Exception as e:
                        log(f"⚠️ Cleanup failed for temporary dataset {temp_dataset_path}: {e}", "WARN")
        
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
            """Run generation in background thread with enhanced handling"""
            start_time = time.time()
            metrics = {"start_time": start_time}
            
            try:
                ProcessManager.update_process_status(process_id, "running")
                log(f"🎨 Starting generation: {process_id}", "INFO")
                
                # Clean GPU memory before generation
                if ENHANCED_IMPORTS and GPUManager:
                    log(f"🧠 Cleaning GPU memory before generation: {process_id}", "INFO")
                    GPUManager.cleanup_gpu_memory()
                    
                    # Check memory for generation (less than training)
                    if not GPUManager.check_gpu_memory_available(required_mb=4096):  # 4GB minimum
                        raise Exception("Insufficient GPU memory available for generation")
                
                # Parse and validate generation config
                try:
                    config_data = yaml.safe_load(config)
                    
                    # Enhanced LoRA path handling
                    lora_path = None
                    if 'model' in config_data and 'lora_path' in config_data['model']:
                        lora_path = config_data['model']['lora_path']
                    elif 'lora' in config_data and 'path' in config_data['lora']:
                        lora_path = config_data['lora']['path']
                    
                    if lora_path:
                        log(f"🎯 LoRA path from config: {lora_path}", "INFO")
                        
                        # Handle S3 LoRA downloads
                        if lora_path.startswith('s3://'):
                            log(f"📥 Downloading LoRA from S3: {lora_path}", "INFO")
                            
                            # Parse S3 URI properly
                            s3_parts = lora_path.replace('s3://', '').split('/', 1)
                            bucket = s3_parts[0]
                            s3_key = s3_parts[1] if len(s3_parts) > 1 else ''
                            
                            # Validate bucket
                            if bucket != (_storage_service.bucket_name if _storage_service else os.environ.get('S3_BUCKET', '')):
                                log(f"⚠️ Different S3 bucket: {bucket}", "WARN")
                            
                            filename = os.path.basename(s3_key)
                            
                            # Create unique local directory for this generation
                            local_lora_dir = f"/workspace/models/loras/{process_id}"
                            os.makedirs(local_lora_dir, exist_ok=True)
                            local_lora_path = os.path.join(local_lora_dir, filename)
                            
                            # Download LoRA model
                            if _storage_service and hasattr(_storage_service, 's3_client') and _storage_service.s3_client:
                                try:
                                    log(f"📦 Downloading LoRA: {filename}", "INFO")
                                    _storage_service.s3_client.download_file(
                                        Bucket=bucket if bucket != _storage_service.bucket_name else _storage_service.bucket_name,
                                        Key=s3_key,
                                        Filename=local_lora_path
                                    )
                                    
                                    # Verify download
                                    if os.path.exists(local_lora_path) and os.path.getsize(local_lora_path) > 0:
                                        # Update config to use local path
                                        if 'model' in config_data:
                                            config_data['model']['lora_path'] = local_lora_path
                                        else:
                                            config_data['lora']['path'] = local_lora_path
                                        config = yaml.dump(config_data)
                                        log(f"✅ LoRA downloaded: {local_lora_path} ({format_file_size(os.path.getsize(local_lora_path))})", "INFO")
                                    else:
                                        raise Exception("Downloaded LoRA file is empty or missing")
                                except Exception as e:
                                    log(f"❌ LoRA download failed: {e}", "ERROR")
                                    raise
                            else:
                                log(f"❌ Storage service not available for LoRA download", "ERROR")
                                raise Exception("Cannot download LoRA without storage service")
                        else:
                            # Local LoRA path - normalize it
                            if ENHANCED_IMPORTS:
                                normalized_path = normalize_workspace_path(lora_path)
                                if 'model' in config_data:
                                    config_data['model']['lora_path'] = normalized_path
                                else:
                                    config_data['lora']['path'] = normalized_path
                                config = yaml.dump(config_data)
                                
                except Exception as e:
                    log(f"⚠️ LoRA download failed, continuing with original config: {e}", "WARNING")
                
                # Create proper output directory structure
                output_dir = f"/workspace/output/generation/{process_id}/images"
                os.makedirs(output_dir, exist_ok=True)
                
                # Write generation config file in proper location
                config_dir = "/workspace/configs/generation"
                os.makedirs(config_dir, exist_ok=True)
                config_path = f"{config_dir}/generation_{process_id}.yaml"
                
                with open(config_path, 'w') as f:
                    f.write(config)
                
                # Save backup config
                backup_path = f"/workspace/logs/generation_{process_id}_config.yaml"
                with open(backup_path, 'w') as f:
                    f.write(config)
                
                log(f"📝 Generation config saved: {config_path}", "INFO")
                
                # TODO: Implement actual Stable Diffusion pipeline
                # For now, create placeholder implementation
                log(f"🎨 Running generation pipeline: {process_id}", "INFO")
                
                # Simulate generation with proper timing
                num_images = config_data.get('num_images', 4)
                generation_time = 2.5 * num_images  # Simulate 2.5s per image
                time.sleep(min(generation_time, 30))  # Cap at 30s for placeholder
                
                # Create placeholder output files
                generated_files = []
                for i in range(num_images):
                    # In real implementation, these would be .png files
                    placeholder_path = os.path.join(output_dir, f"generated_{process_id}_{i:03d}.txt")
                    with open(placeholder_path, 'w') as f:
                        f.write(f"""Generated Image {i+1}
                        Process ID: {process_id}
                        Timestamp: {datetime.now().isoformat()}
                        Config: {config[:200]}...
                        """)
                    generated_files.append(placeholder_path)
                
                log(f"🎆 Generated {len(generated_files)} images", "INFO")
                
                # Upload results to S3
                s3_output_path = None
                if _storage_service and hasattr(_storage_service, 'upload_results_to_s3'):
                    try:
                        log(f"📤 Uploading generation results to S3: {process_id}", "INFO")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            _storage_service.upload_results_to_s3(
                                process_id, output_dir, 'images'
                            )
                        )
                        loop.close()
                        s3_output_path = f"s3://{self.bucket_name}/{self.prefix}/results/{process_id}/images/"
                        log(f"✅ Generation results uploaded to S3: {s3_output_path}", "INFO")
                    except Exception as e:
                        log(f"⚠️ Failed to upload generation results to S3: {e}", "WARNING")
                
                # Calculate metrics
                duration = time.time() - start_time
                metrics.update({
                    "duration_seconds": duration,
                    "images_generated": len(generated_files),
                    "total_output_size": sum(os.path.getsize(f) for f in generated_files),
                    "success": True
                })
                
                # Update process status
                ProcessManager.update_process_status(
                    process_id, "completed",
                    output_path=s3_output_path or output_dir,
                    metrics=metrics
                )
                log(f"✅ Generation completed successfully: {process_id} (Duration: {duration:.1f}s)", "INFO")
                
            except Exception as e:
                error_msg = f"Generation error: {str(e)}"
                metrics["error"] = error_msg
                metrics["success"] = False
                metrics["traceback"] = traceback.format_exc()
                
                ProcessManager.update_process_status(process_id, "failed", error=error_msg, metrics=metrics)
                log(f"❌ Generation error: {process_id} - {error_msg}", "ERROR")
                log(f"Traceback: {traceback.format_exc()}", "ERROR")
            finally:
                # Clean up GPU memory after generation
                if ENHANCED_IMPORTS and GPUManager:
                    log(f"🧠 Cleaning GPU memory after generation: {process_id}", "INFO")
                    GPUManager.cleanup_gpu_memory()
                
                # Clean up temporary LoRA if downloaded
                temp_lora_dir = f"/workspace/models/loras/{process_id}"
                if os.path.exists(temp_lora_dir):
                    try:
                        shutil.rmtree(temp_lora_dir)
                        log(f"🧹 Cleaned up temporary LoRA: {temp_lora_dir}", "INFO")
                    except Exception as e:
                        log(f"⚠️ Cleanup failed for temporary LoRA {temp_lora_dir}: {e}", "WARN")
        
        async def get_process(self, process_id):
            """Get process status with S3 fallback for cross-worker visibility"""
            # Local first
            local = get_process(process_id)
            if local:
                return local
            # Fallback to S3
            try:
                if _storage_service and hasattr(_storage_service, 'get_process_status_from_s3'):
                    s3_proc = await _storage_service.get_process_status_from_s3(process_id)
                    if s3_proc:
                        with PROCESS_LOCK:
                            RUNNING_PROCESSES[process_id] = s3_proc
                        return s3_proc
            except Exception as e:
                log(f"Failed S3 process lookup for {process_id}: {e}", "WARN")
            return {"status": "not_found"}
        
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
            
            # S3 Configuration (ENV-parametrized)
            self.bucket_name = os.environ.get("S3_BUCKET", "tqv92ffpc5")
            self.endpoint_url = os.environ.get("S3_ENDPOINT_URL", "https://s3api-eu-ro-1.runpod.io")
            self.region = os.environ.get("S3_REGION", "eu-ro-1")
            self.prefix = os.environ.get("S3_PREFIX", "lora-dashboard")
            
            if S3_AVAILABLE:
                try:
                    # Read AWS credentials from environment (provided via RunPod Secrets)
                    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
                    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

                    boto_config = BotoConfig(
                        s3={
                            'addressing_style': 'path'
                        },
                        signature_version='s3v4'
                    )

                    client_kwargs = {
                        'service_name': 's3',
                        'endpoint_url': self.endpoint_url,
                        'region_name': self.region,
                        'config': boto_config
                    }
                    # Attach credentials only if provided; otherwise rely on IAM role if available
                    if aws_access_key_id and aws_secret_access_key:
                        client_kwargs['aws_access_key_id'] = aws_access_key_id
                        client_kwargs['aws_secret_access_key'] = aws_secret_access_key

                    self.s3_client = boto3.client(**client_kwargs)
                    log("✅ S3 client initialized", "INFO")
                except Exception as e:
                    log(f"❌ S3 client init failed: {e}", "ERROR")
                    self.s3_client = None
            else:
                log("❌ boto3 not available, S3 features disabled", "WARNING")
                self.s3_client = None
        
        async def _s3_call(self, func, *args, **kwargs):
            """Wrapper to execute S3 operations with circuit breaker if available."""
            try:
                if 'S3_CIRCUIT_BREAKER' in globals() and S3_CIRCUIT_BREAKER:
                    return await S3_CIRCUIT_BREAKER.call(func, *args, **kwargs)
                # Fallback: direct call (sync)
                return func(*args, **kwargs)
            except Exception as e:
                raise
        
        async def health_check(self):
            return "healthy"
        
        async def cleanup_dataset_folder(self):
            """Delete all files under the fixed dataset prefix in S3."""
            if not self.s3_client:
                raise Exception("S3 client not available")
            try:
                s3_prefix = f"{self.prefix}/dataset/"
                log(f"🧹 Cleaning up S3 dataset folder: {s3_prefix}", "INFO")
                continuation_token = None
                total_deleted = 0
                while True:
                    params = {
                        'Bucket': self.bucket_name,
                        'Prefix': s3_prefix
                    }
                    if continuation_token:
                        params['ContinuationToken'] = continuation_token
                    response = await self._s3_call(self.s3_client.list_objects_v2, **params)
                    contents = response.get('Contents', []) if response else []
                    if not contents:
                        break
                    # Delete in batches of 1000 (S3 limit)
                    for i in range(0, len(contents), 1000):
                        batch = contents[i:i+1000]
                        delete_objects = [{'Key': obj['Key']} for obj in batch]
                        await self._s3_call(
                            self.s3_client.delete_objects,
                            Bucket=self.bucket_name,
                            Delete={'Objects': delete_objects}
                        )
                        total_deleted += len(delete_objects)
                    if response.get('IsTruncated'):
                        continuation_token = response.get('NextContinuationToken')
                    else:
                        break
                log(f"✅ Cleanup complete. Deleted: {total_deleted} objects", "INFO")
            except Exception as e:
                log(f"⚠️ Cleanup dataset folder failed: {e}", "WARN")

        async def delete_results_by_type(self, result_type: str) -> dict:
            """Delete all S3 objects for a given results type across all processes.

            Supported types: "lora", "samples", "images"
            """
            if not self.s3_client:
                return {"deleted": 0, "found": 0, "error": "S3 client not available"}

            valid_types = {"lora", "samples", "images"}
            if result_type not in valid_types:
                return {"deleted": 0, "found": 0, "error": f"Invalid result type: {result_type}"}

            try:
                base_prefix = f"{self.prefix}/results/"
                log(f"🧹 Deleting all '{result_type}' results under: {base_prefix}", "INFO")

                continuation_token = None
                keys_to_delete = []
                found = 0

                while True:
                    params = {
                        'Bucket': self.bucket_name,
                        'Prefix': base_prefix
                    }
                    if continuation_token:
                        params['ContinuationToken'] = continuation_token

                    response = await self._s3_call(self.s3_client.list_objects_v2, **params)
                    contents = response.get('Contents', []) if response else []
                    if not contents:
                        break

                    for obj in contents:
                        key = obj['Key']
                        # Match segment '/<type>/' to avoid accidental deletions
                        if f"/{result_type}/" in key:
                            keys_to_delete.append(key)
                            found += 1

                    if response.get('IsTruncated'):
                        continuation_token = response.get('NextContinuationToken')
                    else:
                        break

                deleted = 0
                # Delete in batches of up to 1000
                for i in range(0, len(keys_to_delete), 1000):
                    batch = keys_to_delete[i:i+1000]
                    if not batch:
                        continue
                    await self._s3_call(
                        self.s3_client.delete_objects,
                        Bucket=self.bucket_name,
                        Delete={'Objects': [{'Key': k} for k in batch]}
                    )
                    deleted += len(batch)
                    log(f"🗑️ Deleted batch {i//1000 + 1}: {len(batch)} objects", "INFO")

                log(f"✅ Deletion finished for '{result_type}': found={found}, deleted={deleted}", "INFO")
                return {"deleted": deleted, "found": found}

            except Exception as e:
                log(f"❌ Failed to delete results of type '{result_type}': {e}", "ERROR")
                return {"deleted": 0, "found": 0, "error": str(e)}
        
        async def upload_dataset_to_s3(self, training_name: str, files: list) -> str:
            """Upload training dataset files to S3"""
            if not self.s3_client:
                raise Exception("S3 client not available")
            
            try:
                # Fixed dataset path (singular): lora-dashboard/dataset
                s3_path = f"{self.prefix}/dataset"
                uploaded_files = []
                log(f"📤 Uploading dataset to S3: {s3_path} | Files: {len(files)}", "INFO")
                
                for file_info in files:
                    filename = file_info.get("filename")
                    content_b64 = file_info.get("content")
                    # Pad base64 if needed
                    missing_padding = len(content_b64) % 4
                    if missing_padding:
                        content_b64 += '=' * (4 - missing_padding)
                    file_data = base64.b64decode(content_b64)
                    
                    s3_key = f"{s3_path}/{filename}"
                    
                    await self._s3_call(
                        self.s3_client.put_object,
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Body=file_data,
                        ContentType=file_info.get("content_type") or "application/octet-stream"
                    )
                    uploaded_files.append(s3_key)
                    log(f"✅ Uploaded to S3: {s3_key}", "INFO")
                
                log(f"📦 Dataset upload complete: {len(uploaded_files)} files -> {s3_path}", "INFO")
                return s3_path
                
            except Exception as e:
                log(f"❌ S3 dataset upload failed: {e}", "ERROR")
                raise
        
        async def upload_training_files(self, files: list) -> dict:
            """Compatibility wrapper to upload training files batch.
            Uses enhanced batch uploader when available, otherwise falls back to upload_dataset_to_s3.
            Returns metadata including created S3 path and count of uploaded files."""
            from datetime import datetime
            training_name = f"training_{int(datetime.now().timestamp())}"
            try:
                log(f"📤 upload_training_files invoked | Files: {len(files)}", "INFO")
                # Prefer enhanced batch uploader if available
                if ENHANCED_IMPORTS:
                    try:
                        from storage_utils import S3StorageManager
                        # Fixed dataset path
                        s3_prefix = f"{self.prefix}/dataset"
                        manager = S3StorageManager()
                        uploaded_keys = await manager.batch_upload_files(files, s3_prefix)
                        return {"training_name": training_name, "s3_path": s3_prefix, "files": len(uploaded_keys)}
                    except Exception as e:
                        log(f"⚠️ Enhanced batch upload failed, falling back: {e}", "WARN")
                # Fallback
                s3_path = await self.upload_dataset_to_s3(training_name, files)
                return {"training_name": training_name, "s3_path": s3_path, "files": len(files)}
            except Exception as e:
                log(f"❌ upload_training_files error: {e}", "ERROR")
                raise

        async def download_dataset_from_s3(self, s3_path: str, local_path: str):
            """Download dataset from S3 to local directory"""
            if not self.s3_client:
                raise Exception("S3 client not available")
            
            try:
                log(f"📥 Downloading dataset from S3: {s3_path} -> {local_path}", "INFO")
                os.makedirs(local_path, exist_ok=True)
                
                # List all files in S3 path
                response = await self._s3_call(
                    self.s3_client.list_objects_v2,
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
                    
                    await self._s3_call(
                        self.s3_client.download_file,
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
                log(f"📤 Starting results upload to S3 | process_id: {process_id} | type: {result_type} | source: {local_path}", "INFO")
                
                if os.path.isfile(local_path):
                    # Single file
                    filename = os.path.basename(local_path)
                    s3_key = f"{s3_base_path}/{filename}"
                    
                    await self._s3_call(
                        self.s3_client.upload_file,
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
                            
                            await self._s3_call(
                                self.s3_client.upload_file,
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
                
                await self._s3_call(
                    self.s3_client.put_object,
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
                
                response = await self._s3_call(
                    self.s3_client.get_object,
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                
                status_data = json.loads(response['Body'].read().decode('utf-8'))
                try:
                    status = status_data.get('status') if isinstance(status_data, dict) else None
                    log(f"📥 Loaded process status from S3 for {process_id}{' | status: ' + str(status) if status else ''}", "INFO")
                except Exception:
                    pass
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
                response = await self._s3_call(
                    self.s3_client.list_objects_v2,
                    Bucket=self.bucket_name,
                    Prefix=f"{self.prefix}/processes/"
                )
                
                processes = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        if obj['Key'].endswith('.json'):
                            try:
                                process_response = await self._s3_call(
                                    self.s3_client.get_object,
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
        
        async def get_download_url(self, process_id, s3_key: str | None = None):
            """Generate presigned URL for a process output from S3.
            If s3_key is None, returns the first file's URL under the process."""
            if not self.s3_client:
                return None
            try:
                if not s3_key:
                    s3_prefix = f"{self.prefix}/results/{process_id}/"
                    response = await self._s3_call(
                        self.s3_client.list_objects_v2,
                        Bucket=self.bucket_name,
                        Prefix=s3_prefix,
                        MaxKeys=1
                    )
                    contents = response.get('Contents') if response else None
                    if not contents:
                        log(f"❌ No files found in S3 for process: {process_id}", "ERROR")
                        return None
                    first = contents[0]
                    s3_key = first['Key']
                    size = first.get('Size', 0)
                else:
                    # If key provided, validate object exists and get size
                    try:
                        head = await self._s3_call(
                            self.s3_client.head_object,
                            Bucket=self.bucket_name,
                            Key=s3_key
                        )
                        size = head.get('ContentLength', 0)
                    except Exception as e:
                        log(f"❌ S3 object not found or inaccessible: {s3_key} ({e})", "ERROR")
                        return {"error": f"S3 object not found: {s3_key}"}
                presigned_url = await self._s3_call(
                    self.s3_client.generate_presigned_url,
                    'get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': s3_key,
                        'ResponseContentDisposition': f'attachment; filename="{os.path.basename(s3_key)}"'
                    },
                    ExpiresIn=3600
                )
                log(f"🔗 Presigned URL generated for S3 key: {s3_key} ({format_file_size(size)})", "INFO")
                return {
                    "type": "url",
                    "url": presigned_url,
                    "filename": os.path.basename(s3_key),
                    "size": size
                }
            except Exception as e:
                log(f"❌ Error generating download URL: {e}", "ERROR")
                return None
        
        async def list_files(self, path):
            """List files from S3 for downloads with optional path filter and pagination"""
            if not self.s3_client:
                return []
            try:
                base_prefix = f"{self.prefix}/results/"
                s3_prefix = base_prefix
                if path:
                    # Normalize path, avoid leading slash
                    normalized = path.lstrip('/')
                    s3_prefix = f"{self.prefix}/{normalized}"
                log(f"📁 Listing S3 files under prefix: {s3_prefix}", "INFO")
                files = []
                continuation_token = None
                while True:
                    params = {
                        'Bucket': self.bucket_name,
                        'Prefix': s3_prefix
                    }
                    if continuation_token:
                        params['ContinuationToken'] = continuation_token
                    response = await self._s3_call(self.s3_client.list_objects_v2, **params)
                    contents = response.get('Contents', []) if response else []
                    for obj in contents:
                        s3_key = obj['Key']
                        key_parts = s3_key.replace(base_prefix, '').split('/')
                        if len(key_parts) >= 3:
                            process_id = key_parts[0]
                            result_type = key_parts[1]
                            filename = '/'.join(key_parts[2:])
                            files.append({
                                "key": s3_key,
                                "process_id": process_id,
                                "result_type": result_type,
                                "filename": filename,
                                "size": obj.get('Size', 0),
                                "last_modified": obj.get('LastModified').isoformat() if obj.get('LastModified') else None
                            })
                    if response.get('IsTruncated'):
                        continuation_token = response.get('NextContinuationToken')
                    else:
                        break
                log(f"📄 Listed {len(files)} files from prefix: {s3_prefix}", "INFO")
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

@track_metric("handler_request") if ENHANCED_IMPORTS else lambda f: f
async def async_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced RunPod Serverless handler with validation and monitoring
    
    Expected input format:
    {
        "input": {
            "type": "train" | "generate" | "health" | "processes" | "lora",
            "config": "YAML configuration string",
            "process_id": "optional process ID for status check"
        }
    }
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"
    runpod_job_id = os.environ.get('RUNPOD_JOB_ID', 'local')
    
    try:
        # Initialize services if needed
        await initialize_services()
        
        # Extract and validate input
        job_input = event.get("input", {})
        
        # Enhanced validation with Pydantic if available
        if ENHANCED_IMPORTS:
            try:
                validated_input = validate_request(job_input)
                # Convert back to dict for compatibility (Pydantic v2 prefers model_dump)
                job_input = (
                    validated_input.model_dump()
                    if hasattr(validated_input, "model_dump")
                    else validated_input.dict()
                )
                job_type = job_input.get("type")
            except Exception as validation_error:
                log(f"❌ Validation error: {validation_error}", "ERROR")
                return {
                    "error": f"Invalid request: {str(validation_error)}",
                    "request_id": request_id,
                    "validation_errors": str(validation_error)
                }
        else:
            # Fallback validation
            job_type = job_input.get("type")
            
            # Handle simple prompt requests (auto-detect as generation)
            if not job_type and job_input.get("prompt"):
                job_input["type"] = "generate"
                job_type = "generate"
                log("📝 Auto-detected generation request from prompt", "INFO")
        
        # Enhanced request logging with metadata
        log(f"📨 Incoming {job_type or 'unknown'} request | Request ID: {request_id} | RunPod Job: {runpod_job_id}", "INFO")
        
        # Log request details for debugging
        if job_type in ["train", "train_with_yaml"]:
            config_preview = job_input.get("config", job_input.get("yaml_config", ""))[:200]
            log(f"🎯 TRAINING request - Config preview: {config_preview}...", "INFO")
        elif job_type == "generate":
            prompt = job_input.get("prompt", "")
            if prompt:
                log(f"🎨 GENERATION request - Prompt: {prompt[:100]}...", "INFO")
        elif job_type == "upload_training_data":
            files_count = len(job_input.get("files", []))
            training_name = job_input.get("training_name", "unknown")
            log(f"📤 UPLOAD request - Training: {training_name}, Files: {files_count}", "INFO")
        
        # Create handler mapping for cleaner routing
        handlers = {
            "health": handle_health_check,
            "train": handle_training,
            "train_with_yaml": handle_training,
            "generate": handle_generation,
            "processes": handle_get_processes,
            "process_status": handle_process_status,
            "lora": handle_get_lora_models,
            "list_models": handle_get_lora_models,
            "cancel": handle_cancel_process,
            "download": handle_download_url,
            "download_by_key": handle_download_by_key if 'handle_download_by_key' in globals() else None,
            "upload_training_data": lambda x: handle_upload_training_data(x, request_id),
            "bulk_download": handle_bulk_download,
            "list_files": handle_list_files,
            "download_file": handle_download_file,
            "delete_results": handle_delete_results
        }
        
        # Route to appropriate handler
        handler = handlers.get(job_type)
        if handler:
            # Execute handler with timeout if enhanced imports available
            if ENHANCED_IMPORTS and asyncio.iscoroutinefunction(handler):
                # Wrap in timeout
                handler_timeout = 300  # 5 minutes default
                if job_type in ["train", "train_with_yaml"]:
                    handler_timeout = TRAINING_TIMEOUT
                elif job_type == "generate":
                    handler_timeout = GENERATION_TIMEOUT
                
                response = await asyncio.wait_for(
                    handler(job_input) if job_type != "upload_training_data" else handler(job_input),
                    timeout=handler_timeout
                )
            else:
                response = await handler(job_input) if job_type != "upload_training_data" else await handler(job_input)
        else:
            response = {
                "error": f"Unknown job type: {job_type}",
                "supported_types": list(handlers.keys()),
                "request_id": request_id
            }
        
        # Add request metadata to response
        response["request_id"] = request_id
        response["worker_id"] = os.environ.get('RUNPOD_POD_ID', 'local')
        response["runpod_job_id"] = runpod_job_id
        
        # Calculate and log metrics
        status = "success" if not response.get("error") else "error"
        end_time = time.time()
        # Extract timestamp from request_id (last part after last underscore)
        start_timestamp = int(request_id.split('_')[-1]) / 1000
        duration = end_time - start_timestamp
        
        log(f"✅ Request completed: {status} | Duration: {duration:.3f}s | Request ID: {request_id}", "INFO")
        
        # Log response details based on type
        if job_type == "processes" and status == "success":
            process_count = len(response.get("processes", []))
            log(f"📊 Returned {process_count} processes", "INFO")
        elif job_type in ["train", "train_with_yaml", "generate"] and status == "success":
            process_id = response.get("process_id")
            log(f"🎯 Started process: {process_id}", "INFO")
        elif job_type == "upload_training_data" and status == "success":
            files_count = len(response.get("uploaded_files", []))
            s3_path = response.get("s3_path")
            log(f"📤 Uploaded {files_count} files to: {s3_path}", "INFO")
        
        # Add timing metadata
        response["timing"] = {
            "duration_ms": int(duration * 1000),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
            
    except asyncio.TimeoutError:
        error_msg = f"Request timeout after {handler_timeout if 'handler_timeout' in locals() else 'unknown'} seconds"
        log(f"⏱️ {error_msg} | Request ID: {request_id}", "ERROR")
        return {
            "error": error_msg,
            "request_id": request_id,
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'local'),
            "timeout": True
        }
    except Exception as e:
        log(f"💥 Handler error: {e} | Request ID: {request_id}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        
        # Enhanced error response
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "request_id": request_id,
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'local'),
            "runpod_job_id": runpod_job_id if 'runpod_job_id' in locals() else None
        }
        
        # Add debug info in development
        if os.environ.get('DEBUG', '').lower() == 'true':
            error_response["traceback"] = traceback.format_exc()
        
        return error_response
    finally:
        # Log metrics if available
        if ENHANCED_IMPORTS and 'job_type' in locals():
            metrics = get_metrics()
            if metrics:
                handler_metrics = metrics.get("handler_request", {})
                if handler_metrics:
                    log(f"📊 Handler metrics - Requests: {handler_metrics.get('count', 0)}, Avg time: {handler_metrics.get('avg_time', 0):.3f}s, Errors: {handler_metrics.get('errors', 0)}", "INFO")

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

async def handle_get_lora_models(job_input: Dict[str, Any]) -> Dict[str, Any]:
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
    """Handle download request - returns presigned URL or file data

    Supports optional 's3_key' to generate a presigned URL for a specific
    object stored in S3. If 's3_key' is not provided, falls back to the
    first file under the process results path.
    """
    try:
        process_id = job_input.get("process_id")
        s3_key = job_input.get("s3_key")
        # Require at least process_id or s3_key
        if not process_id and not s3_key:
            return {"error": "Missing 'process_id' or 's3_key' parameter"}
        
        if not _process_manager and not s3_key:
            # For pure s3_key requests we can proceed without process manager
            return {"error": "Process manager not initialized"}
        
        # If process_id provided, validate existence and completion
        if process_id and _process_manager:
            process = await _process_manager.get_process(process_id)
            if not process:
                return {"error": "Process not found"}
            if process.get("status") != "completed":
                return {"error": "Process not completed"}
        
        if not _storage_service:
            return {"error": "Storage service not initialized"}
            
        # Get download data (presigned URL or file data)
        download_data = await _storage_service.get_download_url(process_id, s3_key)
        
        if download_data:
            log(f"✅ Download prepared for process {process_id}: {download_data['filename']} ({format_file_size(download_data['size'])})", "INFO")
            return download_data
        else:
            return {"error": "No download data available for this process"}
            
    except Exception as e:
        log(f"❌ Download error: {e}", "ERROR")
        return {"error": f"Failed to prepare download: {str(e)}"}

@track_metric("upload_training_data") if ENHANCED_IMPORTS else lambda f: f
async def handle_upload_training_data(job_input: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
    """Enhanced training data upload with validation and batch processing"""
    try:
        from datetime import datetime
        
        # Enhanced validation with Pydantic is already done in async_handler if available
        # Extract parameters
        training_name = job_input.get("training_name", f"training_{int(datetime.now().timestamp())}")
        trigger_word = job_input.get("trigger_word", "")
        cleanup_existing = job_input.get("cleanup_existing", True)
        files_data = job_input.get("files", [])
        
        # Basic validation
        if not files_data:
            return {"error": "No files provided", "request_id": request_id}
        
        if not _storage_service:
            return {"error": "Storage service not initialized", "request_id": request_id}
        
        # Additional validation if enhanced imports not available
        if not ENHANCED_IMPORTS:
            # Check file count limit
            if len(files_data) > 100:  # MAX_FILES_PER_UPLOAD
                return {"error": f"Too many files: {len(files_data)} (max: 100)", "request_id": request_id}
            
            # Validate training name
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', training_name):
                return {"error": "Invalid training name. Use only letters, numbers, hyphens and underscores", "request_id": request_id}
        
        log(f"🚀 Uploading dataset to S3: {training_name} | Files: {len(files_data)} | Request ID: {request_id}", "INFO")
        
        uploaded_files = []
        image_count = 0
        caption_count = 0
        total_files_attempted = len(files_data)
        total_files_failed = 0
        total_size_limit = 500 * 1024 * 1024  # 500MB total limit
        total_size_so_far = 0
        
        # Process files with size validation
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
                
                # Decode base64 to get file size and validate
                import base64
                content_padded = content
                missing_padding = len(content) % 4
                if missing_padding:
                    content_padded += '=' * (4 - missing_padding)
                
                file_content = base64.b64decode(content_padded)
                file_size = len(file_content)
                
                # Check individual file size
                if not ENHANCED_IMPORTS:  # Additional validation if models.py not available
                    max_file_size = 100 * 1024 * 1024  # 100MB per file
                    if file_size > max_file_size:
                        log(f"❌ File too large: {filename} ({format_file_size(file_size)})", "ERROR")
                        total_files_failed += 1
                        continue
                
                # Check total size limit
                total_size_so_far += file_size
                if total_size_so_far > total_size_limit:
                    log(f"❌ Total size limit exceeded at file {i+1}", "ERROR")
                    return {
                        "error": f"Total upload size exceeds limit ({format_file_size(total_size_limit)})",
                        "files_processed": len(uploaded_files),
                        "request_id": request_id
                    }
                
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
        
        # Clean up existing data if requested
        if cleanup_existing and _storage_service:
            try:
                if hasattr(_storage_service, 'cleanup_dataset_folder'):
                    await _storage_service.cleanup_dataset_folder()
                else:
                    log("⚠️ Cleanup method not available on storage service", "WARN")
            except Exception as e:
                log(f"⚠️ Cleanup failed: {e}", "WARN")
        
        # Upload dataset to S3 with batch processing
        try:
            # Use batch upload if enhanced imports available
            if ENHANCED_IMPORTS and hasattr(_storage_service, 'upload_dataset_to_s3'):
                # Batch upload for better performance
                log(f"📤 Starting batch upload to S3...", "INFO")
                
                # Upload files in batches of 10
                batch_size = 10
                for batch_start in range(0, len(uploaded_files), batch_size):
                    batch_end = min(batch_start + batch_size, len(uploaded_files))
                    batch = uploaded_files[batch_start:batch_end]
                    
                    if batch_start == 0:
                        # First batch creates the S3 path
                        s3_path = await _storage_service.upload_dataset_to_s3(training_name, batch)
                    else:
                        # Subsequent batches append to existing path
                        await _storage_service.upload_dataset_to_s3(training_name, batch)
                    
                    log(f"📦 Uploaded batch {batch_start//batch_size + 1}/{(len(uploaded_files) + batch_size - 1)//batch_size}", "INFO")
            else:
                # Fallback to single upload
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
        
        # Validate we have proper image/caption pairs
        if image_count == 0:
            log(f"⚠️ No images found in upload", "WARN")
        
        if caption_count < image_count:
            log(f"⚠️ Missing captions: {image_count} images but only {caption_count} captions", "WARN")
        
        response_data = {
            "uploaded_files": uploaded_files,
            "s3_path": s3_path,
            "training_name": training_name,
            "trigger_word": trigger_word,
            "total_images": image_count,
            "total_captions": caption_count,
            "total_size": total_size,
            "total_size_formatted": total_size_formatted,
            "file_types_summary": file_types_summary,
            "message": f"✅ Successfully uploaded {len(uploaded_files)} files ({total_size_formatted}) to S3",
            "detailed_message": f"📁 Uploaded to S3 path: {s3_path}\n📷 {image_count} images\n📝 {caption_count} captions\n💾 Total size: {total_size_formatted}",
            "warnings": [],
            "statistics": {
                "total_files_attempted": total_files_attempted,
                "total_files_succeeded": len(uploaded_files),
                "total_files_failed": total_files_failed,
                "success_rate": (len(uploaded_files) / total_files_attempted * 100) if total_files_attempted > 0 else 0,
                "average_file_size": total_size // len(uploaded_files) if uploaded_files else 0
            }
        }
        
        # Add warnings if needed
        if image_count == 0:
            response_data["warnings"].append("No images found in upload")
        if caption_count < image_count:
            response_data["warnings"].append(f"Missing captions for {image_count - caption_count} images")
        if total_files_failed > 0:
            response_data["warnings"].append(f"{total_files_failed} files failed to process")
        
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
                    
                    # Generate download URL per file
                    file_key = file_info.get('key', '')
                    download_info = await _storage_service.get_download_url(process_id, file_key)
                    if download_info and isinstance(download_info, dict) and download_info.get('url'):
                        download_items.append({
                            "filename": os.path.basename(file_key),
                            "url": download_info.get('url'),
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
        
        # Get generated images (and training samples) from S3
        image_files = []
        try:
            if hasattr(_storage_service, 'list_files'):
                s3_files = await _storage_service.list_files("")  # List all files
                
                for s3_file in s3_files:
                    # Include both generation images and training samples
                    if s3_file.get("result_type") in ["images", "samples"]:
                        filename = s3_file.get("filename", "")
                        if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                            s3_url = f"s3://{_storage_service.bucket_name}/{s3_file.get('key', '')}" if _storage_service else f"s3://{os.environ.get('S3_BUCKET', 'tqv92ffpc5')}/{s3_file.get('key', '')}"
                            image_files.append({
                                "id": f"img_{s3_file.get('process_id', '')}_{filename.split('.')[0]}",
                                "filename": filename,
                                "path": s3_url,
                                "s3_key": s3_file.get("key", ""),
                                "process_id": s3_file.get("process_id", ""),
                                "size": s3_file.get("size", 0),
                                "size_formatted": format_file_size(s3_file.get("size", 0)),
                                "created_at": s3_file.get("last_modified", ""),
                                "type": "image",
                                "result_type": s3_file.get("result_type")
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

async def handle_delete_results(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Handle deletion of results by type across all processes in S3."""
    try:
        if not _storage_service:
            return {"error": "Storage service not initialized"}
        result_type = (job_input.get("result_type") or "").strip().lower()
        if result_type not in {"lora", "samples", "images"}:
            return {"error": "Invalid or missing result_type. Use one of: lora, samples, images"}

        result = await _storage_service.delete_results_by_type(result_type)
        if result.get("error"):
            return {"error": result.get("error"), "result_type": result_type}
        return {
            "status": "success",
            "result_type": result_type,
            **result
        }
    except Exception as e:
        log(f"❌ delete_results error: {e}", "ERROR")
        return {"error": f"Failed to delete results: {str(e)}"}

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
            # Read file and decide response strategy based on size
            max_inline_bytes = 5 * 1024 * 1024  # 5MB threshold to avoid RunPod payload limits
            file_size = os.path.getsize(file_path)
            
            if file_size > max_inline_bytes:
                # For large local files, return error suggesting S3-based download
                # (local direct base64 would exceed RunPod limits)
                log(f"↩️ Large file requested via local path ({format_file_size(file_size)}); use S3 presigned URL instead", "WARN")
                return {
                    "error": "File too large for inline transfer. Use S3 presigned URL via 'download_by_key' with the S3 key.",
                    "size": file_size
                }
            
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

async def handle_download_by_key(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a presigned URL for a given S3 key directly (without process validation)."""
    try:
        s3_key = job_input.get("s3_key")
        if not s3_key:
            return {"error": "Missing 's3_key' parameter"}
        if not _storage_service:
            return {"error": "Storage service not initialized"}

        # Try to derive filename and size
        size = 0
        try:
            head = _storage_service.s3_client.head_object(Bucket=_storage_service.bucket_name, Key=s3_key)
            size = head.get('ContentLength', 0)
        except Exception:
            pass

        download_data = await _storage_service.get_download_url(process_id=None, s3_key=s3_key)
        if download_data:
            log(f"✅ Presigned URL generated for key: {s3_key}", "INFO")
            # Ensure size populated if backend couldn't resolve from get_download_url
            if 'size' not in download_data or not download_data['size']:
                download_data['size'] = size
            return download_data
        else:
            return {"error": "Failed to generate download URL"}
    except Exception as e:
        log(f"❌ download_by_key error: {e}", "ERROR")
        return {"error": f"Failed to generate URL: {str(e)}"}

# Start RunPod Serverless
if __name__ == "__main__":
    log("🚀 Starting LoRA Dashboard RunPod Serverless Handler", "INFO")
    runpod.serverless.start({
        "handler": async_handler  # ✅ Direct async handler
    })