#!/usr/bin/env python3
"""
🛠️ Utility Functions for Backend Operations
GPU management, metrics, file operations, and more
"""

import os
import time
import subprocess
import psutil
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Metrics storage
METRICS = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0, "last_error": None})

# Configuration
TRAINING_TIMEOUT = int(os.environ.get('TRAINING_TIMEOUT', 7200))  # 2 hours default
GENERATION_TIMEOUT = int(os.environ.get('GENERATION_TIMEOUT', 600))  # 10 minutes default
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
FILE_RETENTION_DAYS = int(os.environ.get('FILE_RETENTION_DAYS', 7))
GPU_MEMORY_THRESHOLD = 0.9  # 90% threshold for GPU memory

def log(message: str, level: str = "INFO"):
    """Unified logging to stdout for RunPod visibility"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {message}"
    
    # Write to stdout for RunPod visibility
    print(log_msg)
    import sys
    sys.stdout.flush()
    
    # Also use standard logger
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)

def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def track_metric(operation: str):
    """Decorator to track operation metrics"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                METRICS[operation]["count"] += 1
                METRICS[operation]["total_time"] += duration
                METRICS[operation]["avg_time"] = METRICS[operation]["total_time"] / METRICS[operation]["count"]
                log(f"📊 {operation} completed in {duration:.3f}s", "INFO")
                return result
            except Exception as e:
                METRICS[operation]["errors"] += 1
                METRICS[operation]["last_error"] = str(e)
                log(f"❌ {operation} failed: {e}", "ERROR")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                METRICS[operation]["count"] += 1
                METRICS[operation]["total_time"] += duration
                METRICS[operation]["avg_time"] = METRICS[operation]["total_time"] / METRICS[operation]["count"]
                log(f"📊 {operation} completed in {duration:.3f}s", "INFO")
                return result
            except Exception as e:
                METRICS[operation]["errors"] += 1
                METRICS[operation]["last_error"] = str(e)
                log(f"❌ {operation} failed: {e}", "ERROR")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def get_metrics() -> Dict[str, Any]:
    """Get current metrics summary"""
    return dict(METRICS)

# Lazy torch import helper
_TORCH_REF = None

def _get_torch():
    global _TORCH_REF
    if _TORCH_REF is None:
        try:
            import torch as _torch_mod
            _TORCH_REF = _torch_mod
        except Exception:
            _TORCH_REF = None
    return _TORCH_REF

# GPU Management Functions
class GPUManager:
    """Enhanced GPU management with cleanup and monitoring"""
    
    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory"""
        try:
            torch = _get_torch()
            if torch and torch.cuda.is_available():
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                log("✅ GPU memory cleaned", "INFO")
                return True
        except Exception as e:
            log(f"⚠️ GPU cleanup failed: {e}", "WARN")
            return False
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """Get current GPU memory usage"""
        try:
            torch = _get_torch()
            if not torch or not torch.cuda.is_available():
                return {"available": False}
            
            # Get memory info for all GPUs
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                free = total - reserved
                
                gpu_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mb": total // (1024 * 1024),
                    "allocated_mb": allocated // (1024 * 1024),
                    "reserved_mb": reserved // (1024 * 1024),
                    "free_mb": free // (1024 * 1024),
                    "usage_percent": (reserved / total) * 100
                })
            
            return {
                "available": True,
                "gpu_count": torch.cuda.device_count(),
                "gpus": gpu_info
            }
        except Exception as e:
            log(f"❌ GPU info error: {e}", "ERROR")
            return {"available": False, "error": str(e)}
    
    @staticmethod
    def check_gpu_memory_available(required_mb: int = 4096) -> bool:
        """Check if enough GPU memory is available"""
        try:
            info = GPUManager.get_gpu_memory_info()
            if not info.get("available"):
                return False
            
            # Check if any GPU has enough free memory
            for gpu in info.get("gpus", []):
                if gpu["free_mb"] >= required_mb:
                    return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def kill_gpu_processes(force: bool = False):
        """Kill processes using GPU memory"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return False
            
            killed = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    pid, memory = line.split(', ')
                    pid = int(pid)
                    memory = int(memory)
                    
                    # Only kill if using significant memory or force flag
                    if memory > 1000 or force:  # > 1GB
                        process = psutil.Process(pid)
                        process.terminate()
                        killed.append(pid)
                        log(f"Terminated GPU process {pid} using {memory}MB", "WARN")
                except Exception:
                    continue
            
            # Wait for termination
            if killed:
                time.sleep(2)
                GPUManager.cleanup_gpu_memory()
            
            return True
        except Exception as e:
            log(f"❌ Failed to kill GPU processes: {e}", "ERROR")
            return False

# Retry logic with exponential backoff
async def retry_with_backoff(
    func: Callable,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
) -> Any:
    """Retry function with exponential backoff"""
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Call function and await if it returns an awaitable (coroutine, Task, or Future)
            result = func()
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                log(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...", "WARN")
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                log(f"❌ All {max_retries} attempts failed", "ERROR")
    raise last_error

# File cleanup utilities
async def cleanup_old_files(base_path: str = "/workspace", retention_days: int = FILE_RETENTION_DAYS):
    """Clean up files older than retention period"""
    try:
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        cleaned_size = 0
        
        for root, dirs, files in os.walk(base_path):
            # Skip important directories
            if any(skip in root for skip in ['.git', 'ai-toolkit', 'models/base']):
                continue
            
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    stat = os.stat(file_path)
                    
                    # Check if file is older than cutoff
                    if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                        cleaned_size += stat.st_size
                        os.remove(file_path)
                        cleaned_count += 1
                except Exception as e:
                    log(f"Failed to clean {file}: {e}", "WARN")
                    continue
        
        if cleaned_count > 0:
            log(f"🧹 Cleaned {cleaned_count} files ({format_file_size(cleaned_size)})", "INFO")
        
        return cleaned_count, cleaned_size
    except Exception as e:
        log(f"❌ Cleanup failed: {e}", "ERROR")
        return 0, 0

# Process execution with timeout and retries
async def execute_with_timeout(
    cmd: List[str],
    timeout: int = TRAINING_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Execute command with timeout and proper error handling"""
    
    def run_command():
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy(),
            cwd=cwd
        )
    
    try:
        result = await retry_with_backoff(
            lambda: asyncio.get_event_loop().run_in_executor(None, run_command),
            max_retries=1  # Don't retry long-running commands
        )
        
        if result.returncode != 0:
            error_msg = f"Command failed with code {result.returncode}: {result.stderr}"
            log(error_msg, "ERROR")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        return result
    except subprocess.TimeoutExpired:
        log(f"⏱️ Command timed out after {timeout}s: {' '.join(cmd)}", "ERROR")
        raise
    except Exception as e:
        log(f"❌ Command execution failed: {e}", "ERROR")
        raise

def execute_with_streaming(
    cmd: List[str],
    timeout: int = TRAINING_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Execute command with real-time output streaming for training visibility"""
    
    log(f"🚀 Starting streaming command: {' '.join(cmd)}", "INFO")
    
    try:
        # Try streaming approach first
        return _execute_with_streaming_internal(cmd, timeout, env, cwd)
    except Exception as streaming_error:
        log(f"⚠️ Streaming execution failed: {streaming_error}, falling back to standard mode", "WARNING")
        # Fallback to standard execution with enhanced logging
        return _execute_with_fallback(cmd, timeout, env, cwd)

def _execute_with_streaming_internal(
    cmd: List[str],
    timeout: int,
    env: Optional[Dict[str, str]],
    cwd: Optional[str]
) -> subprocess.CompletedProcess:
    """Internal streaming execution with threading"""
    
    # Start process with streaming stdout/stderr
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr to stdout for simpler handling
        text=True,
        bufsize=0,  # Unbuffered for real-time output
        universal_newlines=True,
        env=env or os.environ.copy(),
        cwd=cwd
    )
    
    output_lines = []
    
    def read_and_log_output():
        """Read and log output in real-time"""
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                    
                line_clean = line.rstrip('\n\r')
                if line_clean:  # Skip empty lines
                    output_lines.append(line_clean)
                    
                    # Enhanced pattern matching for training info
                    line_lower = line_clean.lower()
                    if any(keyword in line_lower for keyword in [
                        'step', 'loss', 'lr', 'epoch', 'learning_rate', 
                        'batch', 'iteration', 'training', 'gradient'
                    ]):
                        log(f"🔥 TRAINING: {line_clean}", "INFO")
                    elif any(keyword in line_lower for keyword in [
                        'error', 'exception', 'failed', 'traceback'
                    ]):
                        log(f"❌ ERROR: {line_clean}", "ERROR")
                    elif any(keyword in line_lower for keyword in [
                        'warning', 'warn'
                    ]):
                        log(f"⚠️ WARNING: {line_clean}", "WARNING")
                    else:
                        log(f"📋 OUTPUT: {line_clean}", "INFO")
        except Exception as e:
            log(f"⚠️ Error in output reading: {e}", "WARNING")
        finally:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
    
    # Start reading thread
    import threading
    output_thread = threading.Thread(target=read_and_log_output, daemon=True)
    output_thread.start()
    
    # Wait for process completion
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"⏱️ Process timed out after {timeout}s, terminating", "ERROR")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log("🔪 Force killing process", "ERROR")
            process.kill()
            process.wait()
        raise subprocess.TimeoutExpired(cmd, timeout)
    
    # Wait for output thread to finish
    output_thread.join(timeout=5)
    
    # Create result
    result = subprocess.CompletedProcess(
        cmd, 
        return_code, 
        stdout='\n'.join(output_lines),
        stderr=''  # We merged stderr to stdout
    )
    
    if return_code != 0:
        log(f"❌ Process failed with return code {return_code}", "ERROR")
        raise subprocess.CalledProcessError(return_code, cmd, result.stdout, result.stderr)
    
    log(f"✅ Process completed successfully", "INFO")
    return result

def _execute_with_fallback(
    cmd: List[str],
    timeout: int,
    env: Optional[Dict[str, str]],
    cwd: Optional[str]
) -> subprocess.CompletedProcess:
    """Fallback execution with standard subprocess.run and enhanced logging"""
    
    log(f"🔄 Using fallback execution mode", "INFO")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
        cwd=cwd
    )
    
    # Enhanced logging of captured output
    if result.stdout:
        log(f"📋 STDOUT CAPTURED:", "INFO")
        for line in result.stdout.split('\n'):
            if line.strip():
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'step', 'loss', 'lr', 'epoch', 'learning_rate'
                ]):
                    log(f"🔥 TRAINING: {line}", "INFO")
                else:
                    log(f"📋 {line}", "INFO")
    
    if result.stderr:
        log(f"📟 STDERR CAPTURED:", "INFO")
        for line in result.stderr.split('\n'):
            if line.strip():
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in [
                    'error', 'exception', 'failed'
                ]):
                    log(f"❌ ERROR: {line}", "ERROR")
                elif any(keyword in line_lower for keyword in [
                    'warning', 'warn'
                ]):
                    log(f"⚠️ WARNING: {line}", "WARNING")
                else:
                    log(f"📟 {line}", "INFO")
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    return result

# Path validation and normalization
def normalize_workspace_path(path: str) -> str:
    """Normalize and validate workspace paths"""
    # Handle S3 paths
    if path.startswith('s3://'):
        return path
    
    # Normalize local paths
    path = os.path.normpath(path)
    
    # Ensure it's within workspace
    if not path.startswith('/workspace'):
        if path.startswith('workspace'):
            path = '/' + path
        else:
            path = os.path.join('/workspace', path.lstrip('/'))
    
    # Prevent path traversal
    if '..' in path:
        raise ValueError("Path traversal not allowed")
    
    return path

# Circuit breaker implementation
class CircuitBreaker:
    """Simple circuit breaker for external services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        # Check if circuit should be reset
        if self.is_open and self.last_failure_time:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
                log("🔌 Circuit breaker reset", "INFO")
        
        # If circuit is open, fail fast
        if self.is_open:
            raise Exception("Circuit breaker is open")
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Reset on success
            self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                log(f"⚡ Circuit breaker opened after {self.failure_count} failures", "WARN")
            
            raise

    # Optional synchronous helper for non-async code paths
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute synchronous function with circuit breaker protection"""
        # Reset check
        if self.is_open and self.last_failure_time:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
                log("🔌 Circuit breaker reset", "INFO")
        if self.is_open:
            raise Exception("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                log(f"⚡ Circuit breaker opened after {self.failure_count} failures", "WARN")
            raise

# Batch operations helper
async def batch_process(items: List[Any], processor: Callable, batch_size: int = 10) -> List[Any]:
    """Process items in batches for better performance"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        if asyncio.iscoroutinefunction(processor):
            batch_results = await asyncio.gather(*[processor(item) for item in batch])
        else:
            batch_results = [processor(item) for item in batch]
        
        results.extend(batch_results)
        
        # Small delay between batches to avoid overwhelming system
        if i + batch_size < len(items):
            await asyncio.sleep(0.1)
    
    return results

# Memory usage monitoring
def get_memory_usage() -> Dict[str, Any]:
    """Get current system memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total // (1024 * 1024),
            "available_mb": memory.available // (1024 * 1024),
            "used_mb": memory.used // (1024 * 1024),
            "percent": memory.percent
        }
    except Exception as e:
        log(f"Failed to get memory info: {e}", "WARN")
        return {}

# Environment validation
def validate_environment() -> Dict[str, bool]:
    """Validate environment setup"""
    # Use lazy torch loader to avoid false negatives when torch isn't imported yet
    torch = _get_torch()
    checks = {
        "workspace_exists": os.path.exists("/workspace"),
        "gpu_available": torch.cuda.is_available() if torch else False,
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "training_data_dir": os.path.exists("/workspace/training_data"),
        "models_dir": os.path.exists("/workspace/models"),
        "output_dir": os.path.exists("/workspace/output"),
        "ai_toolkit": os.path.exists("/workspace/ai-toolkit"),
    }
    
    # Log any missing requirements
    for check, passed in checks.items():
        if not passed:
            log(f"⚠️ Environment check failed: {check}", "WARN")
    
    return checks

# Safe JSON serialization for process data
def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects for JSON"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)
