"""
Process Manager Service
Manages training and generation processes with Redis queue
"""

import asyncio
import json
import logging
import os
import subprocess
import uuid
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import redis.asyncio as redis
from app.core.models import Process, ProcessStatus, ProcessType
from app.services.gpu_manager import GPUManager
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages training and generation processes."""
    
    def __init__(
        self, 
        gpu_manager: GPUManager,
        storage_service: StorageService,
        redis_url: str = "redis://localhost:6379/0"
    ):
        self.gpu_manager = gpu_manager
        self.storage_service = storage_service
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.processes: Dict[str, Process] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize process manager."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Load existing processes from Redis
            await self._load_processes_from_redis()
            
            # Start background task processor
            asyncio.create_task(self._process_queue())
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessManager: {e}")
            # Continue without Redis if it's not available
            self.redis_client = None
            
    async def _load_processes_from_redis(self) -> None:
        """Load existing processes from Redis."""
        if not self.redis_client:
            return
            
        try:
            process_keys = await self.redis_client.keys("process:*")
            for key in process_keys:
                process_data = await self.redis_client.get(key)
                if process_data:
                    process_dict = json.loads(process_data)
                    process = Process(**process_dict)
                    self.processes[process.id] = process
                    
            logger.info(f"Loaded {len(self.processes)} processes from Redis")
            
        except Exception as e:
            logger.error(f"Failed to load processes from Redis: {e}")
            
    async def _save_process_to_redis(self, process: Process) -> None:
        """Save process to Redis."""
        if not self.redis_client:
            return
            
        try:
            process_data = process.model_dump(mode='json')
            await self.redis_client.set(
                f"process:{process.id}",
                json.dumps(process_data, default=str),
                ex=86400 * 7  # Expire in 7 days
            )
        except Exception as e:
            logger.error(f"Failed to save process to Redis: {e}")
            
    async def start_training(self, config_yaml: str) -> str:
        """Start a new training process."""
        try:
            # Parse YAML configuration
            config = yaml.safe_load(config_yaml)
            job_config = config.get('config', {})
            
            # Create process
            process_id = str(uuid.uuid4())
            process = Process(
                id=process_id,
                name=job_config.get('name', f'training_{process_id[:8]}'),
                type=ProcessType.TRAINING,
                status=ProcessStatus.PENDING,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                config=config
            )
            
            # Store process
            async with self._lock:
                self.processes[process_id] = process
                await self._save_process_to_redis(process)
                
            # Queue for execution
            await self._queue_process(process_id)
            
            logger.info(f"Training process {process_id} queued")
            return process_id
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            raise
            
    async def start_generation(self, config_yaml: str) -> str:
        """Start a new generation process."""
        try:
            # Parse YAML configuration
            config = yaml.safe_load(config_yaml)
            job_config = config.get('config', {})
            
            # Create process
            process_id = str(uuid.uuid4())
            process = Process(
                id=process_id,
                name=job_config.get('name', f'generation_{process_id[:8]}'),
                type=ProcessType.GENERATION,
                status=ProcessStatus.PENDING,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                config=config
            )
            
            # Store process
            async with self._lock:
                self.processes[process_id] = process
                await self._save_process_to_redis(process)
                
            # Queue for execution
            await self._queue_process(process_id)
            
            logger.info(f"Generation process {process_id} queued")
            return process_id
            
        except Exception as e:
            logger.error(f"Failed to start generation: {e}")
            raise
            
    async def _queue_process(self, process_id: str) -> None:
        """Queue process for execution."""
        if self.redis_client:
            try:
                await self.redis_client.lpush("process_queue", process_id)
            except Exception as e:
                logger.error(f"Failed to queue process {process_id}: {e}")
                # Fall back to direct execution
                asyncio.create_task(self._execute_process(process_id))
        else:
            # Direct execution if no Redis
            asyncio.create_task(self._execute_process(process_id))
            
    async def _process_queue(self) -> None:
        """Background task to process the queue."""
        while True:
            try:
                if self.redis_client:
                    # Get process from queue
                    result = await self.redis_client.brpop("process_queue", timeout=5)
                    if result:
                        _, process_id_bytes = result
                        process_id = process_id_bytes.decode('utf-8')
                        asyncio.create_task(self._execute_process(process_id))
                else:
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Error in process queue: {e}")
                await asyncio.sleep(5)
                
    async def _execute_process(self, process_id: str) -> None:
        """Execute a specific process."""
        try:
            async with self._lock:
                if process_id not in self.processes:
                    logger.error(f"Process {process_id} not found")
                    return
                    
                process = self.processes[process_id]
                
            # Allocate GPU
            gpu_id = await self.gpu_manager.allocate_gpu(process_id)
            if not gpu_id:
                # Update process status to pending and retry later
                await self._update_process_status(
                    process_id, 
                    ProcessStatus.PENDING,
                    error_message="No GPU available"
                )
                # Re-queue the process
                await asyncio.sleep(30)  # Wait 30 seconds before retry
                await self._queue_process(process_id)
                return
                
            # Update process status
            await self._update_process_status(
                process_id, 
                ProcessStatus.RUNNING,
                gpu_id=gpu_id
            )
            
            # Execute the actual job
            if process.type == ProcessType.TRAINING:
                await self._execute_training(process_id, gpu_id)
            else:
                await self._execute_generation(process_id, gpu_id)
                
        except Exception as e:
            logger.error(f"Failed to execute process {process_id}: {e}")
            await self._update_process_status(
                process_id,
                ProcessStatus.FAILED,
                error_message=str(e)
            )
        finally:
            # Release GPU
            await self.gpu_manager.release_gpu(process_id)
            # Remove from running tasks
            if process_id in self.running_tasks:
                del self.running_tasks[process_id]
                
    async def _execute_training(self, process_id: str, gpu_id: str) -> None:
        """Execute training job."""
        process = self.processes[process_id]
        
        # Create config file
        config_path = f"/tmp/training_config_{process_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(process.config, f)
            
        # Run AI toolkit training
        cmd = [
            "python3", 
            "/workspace/ai-toolkit/run.py", 
            config_path
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Execute training
        await self._run_subprocess(process_id, cmd, env)
        
    async def _execute_generation(self, process_id: str, gpu_id: str) -> None:
        """Execute generation job."""
        process = self.processes[process_id]
        
        # Create config file
        config_path = f"/tmp/generation_config_{process_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(process.config, f)
            
        # Run AI toolkit generation
        cmd = [
            "python3", 
            "/workspace/ai-toolkit/run.py", 
            config_path
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Execute generation
        await self._run_subprocess(process_id, cmd, env)
        
    async def _run_subprocess(self, process_id: str, cmd: List[str], env: Dict[str, str]) -> None:
        """Run subprocess and monitor progress."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            self.running_tasks[process_id] = asyncio.current_task()
            
            # Monitor output for progress
            async for line in process.stdout:
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    await self._parse_progress(process_id, line_str)
                    
            # Wait for completion
            return_code = await process.wait()
            
            if return_code == 0:
                # Upload results to storage
                output_path = await self._upload_results(process_id)
                await self._update_process_status(
                    process_id,
                    ProcessStatus.COMPLETED,
                    progress=100.0,
                    output_path=output_path
                )
            else:
                await self._update_process_status(
                    process_id,
                    ProcessStatus.FAILED,
                    error_message=f"Process exited with code {return_code}"
                )
                
        except asyncio.CancelledError:
            # Process was cancelled
            if process:
                process.terminate()
                await process.wait()
            await self._update_process_status(
                process_id,
                ProcessStatus.CANCELLED
            )
        except Exception as e:
            logger.error(f"Subprocess execution failed for {process_id}: {e}")
            await self._update_process_status(
                process_id,
                ProcessStatus.FAILED,
                error_message=str(e)
            )
            
    async def _parse_progress(self, process_id: str, line: str) -> None:
        """Parse progress from subprocess output."""
        try:
            # Look for step information
            if "Step:" in line or "step" in line.lower():
                # Try to extract step numbers
                parts = line.split()
                for i, part in enumerate(parts):
                    if "step" in part.lower() and i + 1 < len(parts):
                        try:
                            step_info = parts[i + 1]
                            if "/" in step_info:
                                current, total = step_info.split("/")
                                current_step = int(current)
                                total_steps = int(total)
                                progress = (current_step / total_steps) * 100
                                
                                await self._update_process_status(
                                    process_id,
                                    ProcessStatus.RUNNING,
                                    progress=progress,
                                    step=current_step,
                                    total_steps=total_steps
                                )
                                break
                        except ValueError:
                            continue
                            
            # Look for progress percentage
            if "%" in line:
                try:
                    # Extract percentage
                    for part in line.split():
                        if "%" in part:
                            progress_str = part.replace("%", "")
                            progress = float(progress_str)
                            await self._update_process_status(
                                process_id,
                                ProcessStatus.RUNNING,
                                progress=progress
                            )
                            break
                except ValueError:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to parse progress from line: {line}, error: {e}")
            
    async def _upload_results(self, process_id: str) -> Optional[str]:
        """Upload results to storage."""
        try:
            process = self.processes[process_id]
            
            # Determine output directory based on process type
            if process.type == ProcessType.TRAINING:
                output_dir = "/workspace/ai-toolkit/output"
            else:
                output_dir = "/workspace/samples_flux"
                
            # Upload to S3
            s3_path = await self.storage_service.upload_directory(
                local_path=output_dir,
                s3_key=f"results/{process_id}/"
            )
            
            return s3_path
            
        except Exception as e:
            logger.error(f"Failed to upload results for {process_id}: {e}")
            return None
            
    async def _update_process_status(
        self,
        process_id: str,
        status: ProcessStatus,
        progress: Optional[float] = None,
        gpu_id: Optional[str] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        eta: Optional[int] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> None:
        """Update process status."""
        async with self._lock:
            if process_id not in self.processes:
                return
                
            process = self.processes[process_id]
            process.status = status
            process.updated_at = datetime.now(timezone.utc)
            
            if progress is not None:
                process.progress = progress
            if gpu_id is not None:
                process.gpu_id = gpu_id
            if step is not None:
                process.step = step
            if total_steps is not None:
                process.total_steps = total_steps
            if eta is not None:
                process.eta = eta
            if error_message is not None:
                process.error_message = error_message
            if output_path is not None:
                process.output_path = output_path
                
            # Calculate elapsed time
            elapsed = datetime.now(timezone.utc) - process.created_at
            process.elapsed_time = int(elapsed.total_seconds())
            
            await self._save_process_to_redis(process)
            
    async def get_all_processes(self) -> List[Process]:
        """Get all processes."""
        async with self._lock:
            return list(self.processes.values())
            
    async def get_process(self, process_id: str) -> Optional[Process]:
        """Get specific process."""
        async with self._lock:
            return self.processes.get(process_id)
            
    async def cancel_process(self, process_id: str) -> bool:
        """Cancel a running process."""
        async with self._lock:
            if process_id not in self.processes:
                return False
                
            process = self.processes[process_id]
            if process.status not in [ProcessStatus.PENDING, ProcessStatus.RUNNING]:
                return False
                
            # Cancel running task
            if process_id in self.running_tasks:
                task = self.running_tasks[process_id]
                task.cancel()
                
            await self._update_process_status(process_id, ProcessStatus.CANCELLED)
            return True
            
    async def cleanup(self) -> None:
        """Cleanup process manager."""
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
            
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Process manager cleaned up") 