#!/usr/bin/env python3
"""
☁️ Enhanced S3 Storage Utilities
Batch operations, retry logic, and optimized S3 handling
"""

import os
import asyncio
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import base64
from functools import lru_cache
import concurrent.futures

from utils import log, format_file_size, retry_with_backoff, CircuitBreaker

def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val not in (None, "") else default

# S3 Configuration (parameterized via ENV, with safe defaults)
S3_CONFIG = {
    "bucket_name": _env("S3_BUCKET", "tqv92ffpc5"),
    "endpoint_url": _env("S3_ENDPOINT_URL", "https://s3api-eu-ro-1.runpod.io"),
    "region": _env("S3_REGION", "eu-ro-1"),
    "prefix": _env("S3_PREFIX", "lora-dashboard"),
    "max_retries": int(_env("S3_MAX_RETRIES", "3")),
    "multipart_threshold": int(_env("S3_MULTIPART_THRESHOLD", str(50 * 1024 * 1024))),  # 50MB
    "multipart_chunksize": int(_env("S3_MULTIPART_CHUNKSIZE", str(10 * 1024 * 1024))),  # 10MB chunks
    "max_concurrent_uploads": int(_env("S3_MAX_CONCURRENCY", "5")),
}

class S3StorageManager:
    """Enhanced S3 storage manager with batch operations and optimization"""
    
    def __init__(self):
        self.config = S3_CONFIG
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._client = None
        self._transfer_config = None

    def _s3_call(self, func, *args, **kwargs):
        """Execute S3 SDK call protected by circuit breaker (synchronous)."""
        try:
            # Prefer breaker if available
            if self.circuit_breaker and hasattr(self.circuit_breaker, 'call_sync'):
                return self.circuit_breaker.call_sync(func, *args, **kwargs)
            return func(*args, **kwargs)
        except Exception:
            # Re-raise to let callers handle/log
            raise
        
    @property
    def s3_client(self):
        """Lazy initialization of S3 client"""
        if not self._client:
            try:
                self._client = boto3.client(
                    's3',
                    endpoint_url=self.config['endpoint_url'],
                    region_name=self.config['region'],
                    config=BotoConfig(
                        s3={'addressing_style': 'path'},
                        max_pool_connections=50,
                        retries={'max_attempts': self.config['max_retries']},
                    )
                )
                log("✅ S3 client initialized", "INFO")
            except Exception as e:
                log(f"❌ Failed to initialize S3 client: {e}", "ERROR")
                raise
        return self._client
    
    @property
    def transfer_config(self):
        """Get transfer configuration for large files"""
        if not self._transfer_config:
            from boto3.s3.transfer import TransferConfig
            self._transfer_config = TransferConfig(
                multipart_threshold=self.config['multipart_threshold'],
                multipart_chunksize=self.config['multipart_chunksize'],
                max_concurrency=self.config['max_concurrent_uploads'],
                use_threads=True
            )
        return self._transfer_config
    
    async def batch_upload_files(
        self, 
        files: List[Dict[str, Any]], 
        s3_prefix: str,
        batch_size: int = 10
    ) -> List[str]:
        """Upload files in batches for better performance"""
        uploaded_keys = []
        total_files = len(files)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_concurrent_uploads']) as executor:
            for batch_start in range(0, total_files, batch_size):
                batch_end = min(batch_start + batch_size, total_files)
                batch = files[batch_start:batch_end]
                
                log(f"📦 Uploading batch {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}", "INFO")
                
                # Submit batch uploads
                futures = []
                for file_info in batch:
                    future = executor.submit(
                        self._upload_single_file,
                        file_info,
                        s3_prefix
                    )
                    futures.append(future)
                
                # Wait for batch completion
                for future in concurrent.futures.as_completed(futures):
                    try:
                        s3_key = future.result()
                        if s3_key:
                            uploaded_keys.append(s3_key)
                    except Exception as e:
                        log(f"❌ Batch upload error: {e}", "ERROR")
                
                # Small delay between batches
                if batch_end < total_files:
                    await asyncio.sleep(0.1)
        
        return uploaded_keys
    
    def _upload_single_file(self, file_info: Dict[str, Any], s3_prefix: str) -> Optional[str]:
        """Upload single file to S3 (sync)"""
        try:
            filename = file_info.get("filename")
            content = file_info.get("content")  # base64
            
            if not filename or not content:
                return None
            
            # Decode content
            file_data = base64.b64decode(content)
            s3_key = f"{s3_prefix}/{filename}"
            
            # Use multipart upload for large files
            if len(file_data) > self.config['multipart_threshold']:
                log(f"📤 Multipart upload: {filename} ({format_file_size(len(file_data))})", "INFO")
                
                # Write to temp file for multipart
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_data)
                    tmp_file_path = tmp_file.name
                
                try:
                    self._s3_call(
                        self.s3_client.upload_file,
                        Filename=tmp_file_path,
                        Bucket=self.config['bucket_name'],
                        Key=s3_key,
                        Config=self.transfer_config
                    )
                finally:
                    os.unlink(tmp_file_path)
            else:
                # Regular upload for smaller files
                # Attempt to infer content type
                content_type = file_info.get("content_type") or "application/octet-stream"
                self._s3_call(
                    self.s3_client.put_object,
                    Bucket=self.config['bucket_name'],
                    Key=s3_key,
                    Body=file_data,
                    ContentType=content_type,
                )
            
            log(f"✅ Uploaded: {filename}", "INFO")
            return s3_key
            
        except Exception as e:
            log(f"❌ Failed to upload {filename}: {e}", "ERROR")
            return None
    
    async def download_directory(self, s3_prefix: str, local_path: str) -> List[str]:
        """Download all files from S3 directory"""
        try:
            os.makedirs(local_path, exist_ok=True)
            downloaded_files = []
            
            # List objects
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config['bucket_name'],
                Prefix=s3_prefix
            )
            
            # Download files concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_concurrent_uploads']) as executor:
                futures = []
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        filename = os.path.basename(s3_key)
                        local_file_path = os.path.join(local_path, filename)
                        
                        future = executor.submit(
                            self._download_single_file,
                            s3_key,
                            local_file_path
                        )
                        futures.append((future, local_file_path))
                
                # Collect results
                for future, local_file_path in futures:
                    try:
                        if future.result():
                            downloaded_files.append(local_file_path)
                    except Exception as e:
                        log(f"❌ Download error: {e}", "ERROR")
            
            return downloaded_files
            
        except Exception as e:
            log(f"❌ Directory download failed: {e}", "ERROR")
            return []
    
    def _download_single_file(self, s3_key: str, local_path: str) -> bool:
        """Download single file from S3 (sync)"""
        try:
            # Get object size first
            response = self._s3_call(
                self.s3_client.head_object,
                Bucket=self.config['bucket_name'],
                Key=s3_key
            )
            file_size = response.get('ContentLength', 0)
            
            # Use multipart download for large files
            if file_size > self.config['multipart_threshold']:
                log(f"📥 Multipart download: {os.path.basename(s3_key)} ({format_file_size(file_size)})", "INFO")
                
                self._s3_call(
                    self.s3_client.download_file,
                    Bucket=self.config['bucket_name'],
                    Key=s3_key,
                    Filename=local_path,
                    Config=self.transfer_config
                )
            else:
                # Regular download
                response = self._s3_call(
                    self.s3_client.get_object,
                    Bucket=self.config['bucket_name'],
                    Key=s3_key
                )
                
                with open(local_path, 'wb') as f:
                    f.write(response['Body'].read())
            
            log(f"✅ Downloaded: {os.path.basename(local_path)}", "INFO")
            return True
            
        except Exception as e:
            log(f"❌ Failed to download {s3_key}: {e}", "ERROR")
            return False
    
    async def generate_presigned_urls(
        self, 
        s3_keys: List[str], 
        expiration: int = 3600
    ) -> Dict[str, str]:
        """Generate presigned URLs for multiple files"""
        urls = {}
        
        try:
            # Generate URLs concurrently
            loop = asyncio.get_event_loop()
            tasks = []
            
            for s3_key in s3_keys:
                task = loop.run_in_executor(
                    None,
                    self._generate_single_presigned_url,
                    s3_key,
                    expiration
                )
                tasks.append((s3_key, task))
            
            # Collect results
            for s3_key, task in tasks:
                try:
                    url = await task
                    if url:
                        urls[s3_key] = url
                except Exception as e:
                    log(f"❌ Failed to generate URL for {s3_key}: {e}", "ERROR")
            
            return urls
            
        except Exception as e:
            log(f"❌ Batch presigned URL generation failed: {e}", "ERROR")
            return urls
    
    def _generate_single_presigned_url(self, s3_key: str, expiration: int) -> Optional[str]:
        """Generate single presigned URL (sync)"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.config['bucket_name'],
                    'Key': s3_key,
                    # Allow nicer downloaded filename
                    'ResponseContentDisposition': f'attachment; filename="{os.path.basename(s3_key)}"'
                },
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            log(f"❌ Presigned URL generation failed for {s3_key}: {e}", "ERROR")
            return None
    
    async def list_files_with_metadata(
        self, 
        s3_prefix: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """List files with optional metadata"""
        files = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config['bucket_name'],
                Prefix=s3_prefix
            )
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    file_info = {
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'filename': os.path.basename(obj['Key'])
                    }
                    
                    # Get additional metadata if requested
                    if include_metadata:
                        try:
                            head_response = self._s3_call(
                                self.s3_client.head_object,
                                Bucket=self.config['bucket_name'],
                                Key=obj['Key']
                            )
                            file_info['metadata'] = head_response.get('Metadata', {})
                            file_info['content_type'] = head_response.get('ContentType', 'application/octet-stream')
                        except Exception as e:
                            log(f"⚠️ Failed to head object for metadata {obj['Key']}: {e}", "WARN")
                    
                    files.append(file_info)
            
            return files
            
        except Exception as e:
            log(f"❌ Failed to list files: {e}", "ERROR")
            return []
    
    async def delete_files(self, s3_keys: List[str]) -> Tuple[List[str], List[str]]:
        """Delete multiple files from S3"""
        succeeded = []
        failed = []
        
        try:
            # Delete in batches (S3 supports max 1000 per request)
            batch_size = 1000
            
            for i in range(0, len(s3_keys), batch_size):
                batch = s3_keys[i:i+batch_size]
                
                delete_objects = [{'Key': key} for key in batch]
                
                response = self._s3_call(
                    self.s3_client.delete_objects,
                    Bucket=self.config['bucket_name'],
                    Delete={'Objects': delete_objects}
                )
                
                # Process results
                for deleted in response.get('Deleted', []):
                    succeeded.append(deleted['Key'])
                
                for error in response.get('Errors', []):
                    failed.append(error['Key'])
                    log(f"❌ Delete failed for {error['Key']}: {error['Message']}", "ERROR")
            
            return succeeded, failed
            
        except Exception as e:
            log(f"❌ Batch delete failed: {e}", "ERROR")
            return succeeded, failed + s3_keys[len(succeeded):]
    
    @lru_cache(maxsize=100)
    def check_file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3 (cached)"""
        try:
            self._s3_call(
                self.s3_client.head_object,
                Bucket=self.config['bucket_name'],
                Key=s3_key
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
