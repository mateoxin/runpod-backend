"""
Storage Service
S3-compatible storage integration for RunPod Storage
"""

import asyncio
import logging
import os
from typing import Optional, List
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file storage using S3-compatible API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.s3_client = None
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize S3 client."""
        try:
            # Configure S3 client for RunPod Storage
            config = Config(
                region_name=self.settings.s3_region,
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50
            )
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                config=config
            )
            
            logger.info("S3 client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
            
    async def health_check(self) -> str:
        """Check storage service health."""
        if not self.s3_client:
            return "unhealthy - no client"
            
        try:
            # Try to list buckets (minimal operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self.s3_client.head_bucket, 
                {'Bucket': self.settings.s3_bucket}
            )
            return "healthy"
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return "unhealthy - bucket not found"
            else:
                return f"unhealthy - {e.response['Error']['Code']}"
        except Exception as e:
            return f"unhealthy - {str(e)}"
            
    async def upload_file(
        self, 
        local_path: str, 
        s3_key: str,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """Upload a single file to S3."""
        if not self.s3_client or not os.path.exists(local_path):
            return None
            
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.upload_file,
                local_path,
                self.settings.s3_bucket,
                s3_key,
                extra_args
            )
            
            logger.info(f"Uploaded {local_path} to s3://{self.settings.s3_bucket}/{s3_key}")
            return f"s3://{self.settings.s3_bucket}/{s3_key}"
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return None
            
    async def upload_directory(
        self, 
        local_path: str, 
        s3_key: str
    ) -> Optional[str]:
        """Upload entire directory to S3."""
        if not self.s3_client or not os.path.exists(local_path):
            return None
            
        try:
            upload_tasks = []
            
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_file_key = f"{s3_key.rstrip('/')}/{relative_path}".replace("\\", "/")
                    
                    # Determine content type
                    content_type = self._get_content_type(file)
                    
                    upload_tasks.append(
                        self.upload_file(local_file_path, s3_file_key, content_type)
                    )
                    
            # Upload all files concurrently
            await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            logger.info(f"Uploaded directory {local_path} to s3://{self.settings.s3_bucket}/{s3_key}")
            return f"s3://{self.settings.s3_bucket}/{s3_key}"
            
        except Exception as e:
            logger.error(f"Failed to upload directory {local_path}: {e}")
            return None
            
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension."""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.safetensors': 'application/octet-stream',
            '.ckpt': 'application/octet-stream',
            '.pt': 'application/octet-stream',
            '.pth': 'application/octet-stream',
        }
        return content_types.get(ext, 'application/octet-stream')
        
    async def get_download_url(
        self, 
        process_id: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """Generate presigned download URL."""
        if not self.s3_client:
            return None
            
        try:
            # Check if results exist for this process
            s3_key = f"results/{process_id}/"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.s3_client.list_objects_v2,
                {
                    'Bucket': self.settings.s3_bucket,
                    'Prefix': s3_key,
                    'MaxKeys': 1
                }
            )
            
            if not response.get('Contents'):
                logger.warning(f"No results found for process {process_id}")
                return None
                
            # Create presigned URL for the first file (or create a zip)
            first_file = response['Contents'][0]
            file_key = first_file['Key']
            
            url = await loop.run_in_executor(
                None,
                self.s3_client.generate_presigned_url,
                'get_object',
                {'Bucket': self.settings.s3_bucket, 'Key': file_key},
                expiration
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate download URL for {process_id}: {e}")
            return None
            
    async def list_files(self, prefix: str = "") -> List[dict]:
        """List files in storage."""
        if not self.s3_client:
            return []
            
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.s3_client.list_objects_v2,
                {
                    'Bucket': self.settings.s3_bucket,
                    'Prefix': prefix
                }
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
                
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
            
    async def delete_file(self, s3_key: str) -> bool:
        """Delete a file from storage."""
        if not self.s3_client:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.delete_object,
                {
                    'Bucket': self.settings.s3_bucket,
                    'Key': s3_key
                }
            )
            
            logger.info(f"Deleted {s3_key} from storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False 