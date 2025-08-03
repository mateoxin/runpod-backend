"""
Configuration settings for LoRA Dashboard Backend
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

# Import our config loader
try:
    from ..utils.config_loader import get_config_value, get_runpod_token
except ImportError:
    # Fallback for when utils module is not available
    def get_config_value(key: str, default: Optional[str] = None, config_path: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)
    
    def get_runpod_token(config_path: Optional[str] = None) -> str:
        token = os.getenv('RUNPOD_TOKEN')
        if not token:
            raise ValueError("RUNPOD_TOKEN not found in environment variables")
        return token


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    debug: bool = Field(default=False, description="Debug mode")
    mock_mode: bool = Field(default=False, description="Mock mode - use fake services for local testing")
    port: int = Field(default=8000, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    
    # GPU and Process Management
    max_concurrent_jobs: int = Field(default=10, description="Maximum concurrent GPU jobs")
    gpu_timeout: int = Field(default=14400, description="GPU job timeout in seconds (4 hours)")
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # S3 Storage Configuration (RunPod Storage)
    s3_endpoint_url: str = Field(
        default="https://storage.runpod.io",
        description="S3 endpoint URL"
    )
    s3_access_key: str = Field(description="S3 access key")
    s3_secret_key: str = Field(description="S3 secret key") 
    s3_bucket: str = Field(description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    
    # File Paths
    workspace_path: str = Field(
        default="/workspace",
        description="Workspace directory path"
    )
    models_path: str = Field(
        default="/workspace/models",
        description="Models directory path"
    )
    output_path: str = Field(
        default="/workspace/output",
        description="Output directory path"
    )
    
    # AI Toolkit Configuration
    ai_toolkit_path: str = Field(
        default="/workspace/ai-toolkit",
        description="AI Toolkit installation path"
    )
    python_executable: str = Field(
        default="python3",
        description="Python executable path"
    )
    
    # RunPod Configuration - Loaded from config file or environment
    @property
    def runpod_token(self) -> str:
        """Get RunPod API token from config file or environment."""
        try:
            return get_runpod_token()
        except ValueError as e:
            if self.mock_mode:
                return "mock_token_for_testing"
            raise e
    
    @property 
    def runpod_endpoint_id(self) -> Optional[str]:
        """Get RunPod Endpoint ID from config file or environment."""
        from ..utils.config_loader import get_runpod_endpoint_id
        return get_runpod_endpoint_id()

    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Allow reading from environment variables
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            # Override environment variable loading to support config files
            def config_file_settings(settings: BaseSettings) -> dict:
                """Load settings from config.env file."""
                config = {}
                
                # Use our config loader to get values
                config_keys = [
                    'DEBUG', 'MOCK_MODE', 'PORT', 'HOST',
                    'MAX_CONCURRENT_JOBS', 'GPU_TIMEOUT',
                    'REDIS_URL', 'S3_ENDPOINT_URL', 'S3_ACCESS_KEY',
                    'S3_SECRET_KEY', 'S3_BUCKET', 'S3_REGION',
                    'WORKSPACE_PATH', 'MODELS_PATH', 'OUTPUT_PATH',
                    'AI_TOOLKIT_PATH', 'PYTHON_EXECUTABLE'
                ]
                
                for key in config_keys:
                    value = get_config_value(key)
                    if value is not None:
                        config[key.lower()] = value
                
                return config
            
            return (
                init_settings,
                config_file_settings,
                env_settings, 
                file_secret_settings,
            )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings() 