"""
🚀 UNIFIED LOGGING FOR RUNPOD VISIBILITY
Simplified logging with stdout/stderr output for RunPod compatibility
Based on runpod-fastbackend/ successful approach
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

def log(message, level="INFO"):
    """Unified logging to stdout and stderr for RunPod visibility"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {message}"
    
    # Write to both stdout and stderr for maximum visibility
    print(log_msg)
    sys.stderr.write(f"{log_msg}\n")
    sys.stderr.flush()
    sys.stdout.flush()

class RequestResponseLogger:
    """Enhanced logger for request/response debugging with RunPod compatibility"""
    
    def __init__(self, log_dir: str = "/workspace/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handlers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup different loggers for different purposes"""
        
        # Main application logger
        self.app_logger = logging.getLogger("lora_app")
        self.app_logger.setLevel(logging.INFO)
        
        # Request/Response logger
        self.req_resp_logger = logging.getLogger("lora_requests")
        self.req_resp_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger("lora_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        # Clear existing handlers
        for logger in [self.app_logger, self.req_resp_logger, self.error_logger]:
            logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        json_formatter = logging.Formatter('%(message)s')
        
        # File handlers
        app_handler = logging.FileHandler(self.log_dir / "app.log")
        app_handler.setFormatter(detailed_formatter)
        self.app_logger.addHandler(app_handler)
        
        req_resp_handler = logging.FileHandler(self.log_dir / "requests.log")
        req_resp_handler.setFormatter(json_formatter)
        self.req_resp_logger.addHandler(req_resp_handler)
        
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        self.app_logger.addHandler(console_handler)
    
    def log_request(self, 
                   request_type: str,
                   request_data: Dict[str, Any],
                   endpoint: str = "",
                   user_id: Optional[str] = None):
        """Log incoming request"""
        
        request_id = self._generate_request_id()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "REQUEST",
            "request_type": request_type,
            "endpoint": endpoint,
            "user_id": user_id,
            "data": self._sanitize_data(request_data),
            "request_id": request_id
        }
        
        self.req_resp_logger.info(json.dumps(log_entry, indent=2))
        self.app_logger.info(f"📥 REQUEST | {request_type} | {endpoint} | ID: {request_id}")
        
        # Also use unified logging for RunPod visibility
        log(f"📥 REQUEST | {request_type} | {endpoint} | ID: {request_id}", "INFO")
        
        return request_id
    
    def log_response(self,
                    request_id: str,
                    response_data: Dict[str, Any],
                    status_code: int = 200,
                    error: Optional[str] = None):
        """Log outgoing response"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "RESPONSE",
            "request_id": request_id,
            "status_code": status_code,
            "error": error,
            "data": self._sanitize_data(response_data),
            "success": error is None
        }
        
        self.req_resp_logger.info(json.dumps(log_entry, indent=2))
        
        if error:
            self.app_logger.error(f"❌ RESPONSE | ID: {request_id} | Error: {error}")
            self.error_logger.error(f"Request ID: {request_id} | Error: {error}\n{traceback.format_exc()}")
            # Also use unified logging for RunPod visibility
            log(f"❌ RESPONSE | ID: {request_id} | Error: {error}", "ERROR")
        else:
            self.app_logger.info(f"✅ RESPONSE | ID: {request_id} | Status: {status_code}")
            # Also use unified logging for RunPod visibility
            log(f"✅ RESPONSE | ID: {request_id} | Status: {status_code}", "INFO")
    
    def log_file_operation(self,
                          operation: str,
                          file_info: Dict[str, Any],
                          request_id: Optional[str] = None):
        """Log file upload/download operations"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "FILE_OPERATION",
            "operation": operation,
            "request_id": request_id,
            "file_info": self._sanitize_data(file_info)
        }
        
        self.req_resp_logger.info(json.dumps(log_entry, indent=2))
        self.app_logger.info(f"📁 FILE | {operation} | {file_info.get('filename', 'unknown')} | ID: {request_id}")
    
    def log_error(self, 
                 error: Exception,
                 context: Dict[str, Any] = None,
                 request_id: Optional[str] = None):
        """Log detailed error information"""
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "ERROR",
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": self._sanitize_data(context or {})
        }
        
        self.req_resp_logger.info(json.dumps(error_entry, indent=2))
        self.error_logger.error(f"Request ID: {request_id} | {type(error).__name__}: {str(error)}")
        self.app_logger.error(f"💥 ERROR | {type(error).__name__} | ID: {request_id}")
        
        # Also use unified logging for RunPod visibility
        log(f"💥 ERROR | {type(error).__name__} | ID: {request_id}: {str(error)}", "ERROR")
    
    def _sanitize_data(self, data: Any) -> Any:
        """Remove sensitive information from log data"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                    sanitized[key] = "***HIDDEN***"
                elif key.lower() == 'content' and isinstance(value, str) and len(value) > 100:
                    # Truncate large content (like base64 files)
                    sanitized[key] = f"{value[:100]}... [TRUNCATED - {len(value)} chars]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        from uuid import uuid4
        return str(uuid4())[:8]
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "log_directory": str(self.log_dir),
            "files": []
        }
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.exists():
                stats["files"].append({
                    "name": log_file.name,
                    "size": log_file.stat().st_size,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
        
        return stats

# Global logger instance
request_response_logger = RequestResponseLogger()

def get_logger() -> RequestResponseLogger:
    """Get the global logger instance"""
    return request_response_logger 