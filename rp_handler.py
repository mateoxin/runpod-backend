#!/usr/bin/env python3
"""
🚀 MINIMAL RUNPOD HANDLER - WITH WORKING LOGGING
Copied logging pattern from working runpod-fastbackend
"""

import runpod
import sys
import time
from datetime import datetime

def log(message, level="INFO"):
    """Unified logging to stdout and stderr for RunPod visibility - COPIED FROM WORKING runpod-fastbackend"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {level}: {message}"
    
    # Write to both stdout and stderr for maximum visibility
    print(log_msg)
    sys.stderr.write(f"{log_msg}\n")
    sys.stderr.flush()
    sys.stdout.flush()

def handler(job):
    """Handler with working logging pattern - COPIED FROM runpod-fastbackend"""
    try:
        log(f"🎯 Received job: {job}", "INFO")
        
        job_input = job["input"]
        job_type = job_input.get("type", "unknown")
        
        log(f"📦 Processing job type: {job_type}", "INFO")
        
        if job_type == "health" or job_type == "health_check":
            result = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "message": "Minimal handler working - WITH LOGGING!",
                "job_id": job.get('id', 'unknown')
            }
            log(f"✅ Health check completed", "INFO")
            return result
        
        log(f"🔄 Processing {job_type} request...", "INFO")
        result = {
            "status": "completed", 
            "job_type": job_type,
            "message": f"Processed {job_type} successfully",
            "timestamp": datetime.now().isoformat(),
            "job_id": job.get('id', 'unknown')
        }
        log(f"✅ Request completed", "INFO")
        return result
        
    except Exception as e:
        log(f"💥 Handler error: {e}", "ERROR")
        return {"error": str(e), "job_id": job.get('id', 'unknown')}

if __name__ == "__main__":
    log("🚀 Starting MINIMAL RunPod Handler - WITH WORKING LOGGING!", "INFO")
    runpod.serverless.start({"handler": handler})