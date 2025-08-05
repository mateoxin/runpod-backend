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
    """ULTRA-MINIMAL HANDLER - ONLY HEALTH CHECK"""
    log(f"🎯 Received job: {job}", "INFO")
    
    job_input = job["input"]
    job_type = job_input.get("type", "unknown")
    
    log(f"📦 Processing job type: {job_type}", "INFO")
    
    # ONLY HEALTH CHECK - NOTHING ELSE
    if job_type == "health":
        log(f"✅ Health check completed", "INFO")
        return {
            "status": "healthy",
            "message": "MINIMAL HANDLER WORKS!",
            "timestamp": datetime.now().isoformat()
        }
    
    # Everything else returns error
    log(f"❌ Unsupported job type: {job_type}", "ERROR")
    return {
        "error": f"Only 'health' type supported, got: {job_type}"
    }

if __name__ == "__main__":
    log("🚀 Starting MINIMAL RunPod Handler - WITH WORKING LOGGING!", "INFO")
    runpod.serverless.start({"handler": handler})