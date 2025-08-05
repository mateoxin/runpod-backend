#!/usr/bin/env python3
"""
🚀 MINIMAL RUNPOD HANDLER - EXACTLY LIKE DOCUMENTATION
Test logging with absolute minimum code
"""

import runpod
import time

def handler(job):
    """Handler exactly like RunPod documentation examples"""
    print(f"🚀 Handler starting - job ID: {job.get('id', 'unknown')}")
    
    job_input = job["input"]
    job_type = job_input.get("type", "unknown")
    
    print(f"📦 Processing job type: {job_type}")
    print(f"📝 Full input: {job_input}")
    
    if job_type == "health" or job_type == "health_check":
        print("✅ Health check completed")
        return {
            "status": "healthy",
            "message": "Minimal handler working",
            "job_id": job.get('id', 'unknown')
        }
    
    print(f"🔄 Processing {job_type} request...")
    return {
        "status": "completed", 
        "job_type": job_type,
        "message": f"Processed {job_type} successfully",
        "job_id": job.get('id', 'unknown')
    }

if __name__ == "__main__":
    print("🚀 Starting MINIMAL RunPod Handler")
    runpod.serverless.start({"handler": handler})