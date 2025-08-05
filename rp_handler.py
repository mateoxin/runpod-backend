#!/usr/bin/env python3
"""
🚀 RUNPOD HANDLER - EXACTLY LIKE OFFICIAL DOCUMENTATION
Based on official RunPod documentation examples
"""

import runpod

def handler(job):
    """Handler exactly like RunPod documentation examples"""
    print(f"🚀 Worker Start - job ID: {job.get('id', 'unknown')}")
    
    job_input = job["input"]
    job_type = job_input.get("type", "unknown")
    
    print(f"📦 Processing job type: {job_type}")
    
    # ONLY HEALTH CHECK - EXACTLY LIKE DOCS
    if job_type == "health":
        print(f"✅ Health check completed")
        return {
            "status": "healthy",
            "message": "DOCS EXAMPLE WORKS!"
        }
    
    # Everything else returns error
    print(f"❌ Unsupported job type: {job_type}")
    return {
        "error": f"Only 'health' type supported, got: {job_type}"
    }

if __name__ == "__main__":
    print("🚀 Starting RunPod Handler - DOCS EXAMPLE!")
    runpod.serverless.start({"handler": handler})