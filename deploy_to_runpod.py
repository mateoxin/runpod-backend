#!/usr/bin/env python3
"""
🚀 Deploy LoRA Dashboard Backend to RunPod
With pre-configured secure tokens
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(cmd, description):
    """Run command with error handling"""
    log(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"❌ Error: {result.stderr}")
            return False
        log(f"✅ {description} completed")
        return True
    except Exception as e:
        log(f"❌ Exception: {e}")
        return False

def main():
    log("🚀 Starting RunPod deployment with secure tokens...")
    
    # Step 1: Build Docker image
    if not run_command("docker build -t lora-dashboard-backend .", "Building Docker image"):
        sys.exit(1)
    
    # Step 2: Tag for registry (update with your registry)
    registry = input("Enter your Docker registry (e.g., your-username): ")
    if registry:
        image_name = f"{registry}/lora-dashboard-backend:latest"
        if not run_command(f"docker tag lora-dashboard-backend {image_name}", "Tagging image"):
            sys.exit(1)
        
        # Step 3: Push to registry
        if not run_command(f"docker push {image_name}", "Pushing to registry"):
            sys.exit(1)
        
        log(f"✅ Image pushed: {image_name}")
    
    log("📋 Next steps:")
    log("1. Go to RunPod console: https://runpod.io/console")
    log("2. Create new Serverless Endpoint")
    log("3. Use the following configuration:")
    log(f"   - Image: {image_name if registry else 'lora-dashboard-backend'}")
    log("   - GPU: A40, RTX A6000, or A100")
    log("   - Environment variables are pre-configured in image")
    log("4. Deploy and test!")
    
    log("🔑 Tokens are securely split and assembled at runtime")
    log("🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()