#!/bin/bash
# 🚀 Startup script for Backend/ 
# Based on successful runpod-fastbackend/ approach - FLAT STRUCTURE

echo "🚀 Starting LoRA Dashboard Backend..."

# Set environment variables (flat structure like runpod-fastbackend)
export PYTHONUNBUFFERED=1
export WORKSPACE_PATH=${WORKSPACE_PATH:-/workspace}

# S3-compatible storage defaults
# Only AWS credentials are expected to be provided as environment variables.
# All other S3 parameters are set here at startup and can be overridden per-branch/image if needed.
export PROCESS_STORE_BACKEND=${PROCESS_STORE_BACKEND:-s3}
export S3_BUCKET=${S3_BUCKET:-tqv92ffpc5}
export S3_REGION=${S3_REGION:-eu-ro-1}
export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-https://s3api-eu-ro-1.runpod.io}
export S3_FORCE_PATH_STYLE=${S3_FORCE_PATH_STYLE:-true}
export S3_PREFIX=${S3_PREFIX:-lora-dashboard}

echo "🗄️  S3 storage configured: bucket=$S3_BUCKET region=$S3_REGION endpoint=$S3_ENDPOINT_URL prefix=$S3_PREFIX"

# Create workspace directories
mkdir -p $WORKSPACE_PATH/training_data
mkdir -p $WORKSPACE_PATH/models 
mkdir -p $WORKSPACE_PATH/logs

echo "📁 Created workspace directories in $WORKSPACE_PATH"

# Set permissions if needed
if [ -w $WORKSPACE_PATH ]; then
    chmod -R 755 $WORKSPACE_PATH
    echo "✅ Set workspace permissions"
fi

# Check if we're in RunPod or local environment
if [ -n "$RUNPOD_WORKER_ID" ]; then
    echo "🔧 Running in RunPod environment (Worker: $RUNPOD_WORKER_ID)"
    # RunPod specific setup
    export IS_RUNPOD=true
else
    echo "🏠 Running in local environment" 
    export IS_RUNPOD=false
fi

# Start the handler (flat structure like runpod-fastbackend)
echo "🚀 Starting RunPod handler..."
cd /
python -u /rp_handler.py