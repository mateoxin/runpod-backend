#!/bin/bash
# 🚀 Startup script for Backend/ 
# Based on successful runpod-fastbackend/ approach - FLAT STRUCTURE

echo "🚀 Starting LoRA Dashboard Backend..."

# Set environment variables (flat structure like runpod-fastbackend)
export PYTHONUNBUFFERED=1
export WORKSPACE_PATH=${WORKSPACE_PATH:-/workspace}

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