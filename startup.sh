#!/bin/bash
# 🚀 Startup script for Backend/ 
# Based on successful runpod-fastbackend/ approach

echo "🚀 Starting LoRA Dashboard Backend..."

# Set environment variables
export PYTHONUNBUFFERED=1
export WORKSPACE_PATH=${WORKSPACE_PATH:-/workspace}
export PYTHONPATH=/app

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

# Start the handler
echo "🚀 Starting RunPod handler..."
python -u /app/rp_handler.py