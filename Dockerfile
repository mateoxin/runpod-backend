# 🚀 RunPod FastBackend Dockerfile
# Optimized for quick deployment and runtime setup
# Based on successful runpod-fastbackend/ approach - FLAT STRUCTURE
# Using RunPod PyTorch template with CUDA support

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /

# Install additional system dependencies (build tools already included in RunPod template)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install minimal dependencies
COPY requirements_minimal.txt .
RUN pip install --no-cache-dir -r requirements_minimal.txt

# Copy the handler (flat structure like runpod-fastbackend)
COPY rp_handler.py .

# Create necessary directories
RUN mkdir -p /workspace/training_data \
    && mkdir -p /workspace/models \
    && mkdir -p /workspace/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_PATH=/workspace

# Command to run the handler (flat structure like runpod-fastbackend)
CMD ["python", "-u", "/rp_handler.py"] 