# 🚀 RunPod FastBackend Dockerfile
# Optimized for quick deployment and runtime setup
# Based on successful runpod-fastbackend/ approach - FLAT STRUCTURE

FROM python:3.11.1-slim

# Set working directory
WORKDIR /

# Install system dependencies including build tools for C compilation
# build-essential, gcc, g++, make are required for bitsandbytes/triton compilation
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
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