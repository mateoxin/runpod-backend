# 🚀 RunPod FastBackend Dockerfile
# Optimized for quick deployment and runtime setup
# Based on successful runpod-fastbackend/ approach - FLAT STRUCTURE

FROM python:3.11.1-slim

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
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