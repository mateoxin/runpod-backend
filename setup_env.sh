#!/bin/bash
# 🚀 Environment Setup Script for Backend/
# Based on successful runpod-fastbackend/ approach

echo "🚀 Setting up Backend/ environment..."

# Copy config template if config doesn't exist
if [ ! -f "config.env" ]; then
    if [ -f "config.env.template" ]; then
        cp config.env.template config.env
        echo "📋 Created config.env from template"
    else
        echo "⚠️ No config.env.template found"
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
echo "🔧 Activated virtual environment"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install minimal requirements
echo "📦 Installing minimal requirements..."
pip install -r requirements_minimal.txt

echo "✅ Backend/ environment setup complete!"
echo "💡 To activate: source venv/bin/activate"
echo "💡 To test locally: python app/main.py"
echo "💡 To test RunPod handler: python app/rp_handler.py"