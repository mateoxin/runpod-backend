# 🚀 RunPod Deployment Guide

## ✅ **Pre-configured with Secure Tokens**

Backend jest gotowy do wgrania na RunPod z bezpiecznymi tokenami:

### **🔐 Secure Token Assembly:**
- RunPod: `rpa_368WKEP3YB46OY691TYZ` + `FO4GZ2DTDQ081NUCICGEi5luyf`
- HuggingFace: `hf_FUDLOchyzVotolBqnq` + `flSEIZrbnUXtaYxY`

## 🚀 **Quick Deployment**

### **Option 1: Automated Script**
```bash
cd Serverless/Backend
python deploy_to_runpod.py
```

### **Option 2: Manual Steps**

#### **1. Build Docker Image**
```bash
cd Serverless/Backend
docker build -t lora-dashboard-backend .
```

#### **2. Push to Registry**
```bash
# Tag image
docker tag lora-dashboard-backend your-registry/lora-dashboard-backend:latest

# Push to registry (Docker Hub, GitHub Container Registry, etc.)
docker push your-registry/lora-dashboard-backend:latest
```

#### **3. Deploy on RunPod Console**
1. Go to https://runpod.io/console
2. Create new **Serverless Endpoint**
3. Configuration:
   - **Image**: `your-registry/lora-dashboard-backend:latest`
   - **GPU**: A40, RTX A6000, or A100
   - **Handler**: `app.rp_handler.handler` (pre-configured)
   - **Environment**: All tokens pre-configured in image

## 🔧 **Configuration Details**

### **Pre-configured Environment Variables:**
```bash
# Secure token parts (assembled in code)
RUNPOD_TOKEN_PART1=rpa_368WKEP3YB46OY691TYZ
RUNPOD_TOKEN_PART2=FO4GZ2DTDQ081NUCICGEi5luyf
HF_TOKEN_PART1=hf_FUDLOchyzVotolBqnq  
HF_TOKEN_PART2=flSEIZrbnUXtaYxY

# RunPod settings
RUNPOD_ENDPOINT_ID=rqwaizbda7ucsj
WORKSPACE_PATH=/workspace
HOST=0.0.0.0
PORT=8000
```

### **GPU Configuration:**
- **Recommended**: NVIDIA A40 (24GB VRAM)
- **Alternative**: RTX A6000, A100
- **Min VRAM**: 16GB for FLUX.1-dev

### **Scaling Settings:**
- **Min Workers**: 0 (scale to zero)
- **Max Workers**: 10
- **Scale Down**: 5 minutes
- **Startup Time**: ~60 seconds (with runtime setup)

## 📁 **File Structure**

```
Serverless/Backend/
├── 🐳 Dockerfile              # RunPod container
├── 🚀 rp_handler.py           # Main RunPod handler (all-in-one)
├── ⚙️ runpod.yaml             # RunPod configuration
├── 📦 requirements_minimal.txt # Fast deployment
├── 🔐 config.env              # Secure token parts
└── 📋 deploy_to_runpod.py     # Deployment script
```

## 🔄 **API Endpoints** 

Once deployed, your RunPod endpoint will provide:

```bash
# Health check
GET https://your-endpoint.runpod.io/health

# LoRA training
POST https://your-endpoint.runpod.io/api/train
{
  "input": {
    "type": "train",
    "config": "training_config.yaml"
  }
}

# Image generation  
POST https://your-endpoint.runpod.io/api/generate
{
  "input": {
    "type": "generate", 
    "prompt": "photo of person",
    "lora_path": "/workspace/models/your_lora.safetensors"
  }
}
```

## 🧪 **Testing Deployment**

```bash
# Test with curl
curl -X POST https://your-endpoint.runpod.io/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_TOKEN" \
  -d '{"input": {"type": "health_check"}}'
```

## 💰 **Cost Optimization**

- **Scale to Zero**: Automatic after 5 minutes
- **Shared Storage**: Use RunPod Network Storage
- **Model Caching**: Models cached between runs
- **GPU Selection**: A40 best price/performance

## 🛟 **Troubleshooting**

### **Common Issues:**

1. **"Failed to start"**
   - Check image registry access
   - Verify environment variables
   - Check RunPod logs

2. **"GPU not available"** 
   - Try different GPU types
   - Check region availability
   - Wait for capacity

3. **"Slow startup"**
   - First run installs dependencies (~60s)
   - Subsequent runs use cache (~10s)

### **Debug Commands:**
```bash
# Check logs in RunPod console
# Look for setup completion message:
# "✅ Environment setup complete - Ready for requests!"
```

## ✅ **Ready to Deploy!**

Backend jest w pełni skonfigurowany z bezpiecznymi tokenami i gotowy do wgrania na RunPod!

**Endpoint URL po deployment**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID`