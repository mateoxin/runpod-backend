# 🚀 LoRA Dashboard Backend

FastAPI backend for LoRA Dashboard - Serverless Training & Generation Suite
Optimized for RunPod deployment with fast startup times.

## 🎯 Features

- **Ultra-Fast Deployment**: ~30 seconds instead of 20 minutes
- **Runtime Setup**: Heavy dependencies installed at runtime using RunPod cache  
- **Unified Logging**: stdout/stderr output for maximum RunPod visibility
- **Git Deploy Ready**: Optimized for deployment from Git repositories

## 🚀 Quick Start

### Local Development

```bash
# Setup environment
./setup_env.sh

# Activate virtual environment  
source venv/bin/activate

# Start RunPod handler
python rp_handler.py
```

### RunPod Deployment

1. **From Git Repository** (Recommended):
   ```bash
   # Use this repository URL in RunPod endpoint configuration
   https://github.com/your-username/your-repo.git
   ```

2. **Manual Docker**:
   ```bash
   docker build -t lora-backend .
   docker run -p 8000:8000 lora-backend
   ```

## 📁 Project Structure

```
Backend/
├── rp_handler.py              # RunPod serverless handler (all-in-one)
├── requirements_minimal.txt   # Minimal dependencies for fast startup
├── Dockerfile                 # Optimized for quick builds
├── startup.sh                 # Environment startup script
├── setup_env.sh               # Local development setup
├── config.env.template        # Configuration template
└── runpod.yaml                # RunPod deployment configuration
```

## 🔧 Configuration

1. Copy configuration template:
   ```bash
   cp config.env.template config.env
   ```

2. Edit `config.env` with your settings:
   ```bash
   # HuggingFace Token (set via RunPod environment variables)
   HF_TOKEN=your_huggingface_token
   
   # Environment (auto-configured)
   WORKSPACE_PATH=/workspace
   ```

## 🧪 Testing

Test specific components:
```bash
# Test handler syntax
python -m py_compile rp_handler.py

# Test RunPod handler
python rp_handler.py

# Test imports
python -c "import rp_handler"
```

## 📊 Deployment Approach

### Fast Deployment Strategy
- **Minimal Docker Image**: Only essential packages in Dockerfile
- **Runtime Setup**: Heavy ML libraries installed when needed
- **RunPod Cache**: Leverages RunPod's package cache for faster installs
- **Lazy Loading**: Services imported only when required

### Traditional vs Fast Deployment
| Aspect | Traditional | Fast Approach |
|--------|-------------|---------------|
| Build Time | 15-20 minutes | 30-60 seconds |
| Image Size | 5-8 GB | 500MB-1GB |
| Startup Time | 2-3 minutes | 30-60 seconds |
| Cache Usage | Limited | Full RunPod cache |

## 🛠️ Dependencies

### Minimal (Pre-installed)
- `runpod>=1.7.0` - RunPod serverless SDK
- `fastapi>=0.104.1` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `pydantic>=2.5.0` - Data validation
- `httpx>=0.25.0` - HTTP client

### Runtime (Installed when needed)
- `boto3>=1.34.0` - AWS S3 operations
- `torch` - Machine learning framework
- `transformers` - NLP models
- `diffusers` - Diffusion models

**Note**: Redis not used - RunPod provides built-in queue system

## 🔄 API Endpoints

- `GET /api/health` - Health check
- `POST /api/train` - Start training process
- `POST /api/generate` - Start generation process  
- `GET /api/processes` - List all processes
- `GET /api/processes/{id}` - Get process status
- `DELETE /api/processes/{id}` - Cancel process
- `POST /api/upload/training-data` - Upload training files
- `GET /api/download/{id}` - Get download URL

## 📝 Logging

The system uses unified logging for RunPod compatibility:
- **stdout/stderr**: Maximum visibility in RunPod logs
- **File logging**: Detailed logs saved to `/workspace/logs/`
- **Request tracking**: Each request gets unique ID
- **Error tracking**: Detailed error information with context

## 🎛️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKSPACE_PATH` | Working directory | `/workspace` |
| `HF_TOKEN` | HuggingFace token | Required |
| `DEBUG` | Debug mode | `false` |

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Run runtime setup
   ```bash
   # Inside RunPod container
   python rp_handler.py
   ```

2. **Permission Errors**: Check workspace permissions
   ```bash
   chmod -R 755 /workspace
   ```

3. **Memory Issues**: Reduce concurrent processes
   ```bash
   export MAX_CONCURRENT_JOBS=2
   ```

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## 📈 Performance Tips

1. **Use RunPod Cache**: Let runtime setup handle heavy packages
2. **Minimal Base Image**: Keep Dockerfile dependencies minimal  
3. **Lazy Loading**: Import services only when needed
4. **Process Pooling**: Reuse initialized services
5. **Memory Management**: Monitor memory usage in `/api/health`

## 🤝 Contributing

1. Follow the fast deployment pattern
2. Add runtime setup for heavy dependencies
3. Use unified logging for all output
4. Test deployment on RunPod
5. Update documentation

## 📄 License

This project is licensed under the MIT License.