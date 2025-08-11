# 🚀 LoRA Dashboard Backend (RunPod Serverless)

Serverless handler for LoRA Dashboard — training and generation on RunPod with fast cold‑start and S3 storage. No FastAPI server is used; the worker is started via `runpod.serverless.start` and communicates through RunPod input/output payloads.

## 🎯 Features

- **Ultra‑fast deployment**: ~30–60s cold‑start, ciężkie biblioteki dogrywane w runtime z cache RunPod
- **Serverless (Dockerless) ready**: pracuje świetnie z RunPod Projects (repo GitHub → Endpoint)
- **S3 jako source of truth**: upload datasetów, wyniki i statusy procesów przez S3 (presigned URLs)
- **Cross‑worker visibility**: statusy procesów z S3 widoczne między workerami
- **Bezpośrednie logowanie**: stdout/stderr dla maksymalnej widoczności w RunPod

## 🚀 Quick Start

### Local Development

```bash
# Setup environment
./setup_env.sh

# Activate virtual environment  
source venv/bin/activate

# Start RunPod handler (local run for smoke tests)
python rp_handler.py
```

### RunPod Deployment (Projects — Dockerless, z GitHub)

1) W konsoli RunPod utwórz Project i podepnij repo (Gałąź: `main`).
2) Utwórz Endpoint z Project, wybierz GPU, Workers.
3) Ustaw zmienne środowiskowe (sekcje poniżej). Sekrety podaj jako Secrets.
4) W `Test Input` używaj formatu `{"input": { ... }}` (przykłady niżej).
5) Deploy. Worker uruchamia `rp_handler.py` i nasłuchuje przez RunPod API.

## 📁 Project Structure

```
Backend/
├── rp_handler.py              # RunPod serverless handler (entrypoint)
├── storage_utils.py           # S3 manager (batch, multipart, presigned URLs)
├── models.py                  # Pydantic modele walidacji inputu
├── utils.py                   # GPU utils, retry, metrics, path safety
├── requirements_minimal.txt   # Minimal dependencies for fast startup
├── Dockerfile                 # (opcjonalnie) pod dockerowy deploy
├── startup.sh                 # (opcjonalnie) przykład ENV pod docker
├── setup_env.sh               # Local development setup
├── config.env.template        # Configuration template
└── runpod.yaml                # RunPod deployment configuration
```

## 🔧 Configuration (ENV)

1. Copy configuration template:
   ```bash
   cp config.env.template config.env
   ```

Ustaw w Endpoint → Variables/Secrets:

- `HF_TOKEN` (Secret) — token HuggingFace
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (Secrets) — dostęp do S3 kompatybilnego
- `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`, `S3_PREFIX` — konfiguracja storage (np. `tqv92ffpc5`, `eu-ro-1`, `https://s3api-eu-ro-1.runpod.io`, `lora-dashboard`)
- (opcjonalnie) `TRAINING_TIMEOUT`, `GENERATION_TIMEOUT`, `MAX_RETRIES`

## 🧪 Testing (local smoke)

Test specific components:
```bash
# Test handler syntax
python -m py_compile rp_handler.py

# Test RunPod handler
python rp_handler.py

# Test imports
python -c "import rp_handler"
```

## 📦 Payloads i wyniki

- Limity payload (RunPod): `run` 10 MB, `runsync` 20 MB.
- Wyniki i artefakty zwracamy jako presigned URL z S3 (nie base64 dla dużych plików).
- Statusy procesów są zapisywane do S3, co zapewnia widoczność między workerami.

## 🛠️ Dependencies

Minimal (preinstall): `runpod`, `pyyaml`, `pydantic`, `httpx`, `python-multipart`, `boto3`, `psutil`.
Runtime (instalowane w locie przez handler): `torch`, `transformers`, `diffusers`, `accelerate`, `huggingface_hub[cli]`.

**Note**: Redis not used - RunPod provides built-in queue system

## 🔄 Serverless input types (RunPod `input`)

Wywołujemy Endpoint z `{ "input": { ... } }`. Obsługiwane typy:

- `health`
- `train` / `train_with_yaml` — YAML config jako string
- `generate` — `config` lub `prompt`
- `processes`, `process_status`, `cancel`
- `upload_training_data` — pliki w base64
- `bulk_download`, `list_files`, `download_file`
- `list_dataset_folders` — lista folderów pod `s3://<bucket>/<prefix>/dataset/`

Przykład `Test Input` (train):

```json
{
  "input": {
    "type": "train_with_yaml",
    "yaml_config": "config: { process: [ { type: lora } ], datasets: [ { folder_path: 'my-training' } ] }"
  }
}
```

Przykład upload datasetu (do wybranego folderu w `dataset/`):

```json
{
  "input": {
    "type": "upload_training_data",
    "training_name": "my-training",
    "dataset_folder": "matt_2025_08_10",
    "files": [
      { "filename": "img_001.jpg", "content": "<base64>", "content_type": "image/jpeg" },
      { "filename": "img_001.txt", "content": "<base64>", "content_type": "text/plain" }
    ]
  }
}
Przykład listy folderów datasetu:

```json
{
  "input": {
    "type": "list_dataset_folders"
  }
}
```
```

## 📝 Logging

The system uses unified logging for RunPod compatibility:
- **stdout/stderr**: Maximum visibility in RunPod logs
- **File logging**: Detailed logs saved to `/workspace/logs/`
- **Request tracking**: Each request gets unique ID
- **Error tracking**: Detailed error information with context

## 🎛️ Environment Variables — skrót

- `HF_TOKEN` (required)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (required)
- `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`, `S3_PREFIX`
- `TRAINING_TIMEOUT`, `GENERATION_TIMEOUT`, `MAX_RETRIES` (opcjonalne)

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