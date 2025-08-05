# 🚀 **PEŁNA INSTRUKCJA DEPLOY BACKEND Z GITHUB NA RUNPOD**

## 📋 **KROK ZA KROKIEM - RUNPOD SERVERLESS ENDPOINT**

### **1️⃣ PRZYGOTOWANIE GITHUB REPOSITORY**

✅ **Repository już gotowe**: `https://github.com/mateoxin/runpod-backend.git`
✅ **Branch**: `main`
✅ **Kod zaktualizowany**: HF token ze zmiennych środowiskowych

### **2️⃣ LOGOWANIE DO RUNPOD CONSOLE**

1. **Idź do**: [https://runpod.io/console](https://runpod.io/console)
2. **Zaloguj się** do swojego konta RunPod
3. **Przejdź do**: `Serverless` → `My Endpoints`

### **3️⃣ TWORZENIE NOWEGO ENDPOINT**

1. **Kliknij**: `+ New Endpoint`
2. **Wybierz**: `Deploy from GitHub`

### **4️⃣ KONFIGURACJA GITHUB REPOSITORY**

```yaml
Repository Settings:
  Repository URL: https://github.com/mateoxin/runpod-backend.git
  Branch: main
  Docker Context: /
  Dockerfile Path: Dockerfile
```

### **5️⃣ KONFIGURACJA ŚRODOWISKA (ENVIRONMENT VARIABLES)**

⚠️ **TYLKO JEDNA ZMIENNA DO USTAWIENIA**:

```bash
# 🔑 JEDYNA WYMAGANA ZMIENNA
HF_TOKEN=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
```

✅ **AUTOMATYCZNIE USTAWIONE W DOCKERFILE** (nie musisz ich dodawać):
```bash
# ✅ Już skonfigurowane automatycznie:
WORKSPACE_PATH=/workspace
PYTHONUNBUFFERED=1
HOST=0.0.0.0
PORT=8000
DEBUG=false
MOCK_MODE=false
MAX_CONCURRENT_JOBS=10
GPU_TIMEOUT=14400
```

🎯 **OPCJONALNE** (tylko jeśli chcesz zmienić domyślne):
```bash
# Możesz nadpisać w RunPod Console jeśli potrzebujesz:
REDIS_URL=redis://your-redis:6379/0  # domyślnie: redis://localhost:6379/0
MAX_CONCURRENT_JOBS=20               # domyślnie: 10
GPU_TIMEOUT=7200                     # domyślnie: 14400 (4h)
```

### **6️⃣ KONFIGURACJA GPU**

```yaml
GPU Configuration:
  GPU Type: NVIDIA A40 (24GB VRAM) - ZALECANE
  Alternative: RTX A6000, A100
  Min VRAM: 16GB dla FLUX.1-dev
  
Scaling:
  Min Workers: 0 (scale to zero)
  Max Workers: 10
  Scale Down Delay: 5 minutes
  Worker Timeout: 15 minutes
```

### **7️⃣ KONFIGURACJA DOCKER**

```yaml
Docker Settings:
  Container Start Command: python -u /app/app/rp_handler.py
  Container Registry: (leave empty for GitHub)
  Container Disk: 50GB (recommended)
```

### **8️⃣ DEPLOY I MONITORING**

1. **Kliknij**: `Deploy`
2. **Monitoruj logi** - oczekiwany output:

```
🚀 Setting up environment at runtime...
📦 Installing Redis and AWS dependencies...
✅ Redis and AWS dependencies installed successfully
📁 Creating workspace directories...
🤗 Setting up HuggingFace token...
✅ HuggingFace token configured
🤖 Installing AI-Toolkit (ostris/ai-toolkit)...
📥 Cloning AI-Toolkit repository...
✅ AI-Toolkit repository cloned successfully
📦 Installing AI-Toolkit dependencies...
✅ AI-Toolkit dependencies installed successfully
✅ Environment setup completed successfully!
```

**⏱️ Czas startup**: 3-5 minut (pierwsze uruchomienie)

### **9️⃣ TESTOWANIE ENDPOINT**

Po udanym deploy otrzymasz **Endpoint URL**:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
```

#### **Test Health Check:**
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_TOKEN" \
  -d '{
    "input": {
      "type": "health_check"
    }
  }'
```

#### **Oczekiwany Response:**
```json
{
  "id": "unique-request-id",
  "status": "COMPLETED",
  "output": {
    "status": "healthy",
    "environment_ready": true,
    "hf_token_configured": true,
    "ai_toolkit_available": true,
    "uptime": "00:05:23"
  }
}
```

---

## 🔧 **ADVANCED CONFIGURATION**

### **🌐 Environment Variables - Pełna Lista**

| Variable | Required | Auto-Set | Default | Description |
|----------|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ **YES** | ❌ | - | HuggingFace token (USTAW RĘCZNIE) |
| `WORKSPACE_PATH` | No | ✅ | `/workspace` | Main workspace path |
| `PYTHONUNBUFFERED` | No | ✅ | `1` | Python logging |
| `HOST` | No | ✅ | `0.0.0.0` | Server host |
| `PORT` | No | ✅ | `8000` | Server port |
| `DEBUG` | No | ✅ | `false` | Debug mode |
| `MOCK_MODE` | No | ✅ | `false` | Mock mode for testing |
| `MAX_CONCURRENT_JOBS` | No | ✅ | `10` | Max parallel jobs |
| `GPU_TIMEOUT` | No | ✅ | `14400` | GPU timeout (seconds) |
| `REDIS_URL` | No | ❌ | `redis://localhost:6379/0` | Redis connection |

### **📊 Performance Optimization**

#### **GPU Selection Guide:**
- **NVIDIA A40** (24GB): Najlepszy stosunek cena/wydajność
- **RTX A6000** (48GB): Dla dużych modeli
- **A100** (40GB/80GB): Najszybszy, ale droższy

#### **Cost Optimization:**
- **Scale to Zero**: Automatyczne po 5 minutach bezczynności
- **Shared Storage**: Użyj RunPod Network Storage dla modeli
- **Model Caching**: Modele cachowane między uruchomieniami

### **🔍 Monitoring i Debugging**

#### **Logi do monitorowania:**
```
✅ Environment setup completed successfully!
🔄 Handler started successfully
📡 Server listening on 0.0.0.0:8000
```

#### **Błędy do obserwacji:**
```
❌ HF token login failed
❌ AI-Toolkit clone failed  
❌ GPU allocation failed
```

---

## 🚨 **TROUBLESHOOTING**

### **Problem 1: "Environment Setup Failed"**
**Przyczyna**: Brak HF_TOKEN lub błędna wartość
**Rozwiązanie**: 
1. Sprawdź Environment Variables w RunPod Console
2. Upewnij się że `HF_TOKEN=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`

### **Problem 2: "AI-Toolkit Clone Failed"**
**Przyczyna**: Problemy z siecią lub brak git
**Rozwiązanie**: 
1. Sprawdź logi - czy git jest dostępny
2. Może być timeout - zwiększ timeout w kodzie

### **Problem 3: "GPU Not Available"**
**Przyczyna**: Brak dostępnych GPU w regionie
**Rozwiązanie**:
1. Zmień region w RunPod Console
2. Wybierz inny typ GPU (A40 → RTX A6000)
3. Poczekaj na dostępność

### **Problem 4: "Slow Startup"**
**Przyczyna**: Pierwsze uruchomienie - instalacja zależności
**Rozwiązanie**: 
- Pierwszy start: 3-5 minut (normalne)
- Kolejne starty: 30-60 sekund (cache)

---

## ✅ **SUPER PROSTY CHECKLIST PRZED DEPLOY**

- [ ] Repository URL: `https://github.com/mateoxin/runpod-backend.git`
- [ ] Branch: `main`
- [ ] Dockerfile Path: `Dockerfile`
- [ ] **JEDYNA RĘCZNA KONFIGURACJA**: HF_TOKEN = `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`
- [ ] GPU Type: A40 lub lepszy
- [ ] Min Workers: 0, Max Workers: 10
- [ ] Container Start Command: `python -u /app/app/rp_handler.py`

🎉 **POZOSTAŁE ZMIENNE AUTOMATYCZNIE USTAWIONE W DOCKERFILE!**

---

## 🎉 **PO UDANYM DEPLOY**

### **Dostępne API Endpoints:**
```
✅ Health Check: POST /runsync {"input": {"type": "health_check"}}
✅ LoRA Training: POST /runsync {"input": {"type": "train", "config": {...}}}
✅ Image Generation: POST /runsync {"input": {"type": "generate", "prompt": "..."}}
✅ Process Management: GET /health, /api/processes
✅ File Management: POST /api/upload, GET /api/files
```

### **Przykład użycia:**
```python
import requests

endpoint_url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID"
headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_TOKEN",
    "Content-Type": "application/json"
}

# Health check
response = requests.post(
    f"{endpoint_url}/runsync",
    headers=headers,
    json={"input": {"type": "health_check"}}
)

print(response.json())
```

---

## 📞 **SUPPORT**

W przypadku problemów:
1. **Sprawdź logi** w RunPod Console
2. **Zweryfikuj Environment Variables**
3. **Sprawdź dostępność GPU** w wybranym regionie
4. **Poczekaj na pełny startup** (5 minut)

**🚀 GOTOWE! Backend jest teraz dostępny na RunPod!**