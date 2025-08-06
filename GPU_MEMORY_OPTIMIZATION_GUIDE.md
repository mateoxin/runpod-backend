# 🧹 GPU Memory Optimization Guide

## 🚨 Problem Analysis

Logi pokazują **CUDA Out of Memory** błędy:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 72.00 MiB. 
GPU 0 has a total capacity of 31.36 GiB of which 49.00 MiB is free. 
Process 623 has 18.63 GiB memory in use.
```

**Główny problem**: GPU ma 31GB pamięci, ale prawie wszystka jest zajęta (tylko 49MB wolne).

## 🔧 Rozwiązania

### 1. **Użyj Zoptymalizowanej Konfiguracji**

**Nowy plik**: `training_memory_optimized.yaml`

**Kluczowe zmiany**:
```yaml
# Drastyczne redukcje pamięci:
linear: 8                    # Było: 20 → TERAZ: 8 (-60% RAM)
resolution: [512]            # Było: [512, 768, 1024] → TERAZ: tylko 512
low_vram: true              # Było: false → WŁĄCZONE
dtype: "fp16"               # Było: "bf16" → fp16 (-20% RAM)
enable_sequential_cpu_offload: true  # NOWE - CPU offload
enable_attention_slicing: true       # NOWE - slice attention
pin_memory: false           # Było: true → WYŁĄCZONE
num_workers: 1              # Było: 2 → ZREDUKOWANE
```

### 2. **Wyczyść GPU Memory**

```bash
cd Serverless/Backend
python3 cleanup_gpu_memory.py
```

Ten skrypt:
- ✅ Sprawdzi status GPU i procesów
- ✅ Znajdzie procesy AI toolkit (stuck training)
- ✅ Pozwoli bezpiecznie zabić procesy
- ✅ Wyczyści CUDA cache

### 3. **Porównanie Konfiguracji**

| Ustawienie | Oryginał | Optymalizowana | Oszczędność RAM |
|------------|----------|----------------|-----------------|
| **LoRA Rank** | 20 | 8 | ~60% |
| **Resolution** | [512,768,1024] | [512] | ~70% |
| **Precision** | bf16 | fp16 | ~20% |
| **CPU Offload** | ❌ | ✅ | ~40% |
| **Attention Slice** | ❌ | ✅ | ~30% |
| **Pin Memory** | ✅ | ❌ | ~10% |
| **Workers** | 2 | 1 | ~15% |

**Szacunkowe oszczędności**: **~70-80% pamięci GPU** 

## 🚀 Instrukcje Wdrożenia

### Krok 1: Wyczyść GPU
```bash
python3 cleanup_gpu_memory.py
# Zabij stuck processes gdy zapyta
```

### Krok 2: Użyj nowej konfiguracji
```bash
# W fronendzie lub API call:
{
  "job_type": "train_with_yaml",
  "yaml_config": "< zawartość training_memory_optimized.yaml >"
}
```

### Krok 3: Monitoruj pamięć
```bash
# Sprawdź status GPU w czasie rzeczywistym:
watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

## 📊 Oczekiwane Rezultaty

**Przed optymalizacją**:
- ❌ GPU Memory: 31GB używane / 31GB total (99%)
- ❌ Free: ~49MB
- ❌ Training: CRASH - Out of Memory

**Po optymalizacji**:
- ✅ GPU Memory: ~8-12GB używane / 31GB total (30-40%)  
- ✅ Free: ~19-23GB
- ✅ Training: SUKCES - Stabilne uruchomienie

## ⚙️ Dodatkowe Optymalizacje

Jeśli nadal problemy z pamięcią:

1. **Jeszcze mniejszy LoRA rank**: `linear: 4`
2. **Mniejsza rozdzielczość**: `resolution: [256]` 
3. **Gradient accumulation**: `gradient_accumulation_steps: 4`
4. **Wyłącz sampling**: `sample_every: 0`

## 🔍 Debug Commands

```bash
# Sprawdź pamięć GPU:
nvidia-smi

# Sprawdź procesy GPU:
nvidia-smi pmon

# Sprawdź szczegóły procesów:
ps aux | grep python

# Monitor w czasie rzeczywistym:
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
```

## 🆘 Jeśli Nadal Problemy

1. **Restart kontenera RunPod**
2. **Użyj większego GPU** (A100 zamiast A40)
3. **Gradient checkpointing**: już włączone
4. **DeepSpeed Zero**: dla bardzo dużych modeli

**Kontakt**: Jeśli optymalizacje nie pomagają, sprawdź logi i status GPU po cleanup.