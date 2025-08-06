#!/usr/bin/env python3
"""
🧹 GPU Memory Cleanup Script
Sprawdza i czyści procesy zajmujące pamięć GPU
"""

import subprocess
import json
import time
import os
import signal
from datetime import datetime

def log(message, level="INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def get_gpu_memory_info():
    """Pobiera szczegółowe informacje o pamięci GPU"""
    try:
        # GPU memory usage
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 5:
                    name, total, used, free, util = parts
                    gpus.append({
                        "id": i,
                        "name": name,
                        "memory_total": int(total),
                        "memory_used": int(used), 
                        "memory_free": int(free),
                        "utilization": int(util),
                        "memory_percent": (int(used) / int(total)) * 100
                    })
            return gpus
        else:
            log(f"nvidia-smi failed: {result.stderr}", "ERROR")
            return []
    except Exception as e:
        log(f"Failed to get GPU info: {e}", "ERROR")
        return []

def get_gpu_processes():
    """Pobiera listę procesów używających GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            processes = []
            for line in lines:
                if line.strip() and 'No running processes found' not in line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts
                        processes.append({
                            "pid": int(pid),
                            "name": name,
                            "memory_mb": int(memory) if memory.isdigit() else 0
                        })
            return processes
        return []
    except Exception as e:
        log(f"Failed to get GPU processes: {e}", "ERROR")
        return []

def get_process_details(pid):
    """Pobiera szczegóły procesu"""
    try:
        # Get process command line
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,ppid,cmd', '--no-headers'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return f"PID {pid} - details unavailable"
    except:
        return f"PID {pid} - error getting details"

def check_if_ai_toolkit_process(cmd_line):
    """Sprawdza czy proces to AI toolkit"""
    ai_keywords = [
        'ai-toolkit', 'run.py', 'python', 'torch', 'transformers', 
        'diffusers', 'training', 'Matt_flux', 'sd_trainer'
    ]
    cmd_lower = cmd_line.lower()
    return any(keyword in cmd_lower for keyword in ai_keywords)

def kill_process_safely(pid, process_name, cmd_line):
    """Bezpiecznie kończy proces"""
    try:
        log(f"🔥 Attempting to kill process {pid} ({process_name})", "WARN")
        log(f"   Command: {cmd_line[:100]}...", "INFO")
        
        # Try graceful termination first
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        
        # Check if still running
        try:
            os.kill(pid, 0)  # Check if process exists
            log(f"   Process {pid} still running, using SIGKILL", "WARN")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            log(f"   ✅ Process {pid} terminated gracefully", "INFO")
            return True
            
        return True
    except ProcessLookupError:
        log(f"   ✅ Process {pid} already terminated", "INFO")
        return True
    except PermissionError:
        log(f"   ❌ Permission denied to kill process {pid}", "ERROR")
        return False
    except Exception as e:
        log(f"   ❌ Failed to kill process {pid}: {e}", "ERROR")
        return False

def clear_cuda_cache():
    """Czyści CUDA cache jeśli można"""
    try:
        log("🧹 Attempting to clear CUDA cache...", "INFO")
        
        # Try Python CUDA cache clear
        python_code = """
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("CUDA cache cleared")
else:
    print("CUDA not available")
"""
        
        result = subprocess.run(['python3', '-c', python_code], 
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            log(f"   ✅ {result.stdout.strip()}", "INFO")
        else:
            log(f"   ⚠️ CUDA cache clear failed: {result.stderr}", "WARN")
            
    except Exception as e:
        log(f"   ❌ Failed to clear CUDA cache: {e}", "ERROR")

def main():
    """Main cleanup function"""
    log("🚀 Starting GPU Memory Cleanup", "INFO")
    log("=" * 60, "INFO")
    
    # 1. Check current GPU status
    log("📊 Checking GPU memory status...", "INFO")
    gpus = get_gpu_memory_info()
    
    if not gpus:
        log("❌ No GPU information available", "ERROR")
        return
    
    for gpu in gpus:
        log(f"🎮 GPU {gpu['id']}: {gpu['name']}", "INFO")
        log(f"   Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_percent']:.1f}% used)", "INFO")
        log(f"   Free: {gpu['memory_free']} MB", "INFO")
        log(f"   Utilization: {gpu['utilization']}%", "INFO")
        
        if gpu['memory_percent'] > 80:
            log(f"   ⚠️ GPU {gpu['id']} memory usage is HIGH!", "WARN")
    
    log("", "INFO")
    
    # 2. Check GPU processes
    log("🔍 Checking GPU processes...", "INFO")
    processes = get_gpu_processes()
    
    if not processes:
        log("✅ No GPU processes found", "INFO")
        clear_cuda_cache()
        return
    
    log(f"Found {len(processes)} GPU processes:", "INFO")
    
    # 3. Analyze processes
    ai_processes = []
    other_processes = []
    
    for proc in processes:
        cmd_line = get_process_details(proc['pid'])
        log(f"📋 PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)", "INFO")
        log(f"   {cmd_line}", "INFO")
        
        if check_if_ai_toolkit_process(cmd_line):
            ai_processes.append((proc, cmd_line))
        else:
            other_processes.append((proc, cmd_line))
    
    log("", "INFO")
    
    # 4. Handle AI toolkit processes (likely stuck training)
    if ai_processes:
        log(f"🤖 Found {len(ai_processes)} AI toolkit processes (likely stuck training):", "WARN")
        
        for proc, cmd_line in ai_processes:
            log(f"   PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)", "WARN")
        
        response = input("\n🗑️  Kill these AI toolkit processes? [y/N]: ").strip().lower()
        
        if response == 'y':
            for proc, cmd_line in ai_processes:
                kill_process_safely(proc['pid'], proc['name'], cmd_line)
        else:
            log("⏭️  Skipping AI toolkit processes", "INFO")
    
    # 5. Handle other processes
    if other_processes:
        log(f"🔧 Found {len(other_processes)} other GPU processes:", "INFO")
        
        for proc, cmd_line in other_processes:
            log(f"   PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)", "INFO")
        
        response = input("\n🗑️  Kill these other processes? [y/N]: ").strip().lower()
        
        if response == 'y':
            for proc, cmd_line in other_processes:
                kill_process_safely(proc['pid'], proc['name'], cmd_line)
        else:
            log("⏭️  Skipping other processes", "INFO")
    
    # 6. Clear CUDA cache
    log("", "INFO")
    clear_cuda_cache()
    
    # 7. Final status check
    log("", "INFO")
    log("📊 Final GPU status:", "INFO")
    final_gpus = get_gpu_memory_info()
    
    for gpu in final_gpus:
        log(f"🎮 GPU {gpu['id']}: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_percent']:.1f}% used)", "INFO")
        log(f"   Free: {gpu['memory_free']} MB", "INFO")
    
    final_processes = get_gpu_processes()
    log(f"📋 Remaining GPU processes: {len(final_processes)}", "INFO")
    
    log("✅ GPU cleanup completed!", "INFO")

if __name__ == "__main__":
    main()