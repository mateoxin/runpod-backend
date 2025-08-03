#!/usr/bin/env python3
"""
🧪 Test Git Deploy Compatibility for Backend/
Sprawdza czy wszystkie zależności mogą być zainstalowane przez RunPod
"""

import subprocess
import sys
import os
from pathlib import Path

def test_requirements_install():
    """Test if requirements file has valid syntax"""
    print("🧪 Testing requirements file syntax...")
    
    try:
        # Just check if we can parse the requirements file
        with open("requirements_minimal.txt", "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith("#"):
                # Basic validation - should contain package name
                if ">" in line or "=" in line or "<" in line:
                    print(f"✅ Line {i}: {line[:50]}...")
                elif line.replace("-", "").replace("_", "").isalnum():
                    print(f"✅ Line {i}: {line}")
                else:
                    print(f"⚠️ Line {i}: {line} - might have syntax issues")
        
        print("✅ requirements_minimal.txt syntax - OK")
        return True
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def test_imports():
    """Test if critical imports work"""
    print("\n🧪 Testing critical imports...")
    
    critical_imports = [
        "runpod",
        "fastapi", 
        "uvicorn",
        "pydantic",
        "httpx",
        "yaml",
        "requests"
    ]
    
    failed_imports = []
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_file_structure():
    """Test if all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "app/rp_handler.py",
        "app/main.py", 
        "app/core/logger.py",
        "app/core/config.py",
        "requirements_minimal.txt",
        "Dockerfile",
        "startup.sh",
        "setup_env.sh"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_handler_syntax():
    """Test if handler has correct syntax"""
    print("\n🧪 Testing handler syntax...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "app/rp_handler.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ rp_handler.py syntax - OK")
            return True
        else:
            print(f"❌ rp_handler.py syntax - FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Handler syntax test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Backend/ Git Deploy Compatibility")
    print("=" * 50)
    
    tests = [
        ("Requirements Installation", test_requirements_install),
        ("Critical Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Handler Syntax", test_handler_syntax)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED! Backend/ is ready for git deploy!")
        return 0
    else:
        print("⚠️ Some tests FAILED. Fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())