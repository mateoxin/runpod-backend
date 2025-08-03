#!/usr/bin/env python3
"""
🚀 Test GitHub Deploy Readiness for Backend/
Comprehensive check for RunPod serverless deployment via GitHub
"""

import os
import subprocess
import sys
from pathlib import Path

def test_essential_files():
    """Test if all essential files for GitHub deploy exist"""
    print("🧪 Testing essential files...")
    
    essential_files = [
        "Dockerfile",
        "requirements_minimal.txt", 
        "app/rp_handler.py",
        "app/main.py",
        "app/__init__.py",
        ".gitignore",
        "README.md",
        "config.env.template"
    ]
    
    missing_files = []
    
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_dockerfile_github_ready():
    """Test if Dockerfile is optimized for GitHub deploy"""
    print("\n🧪 Testing Dockerfile GitHub readiness...")
    
    try:
        with open("Dockerfile", 'r') as f:
            content = f.read()
        
        checks = [
            ("Base image", "FROM python:3.11" in content),
            ("Requirements copy", "requirements_minimal.txt" in content), 
            ("App copy", "COPY . /app" in content or "COPY app/ /app/" in content),
            ("Working directory", "WORKDIR" in content),
            ("CMD specified", "CMD" in content),
            ("Handler path", "rp_handler.py" in content)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading Dockerfile: {e}")
        return False

def test_no_duplicate_files():
    """Test that there are no duplicate files"""
    print("\n🧪 Testing for duplicate files...")
    
    try:
        result = subprocess.run(
            ["find", ".", "-name", "*\\(1\\)*"], 
            capture_output=True, text=True
        )
        
        duplicates = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if len(duplicates) == 0 or (len(duplicates) == 1 and duplicates[0] == ''):
            print("✅ No duplicate files found")
            return True
        else:
            print(f"❌ Found {len(duplicates)} duplicate files:")
            for dup in duplicates[:5]:  # Show first 5
                print(f"   {dup}")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates) - 5} more")
            return False
            
    except Exception as e:
        print(f"❌ Error checking duplicates: {e}")
        return False

def test_secrets_not_exposed():
    """Test that secrets are not exposed in Git"""
    print("\n🧪 Testing secrets protection...")
    
    try:
        with open(".gitignore", 'r') as f:
            gitignore_content = f.read()
        
        protected_patterns = [
            "config.env",
            "*.log", 
            "__pycache__",
            ".env"
        ]
        
        all_protected = True
        for pattern in protected_patterns:
            if pattern in gitignore_content:
                print(f"✅ {pattern} protected in .gitignore")
            else:
                print(f"❌ {pattern} NOT protected in .gitignore")
                all_protected = False
        
        # Check if config.env exists and has tokens
        if Path("config.env").exists():
            print("⚠️ config.env exists - make sure tokens are set via environment variables for GitHub deploy")
        
        return all_protected
        
    except Exception as e:
        print(f"❌ Error checking .gitignore: {e}")
        return False

def test_handler_entry_point():
    """Test if handler has correct entry point for RunPod"""
    print("\n🧪 Testing handler entry point...")
    
    try:
        with open("app/rp_handler.py", 'r') as f:
            content = f.read()
        
        checks = [
            ("RunPod import", "import runpod" in content),
            ("Handler function", "def handler(" in content),
            ("RunPod start", "runpod.serverless.start" in content),
            ("Main check", 'if __name__ == "__main__"' in content),
            ("Environment setup", "setup_environment" in content),
            ("Unified logging", "def log(" in content)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading rp_handler.py: {e}")
        return False

def test_python_syntax():
    """Test Python syntax of critical files"""
    print("\n🧪 Testing Python syntax...")
    
    python_files = [
        "app/rp_handler.py",
        "app/main.py"
    ]
    
    all_valid = True
    
    for file_path in python_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", file_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {file_path} syntax OK")
            else:
                print(f"❌ {file_path} syntax ERROR: {result.stderr}")
                all_valid = False
                
        except Exception as e:
            print(f"❌ Error checking {file_path}: {e}")
            all_valid = False
    
    return all_valid

def test_github_deploy_instructions():
    """Test if README has GitHub deploy instructions"""
    print("\n🧪 Testing GitHub deploy instructions...")
    
    try:
        with open("README.md", 'r') as f:
            content = f.read()
        
        instructions = [
            ("GitHub repo URL", "github.com" in content.lower() or "git" in content.lower()),
            ("RunPod deployment", "runpod" in content.lower()),
            ("Docker command", "docker" in content.lower() or "cmd" in content.lower()),
            ("Environment setup", "environment" in content.lower() or "config" in content.lower())
        ]
        
        all_present = True
        for instruction, present in instructions:
            status = "✅" if present else "❌"
            print(f"{status} {instruction}")
            if not present:
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        return False

def main():
    """Run all GitHub deploy readiness tests"""
    print("🚀 Testing Backend/ GitHub Deploy Readiness")
    print("=" * 60)
    
    tests = [
        ("Essential Files", test_essential_files),
        ("Dockerfile GitHub Ready", test_dockerfile_github_ready),
        ("No Duplicate Files", test_no_duplicate_files),
        ("Secrets Protection", test_secrets_not_exposed),
        ("Handler Entry Point", test_handler_entry_point),
        ("Python Syntax", test_python_syntax),
        ("GitHub Deploy Instructions", test_github_deploy_instructions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 GitHub Deploy Readiness Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Backend/ is READY for GitHub deploy to RunPod!")
        print("\n📝 Next steps:")
        print("1. Push to GitHub repository")
        print("2. Configure RunPod endpoint with GitHub repo URL")
        print("3. Set environment variables in RunPod console:")
        print("   - HF_TOKEN=your_token")
        print("   - RUNPOD_API_TOKEN=your_token")
        print("4. Deploy!")
        return 0
    else:
        print("⚠️ Some tests FAILED. Fix issues before GitHub deploy.")
        return 1

if __name__ == "__main__":
    sys.exit(main())