#!/usr/bin/env python3
"""
Simple verification script to check if tokens are properly hardcoded.
Does not require any dependencies - just reads files.
"""

import os

def verify_hardcoded_tokens():
    """Verify that tokens are hardcoded in the source files."""
    
    print("🔍 VERIFYING HARDCODED TOKENS")
    print("=" * 50)
    
    errors = []
    success = []
    
    # Expected tokens
    expected_hf_token = "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"
    expected_runpod_token = "rpa_G4713KLVTYYBJYWPO157LX7VVPGV7NZ2K87SX6B17otl1t"
    
    # Check 1: HF_TOKEN in rp_handler.py
    print("1. Checking HF_TOKEN in app/rp_handler.py...")
    try:
        with open('app/rp_handler.py', 'r') as f:
            content = f.read()
        
        if f'hf_token = "{expected_hf_token}"' in content:
            success.append("✅ HF_TOKEN properly hardcoded in rp_handler.py")
        elif 'os.environ.get("HF_TOKEN"' in content:
            errors.append("❌ HF_TOKEN still uses os.environ.get in rp_handler.py")
        else:
            errors.append("❌ HF_TOKEN not found in expected format in rp_handler.py")
            
    except FileNotFoundError:
        errors.append("❌ app/rp_handler.py not found")
    
    # Check 2: RUNPOD_TOKEN in config_loader.py
    print("2. Checking RUNPOD_TOKEN in app/utils/config_loader.py...")
    try:
        with open('app/utils/config_loader.py', 'r') as f:
            content = f.read()
        
        if f'return "{expected_runpod_token}"' in content:
            success.append("✅ RUNPOD_TOKEN properly hardcoded in config_loader.py")
        elif 'get_config_value(' in content and 'RUNPOD_TOKEN' in content:
            errors.append("❌ RUNPOD_TOKEN still uses get_config_value in config_loader.py")
        else:
            errors.append("❌ RUNPOD_TOKEN not found in expected format in config_loader.py")
            
    except FileNotFoundError:
        errors.append("❌ app/utils/config_loader.py not found")
    
    # Check 3: RUNPOD_TOKEN in core/config.py
    print("3. Checking RUNPOD_TOKEN in app/core/config.py...")
    try:
        with open('app/core/config.py', 'r') as f:
            content = f.read()
        
        if f'return "{expected_runpod_token}"' in content:
            success.append("✅ RUNPOD_TOKEN properly hardcoded in core/config.py")
        elif 'os.getenv(' in content and 'RUNPOD_TOKEN' in content:
            errors.append("❌ RUNPOD_TOKEN still uses os.getenv in core/config.py")
        else:
            errors.append("❌ RUNPOD_TOKEN not found in expected format in core/config.py")
            
    except FileNotFoundError:
        errors.append("❌ app/core/config.py not found")
    
    # Check 4: No environment variable dependencies
    print("4. Checking for remaining environment variable dependencies...")
    files_to_check = [
        'app/rp_handler.py',
        'app/utils/config_loader.py', 
        'app/core/config.py'
    ]
    
    env_patterns = [
        'os.environ.get("HF_TOKEN"',
        'os.environ.get("RUNPOD_TOKEN"',
        'os.getenv("HF_TOKEN"',
        'os.getenv("RUNPOD_TOKEN"'
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in env_patterns:
                if pattern in content:
                    errors.append(f"❌ Found environment dependency in {file_path}: {pattern}")
        except FileNotFoundError:
            pass
    
    if not any(pattern in content for pattern in env_patterns for file_path in files_to_check if os.path.exists(file_path)):
        success.append("✅ No environment variable dependencies found for tokens")
    
    # Print results
    print("\n📊 VERIFICATION RESULTS:")
    print("=" * 30)
    
    for msg in success:
        print(msg)
    
    for msg in errors:
        print(msg)
    
    if errors:
        print(f"\n❌ VERIFICATION FAILED: {len(errors)} error(s) found")
        return False
    else:
        print(f"\n🎉 VERIFICATION PASSED: {len(success)} check(s) successful")
        print("✅ All tokens are properly hardcoded")
        print("✅ No environment variables needed")
        print("✅ Ready for RunPod deployment")
        return True

if __name__ == "__main__":
    success = verify_hardcoded_tokens()
    exit(0 if success else 1)