#!/usr/bin/env python3
"""
🧪 Test Token Configuration
Sprawdza czy tokeny są poprawnie skonfigurowane
"""

import os
from pathlib import Path

def test_config_file():
    """Test if config.env contains required tokens"""
    print("🧪 Testing config.env file...")
    
    config_path = Path("config.env")
    if not config_path.exists():
        print("❌ config.env file not found")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    required_tokens = [
        "RUNPOD_API_TOKEN=rpa_G4713KLVTYYBJYWPO157LX7VVPGV7NZ2K87SX6B17otl1t",
        'HF_TOKEN="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"'
    ]
    
    for token in required_tokens:
        if token in content:
            print(f"✅ Found: {token[:20]}...")
        else:
            print(f"❌ Missing: {token[:20]}...")
            return False
    
    return True

def test_environment_loading():
    """Test environment variable loading"""
    print("\n🧪 Testing environment variable loading...")
    
    # Simulate loading config.env
    config_vars = {}
    
    try:
        with open("config.env", 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    config_vars[key] = value
        
        # Test key variables
        test_vars = ['RUNPOD_API_TOKEN', 'HF_TOKEN']
        
        for var in test_vars:
            if var in config_vars and config_vars[var]:
                print(f"✅ {var}: {config_vars[var][:15]}...")
            else:
                print(f"❌ {var}: Not found or empty")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_runtime_setup():
    """Test if runtime setup would work with tokens"""
    print("\n🧪 Testing runtime setup compatibility...")
    
    # Check if HF_TOKEN would be available for runtime setup
    with open("app/rp_handler.py", 'r') as f:
        content = f.read()
    
    if 'os.environ.get("HF_TOKEN"' in content:
        print("✅ HF_TOKEN is used in runtime setup")
    else:
        print("❌ HF_TOKEN not found in runtime setup")
        return False
    
    if 'huggingface-cli' in content and 'login' in content:
        print("✅ HuggingFace CLI login implemented")
    else:
        print("❌ HuggingFace CLI login not found")
        return False
    
    return True

def main():
    """Run all token tests"""
    print("🚀 Testing Backend/ Token Configuration")
    print("=" * 50)
    
    tests = [
        ("Config File", test_config_file),
        ("Environment Loading", test_environment_loading),
        ("Runtime Setup", test_runtime_setup)
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
        print("🎉 All token tests PASSED! Tokens are correctly configured!")
    else:
        print("⚠️ Some token tests FAILED. Check configuration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())