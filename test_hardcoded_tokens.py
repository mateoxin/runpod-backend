#!/usr/bin/env python3
"""
Test script to verify hardcoded tokens work without environment variables.
"""

import sys
import os

# Clear environment variables to test hardcoded values
if 'HF_TOKEN' in os.environ:
    del os.environ['HF_TOKEN']
if 'RUNPOD_TOKEN' in os.environ:
    del os.environ['RUNPOD_TOKEN']
if 'RUNPOD_API_TOKEN' in os.environ:
    del os.environ['RUNPOD_API_TOKEN']

print("🧪 TESTING HARDCODED TOKENS")
print("=" * 50)

try:
    # Test 1: Import config modules
    print("1. Testing imports...")
    from app.utils.config_loader import get_runpod_token
    from app.core.config import get_runpod_token as config_get_runpod_token
    print("   ✅ Imports successful")
    
    # Test 2: Test hardcoded RunPod token from config_loader
    print("2. Testing config_loader.get_runpod_token()...")
    token1 = get_runpod_token()
    expected_token = "rpa_G4713KLVTYYBJYWPO157LX7VVPGV7NZ2K87SX6B17otl1t"
    
    if token1 == expected_token:
        print(f"   ✅ Token matches: {token1[:10]}...{token1[-10:]}")
    else:
        print(f"   ❌ Token mismatch: got {token1}")
        sys.exit(1)
    
    # Test 3: Test hardcoded RunPod token from core.config
    print("3. Testing core.config.get_runpod_token()...")
    token2 = config_get_runpod_token()
    
    if token2 == expected_token:
        print(f"   ✅ Token matches: {token2[:10]}...{token2[-10:]}")
    else:
        print(f"   ❌ Token mismatch: got {token2}")
        sys.exit(1)
    
    # Test 4: Test HF token from rp_handler (simulated)
    print("4. Testing HF_TOKEN hardcoding...")
    
    # Read rp_handler.py to check for hardcoded token
    with open('app/rp_handler.py', 'r') as f:
        content = f.read()
    
    expected_hf_token = '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'
    if expected_hf_token in content:
        print(f"   ✅ HF_TOKEN found hardcoded: {expected_hf_token[:10]}...{expected_hf_token[-10:]}")
    else:
        print(f"   ❌ HF_TOKEN not found hardcoded in rp_handler.py")
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ No environment variables needed")
    print("✅ All tokens are properly hardcoded")
    print("✅ Ready for RunPod deployment")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)