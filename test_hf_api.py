#!/usr/bin/env python3
"""
Test HuggingFace Inference API connection for BioMistral.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')
load_dotenv()  # Also try .env

token = os.getenv('HUGGINGFACE_HUB_TOKEN')

print("=" * 60)
print("HuggingFace Inference API Test")
print("=" * 60)
print()

# Check token
if not token:
    print("❌ HUGGINGFACE_HUB_TOKEN not found!")
    print()
    print("Please add your token to .env.local:")
    print("  HUGGINGFACE_HUB_TOKEN=hf_your_token_here")
    print()
    print("Get a token from: https://huggingface.co/settings/tokens")
    exit(1)

print(f"✅ Token found: {token[:10]}...{token[-5:]}")
print()

# Test API connection
print("Testing API connection to BioMistral...")
try:
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(token=token)
    
    # Simple test prompt
    response = client.text_generation(
        "Describe the liver in one sentence.",
        model="BioMistral/BioMistral-7B",
        max_new_tokens=50,
    )
    
    print("✅ API connection successful!")
    print()
    print("Test response:")
    print(f"  {response}")
    print()
    print("=" * 60)
    print("✅ All good! Your API server will use BioMistral via cloud.")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ API test failed: {e}")
    print()
    print("Common issues:")
    print("1. Token might be invalid - check it at huggingface.co/settings/tokens")
    print("2. Model might be loading (try again in 30 seconds)")
    print("3. Internet connection issue")
