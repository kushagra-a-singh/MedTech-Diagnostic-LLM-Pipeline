#!/usr/bin/env python3
"""
Download BioMistral model from HuggingFace.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Downloading BioMistral-7B model...")
print("This will download ~14GB and may take several minutes.")
print()

model_name = "BioMistral/BioMistral-7B"

# Download tokenizer
print("1. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("   ✓ Tokenizer downloaded")

# Download model with 4-bit quantization
print("2. Downloading model with 4-bit quantization...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("   ✓ Model downloaded")
print()
print("=" * 60)
print("BioMistral successfully downloaded!")
print("You can now restart the API server.")
print("=" * 60)
