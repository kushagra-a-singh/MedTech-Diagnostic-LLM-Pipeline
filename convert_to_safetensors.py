#!/usr/bin/env python3
"""
Script to convert PyTorch .bin model files to safetensors format.
This is required for PyTorch < 2.6 due to security restrictions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_model_to_safetensors(model_path: str, output_path: str = None):
    """
    Convert a model from PyTorch .bin format to safetensors format.
    
    Args:
        model_path: Path to the model directory
        output_path: Output path (defaults to model_path, overwrites original)
    """
    model_path = Path(model_path).expanduser().resolve()
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    if output_path is None:
        output_path = model_path
    else:
        output_path = Path(output_path).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting model from {model_path} to safetensors format...")
    logger.info(f"Output directory: {output_path}")
    
    # Check if model already has safetensors
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        logger.warning(f"Model already has safetensors files: {safetensors_files}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Conversion cancelled.")
            return
    
    try:
        # Load the model (this will work even with PyTorch < 2.6 if we use safetensors,
        # but for conversion we need to load from .bin first, which might fail)
        logger.info("Loading model...")
        logger.warning(
            "Note: If you get a PyTorch version error, you may need to:\n"
            "1. Temporarily upgrade PyTorch to 2.6+ for conversion, OR\n"
            "2. Download the model from HuggingFace Hub with safetensors format"
        )
        
        # Try loading with low_cpu_mem_usage to reduce memory requirements
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,  # Use float16 to reduce memory
            low_cpu_mem_usage=True,
            device_map="cpu",  # Load on CPU for conversion
        )
        
        logger.info("Model loaded successfully. Saving in safetensors format...")
        
        # Save with safetensors format
        model.save_pretrained(
            str(output_path),
            safe_serialization=True,  # This creates .safetensors files
        )
        
        # Also copy tokenizer files if they exist
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"✅ Model successfully converted to safetensors format!")
        logger.info(f"   Saved to: {output_path}")
        logger.info(f"   You can now delete the .bin files to save space.")
        
        # List the new files
        safetensors_files = list(output_path.glob("*.safetensors"))
        if safetensors_files:
            logger.info(f"   Created safetensors files: {[f.name for f in safetensors_files]}")
        
    except Exception as e:
        if "upgrade torch to at least v2.6" in str(e) or "CVE-2025-32434" in str(e):
            logger.error(
                f"❌ Cannot convert model: PyTorch version {torch.__version__} is too old.\n"
                f"   Solutions:\n"
                f"   1. Upgrade PyTorch temporarily: pip install --upgrade torch>=2.6.0\n"
                f"   2. Download model from HuggingFace Hub with safetensors:\n"
                f"      from huggingface_hub import snapshot_download\n"
                f"      snapshot_download(repo_id='model_id', local_dir='{model_path}', local_dir_use_symlinks=False)"
            )
        else:
            logger.error(f"❌ Failed to convert model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .bin model files to safetensors format"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (defaults to model_path)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_model_to_safetensors(args.model_path, args.output)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

