#!/usr/bin/env python3
"""
Script to download a model from HuggingFace Hub in safetensors format.
This avoids PyTorch version restrictions for .bin files.
"""

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_model_safetensors(repo_id: str, local_dir: str, token: str = None):
    """
    Download a model from HuggingFace Hub in safetensors format.
    
    Args:
        repo_id: HuggingFace model repository ID
        local_dir: Local directory to save the model
        token: HuggingFace token (optional, for private models)
    """
    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model {repo_id} to {local_dir}...")
    logger.info("This will download the model in safetensors format to avoid PyTorch version restrictions.")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
            ignore_patterns=["*.bin"],  # Skip .bin files, prefer safetensors
        )
        logger.info(f"✅ Model downloaded successfully to {local_dir}")
        
        # Check for safetensors files
        safetensors_files = list(local_dir.glob("*.safetensors"))
        if safetensors_files:
            logger.info(f"   Found safetensors files: {[f.name for f in safetensors_files]}")
        else:
            logger.warning("   No safetensors files found. The model might only be available in .bin format.")
            logger.warning("   You may need to upgrade PyTorch to 2.6+ or convert the model manually.")
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace Hub in safetensors format"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace model repository ID (e.g., 'mistralai/Mistral-7B-v0.1')"
    )
    parser.add_argument(
        "local_dir",
        type=str,
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (for private models). Can also set HF_TOKEN environment variable."
    )
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    try:
        download_model_safetensors(args.repo_id, args.local_dir, token)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

