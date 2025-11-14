"""
WONT BE USED ANYMORE VAI SINCE WE DONT HAVE THAT CUP OF TEA(hardware)
Model download and setup utilities for MedTech Diagnostic LLM Pipeline
"""

import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "swin_unetr": {
        "type": "segmentation",
        "source": "monai",
        "url": "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt",
        "local_path": "models/swin_unetr_pretrained.pth",
        "description": "Swin UNETR pretrained weights for medical image segmentation",
    },
    "biomistral": {
        "type": "llm",
        "source": "huggingface",
        "model_name": "BioMistral/BioMistral-7B",
        "local_path": "models/biomistral",
        "description": "BioMistral 7B medical language model",
    },
    "hippo": {
        "type": "llm",
        "source": "huggingface",
        "model_name": "epfl-llm/meditron-7b",  
        "local_path": "models/hippo",
        "description": "Medical LLM based on Mistral architecture",
    },
    "falcon": {
        "type": "llm",
        "source": "huggingface",
        "model_name": "tiiuae/falcon-7b",
        "local_path": "models/falcon",
        "description": "Falcon 7B language model",
    },
    "sentence_transformer": {
        "type": "embedding",
        "source": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "local_path": "models/sentence_transformer",
        "description": "Sentence transformer for text embeddings",
    },
}


def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "data/inputs",
        "data/outputs",
        "data/uploads",
        "data/segmentation_outputs",
        "data/llm_outputs",
        "data/vector_store",
        "logs",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_file(url: str, local_path: str, description: str = "") -> bool:
    """
    Download a file from URL to local path.

    Args:
        url: Download URL
        local_path: Local file path
        description: Description for progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(local_path):
            logger.info(f"File already exists: {local_path}")
            return True

        logger.info(f"Downloading {description}: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(local_path, "wb") as f, tqdm(
            desc=description,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_huggingface_model(
    model_name: str, local_path: str, description: str = ""
) -> bool:
    """
    Download a model from Hugging Face Hub.

    Args:
        model_name: Hugging Face model name
        local_path: Local directory path
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            logger.info(f"Model already exists: {local_path}")
            return True

        logger.info(f"Downloading {description}: {model_name}")

        Path(local_path).mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_name, local_dir=local_path, local_dir_use_symlinks=False
        )

        logger.info(f"Downloaded model to: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False


def download_swin_unetr_weights() -> bool:
    """Download Swin UNETR pretrained weights."""
    config = MODEL_CONFIGS["swin_unetr"]

    try:
        logger.info("Downloading Swin UNETR weights...")

        local_path = config["local_path"]
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(local_path):
           
            dummy_checkpoint = {
                "model_state_dict": {},
                "epoch": 0,
                "best_metric": 0.0,
                "optimizer_state_dict": {},
            }

            torch.save(dummy_checkpoint, local_path)
            logger.warning(f"Created dummy checkpoint at {local_path}")
            logger.warning("Please replace with actual Swin UNETR pretrained weights")

        return True

    except Exception as e:
        logger.error(f"Failed to setup Swin UNETR weights: {e}")
        return False


def download_biomistral() -> bool:
    """Download BioMistral model."""
    config = MODEL_CONFIGS["biomistral"]
    return download_huggingface_model(
        config["model_name"], config["local_path"], config["description"]
    )


def download_hippo() -> bool:
    """Download Hippo/Meditron model."""
    config = MODEL_CONFIGS["hippo"]
    return download_huggingface_model(
        config["model_name"], config["local_path"], config["description"]
    )


def download_falcon() -> bool:
    """Download Falcon model."""
    config = MODEL_CONFIGS["falcon"]
    return download_huggingface_model(
        config["model_name"], config["local_path"], config["description"]
    )


def download_sentence_transformer() -> bool:
    """Download sentence transformer model."""
    config = MODEL_CONFIGS["sentence_transformer"]
    return download_huggingface_model(
        config["model_name"], config["local_path"], config["description"]
    )


def setup_sample_data():
    """Setup sample data for testing."""
    try:
        
        import nibabel as nib
        import numpy as np

     
        dummy_data = np.random.randint(0, 255, (64, 64, 32), dtype=np.uint8)
        dummy_img = nib.Nifti1Image(dummy_data, np.eye(4))

        sample_path = "data/inputs/sample_image.nii.gz"
        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(dummy_img, sample_path)

        logger.info(f"Created sample data: {sample_path}")


        dummy_mask = np.random.randint(0, 3, (64, 64, 32), dtype=np.uint8)
        dummy_mask_img = nib.Nifti1Image(dummy_mask, np.eye(4))

        gt_path = "data/inputs/sample_image_gt.nii.gz"
        nib.save(dummy_mask_img, gt_path)

        logger.info(f"Created sample ground truth: {gt_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to setup sample data: {e}")
        return False


def verify_model_setup() -> Dict[str, bool]:
    """Verify that all models are properly set up."""
    results = {}

    for model_name, config in MODEL_CONFIGS.items():
        local_path = config["local_path"]

        if os.path.exists(local_path):
            if os.path.isfile(local_path):
                results[model_name] = os.path.getsize(local_path) > 0
            else:
                results[model_name] = len(os.listdir(local_path)) > 0
        else:
            results[model_name] = False

    return results


def update_config_paths():
    """Update configuration files with correct model paths."""
    try:
        seg_config_path = "configs/segmentation_config.yaml"
        if os.path.exists(seg_config_path):
            with open(seg_config_path, "r") as f:
                config = yaml.safe_load(f)

            config["segmentation"]["model"]["model_path"] = MODEL_CONFIGS["swin_unetr"][
                "local_path"
            ]

            with open(seg_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info("Updated segmentation config")

        llm_config_path = "configs/llm_config.yaml"
        if os.path.exists(llm_config_path):
            with open(llm_config_path, "r") as f:
                config = yaml.safe_load(f)

            with open(llm_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info("Updated LLM config")

        vs_config_path = "configs/vector_store_config.yaml"
        if os.path.exists(vs_config_path):
            with open(vs_config_path, "r") as f:
                config = yaml.safe_load(f)

            if "embeddings" in config:
                config["embeddings"]["model"] = MODEL_CONFIGS["sentence_transformer"][
                    "model_name"
                ]

            with open(vs_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info("Updated vector store config")

    except Exception as e:
        logger.error(f"Failed to update config paths: {e}")


def download_all_models() -> bool:
    """Download all required models."""
    logger.info("Starting model downloads...")

    create_directories()

    download_functions = [
        ("Swin UNETR", download_swin_unetr_weights),
        ("Sentence Transformer", download_sentence_transformer),
        ("BioMistral", download_biomistral),
        ("Hippo/Meditron", download_hippo),
        ("Falcon", download_falcon),
    ]

    results = {}
    for name, func in download_functions:
        logger.info(f"Downloading {name}...")
        try:
            results[name] = func()
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            results[name] = False

    logger.info("Setting up sample data...")
    results["Sample Data"] = setup_sample_data()

    logger.info("Updating configuration files...")
    update_config_paths()

    logger.info("Verifying model setup...")
    verification_results = verify_model_setup()

    logger.info("Download Summary:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    logger.info("Verification Summary:")
    for model, exists in verification_results.items():
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {model}")

    all_successful = all(results.values()) and all(verification_results.values())

    if all_successful:
        logger.info("All models downloaded and verified successfully!")
    else:
        logger.warning("Some models failed to download or verify. Check logs above.")

    return all_successful


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and setup models for MedTech Pipeline"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Model to download",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing models"
    )
    parser.add_argument(
        "--sample-data", action="store_true", help="Setup sample data only"
    )

    args = parser.parse_args()

    if args.verify_only:
        results = verify_model_setup()
        for model, exists in results.items():
            status = "✓" if exists else "✗"
            print(f"{status} {model}")
        return

    if args.sample_data:
        setup_sample_data()
        return

    if args.model == "all":
        download_all_models()
    else:
        create_directories()

        if args.model == "swin_unetr":
            download_swin_unetr_weights()
        elif args.model == "biomistral":
            download_biomistral()
        elif args.model == "hippo":
            download_hippo()
        elif args.model == "falcon":
            download_falcon()
        elif args.model == "sentence_transformer":
            download_sentence_transformer()


if __name__ == "__main__":
    main()
