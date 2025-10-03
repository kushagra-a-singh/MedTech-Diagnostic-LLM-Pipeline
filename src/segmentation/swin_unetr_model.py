"""
Swin UNETR model implementation using MONAI.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

logger = logging.getLogger(__name__)


class SwinUNETRModel:
    """
    Swin UNETR model wrapper for medical image segmentation.
    """

    def __init__(self, config: Dict):
        """
        Initialize Swin UNETR model.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get("hardware", {}).get("device", "cuda"))

        self.model = self._create_model()
        self.model.to(self.device)

        self.transforms = self._create_transforms()

        logger.info(f"Swin UNETR model initialized on device: {self.device}")

    def _create_model(self) -> SwinUNETR:
        """
        Create Swin UNETR model based on configuration.

        Returns:
            Swin UNETR model
        """
        model_config = self.config["model"]

        model = SwinUNETR(
            spatial_dims=3,
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            patch_size=model_config.get("patch_size", 2),
            depths=model_config.get("depths", [2, 2, 2, 2]),
            num_heads=model_config.get("num_heads", [3, 6, 12, 24]),
            feature_size=model_config["feature_size"],
            drop_rate=model_config["drop_rate"],
            attn_drop_rate=model_config["attn_drop_rate"],
            dropout_path_rate=model_config["dropout_path_rate"],
            use_checkpoint=model_config["use_checkpoint"],
        )

        if model_config.get("pretrained", False) and model_config.get("model_path"):
            model_path = Path(model_config["model_path"])
            if model_path.exists():
                try:
                    map_location = (
                        "cpu" if not torch.cuda.is_available() else self.device
                    )
                    checkpoint = torch.load(
                        model_config["model_path"], map_location=map_location
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(
                        f"Loaded pretrained weights from {model_config['model_path']}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
            else:
                logger.warning(
                    f"Pretrained model file not found: {model_path}. Using random weights."
                )

        return model

    def _create_transforms(self) -> Compose:
        """
        Create preprocessing transforms.

        Returns:
            Compose transform
        """
        preprocess_config = self.config["preprocessing"]

        transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes=preprocess_config["orientation"]),
                Spacingd(
                    keys=["image"],
                    pixdim=preprocess_config["spacing"],
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=preprocess_config["normalize_params"]["lower"],
                    a_max=preprocess_config["normalize_params"]["upper"],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                SpatialPadd(
                    keys=["image"],
                    spatial_size=self.config["model"].get("input_size", [96, 96, 96]),
                ),
                ToTensord(keys=["image"]),
            ]
        )

        return transforms

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocess medical image.

        Args:
            image_path: Path to medical image

        Returns:
            Preprocessed tensor
        """
        data = {"image": image_path}
        transformed = self.transforms(data)
        return transformed["image"].unsqueeze(0).to(self.device)

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on preprocessed image.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Segmentation prediction
        """
        self.model.eval()

        inference_config = self.config["inference"]

        with torch.no_grad():
            # Use sliding window inference for large volumes
            input_size = self.config["model"].get("input_size", [96, 96, 96])
            if image_tensor.shape[-1] > max(input_size):
                prediction = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=input_size,
                    sw_batch_size=inference_config["sw_batch_size"],
                    sw_overlap=inference_config["sw_overlap"],
                    mode=inference_config["mode"],
                    predictor=self.model,
                )
            else:
                prediction = self.model(image_tensor)

        return prediction

    def postprocess(
        self, prediction: torch.Tensor, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Postprocess segmentation prediction.

        Args:
            prediction: Model prediction
            original_shape: Original image shape

        Returns:
            Postprocessed segmentation mask
        """
        postprocess_config = self.config["postprocessing"]

        prediction = prediction.cpu().numpy()

        # Apply softmax to get probabilities
        prediction = torch.softmax(torch.from_numpy(prediction), dim=1).numpy()

        # Get class predictions
        mask = np.argmax(prediction, axis=1)

        # Apply postprocessing
        if postprocess_config["connected_component"]:
            mask = self._apply_connected_components(mask)

        if postprocess_config["fill_holes"]:
            mask = self._fill_holes(mask)

        if postprocess_config["remove_small_objects"]:
            mask = self._remove_small_objects(mask, postprocess_config["min_size"])

        # Resize to original shape if needed
        if mask.shape != original_shape:
            mask = self._resize_mask(mask, original_shape)

        return mask

    def _apply_connected_components(self, mask: np.ndarray) -> np.ndarray:
        """Apply connected component analysis."""
        from scipy import ndimage

        processed_mask = np.zeros_like(mask)
        for class_id in np.unique(mask):
            if class_id == 0:  # Background
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            labeled, num_features = ndimage.label(class_mask)
            if num_features > 0:
                # Keep largest component
                sizes = ndimage.sum(class_mask, labeled, range(1, num_features + 1))
                largest_component = np.where(sizes == np.max(sizes))[0][0] + 1
                processed_mask[labeled == largest_component] = class_id

        return processed_mask

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in segmentation mask."""
        from scipy import ndimage

        processed_mask = np.zeros_like(mask)
        for class_id in np.unique(mask):
            if class_id == 0:  # Background
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            filled_mask = ndimage.binary_fill_holes(class_mask)
            processed_mask[filled_mask] = class_id

        return processed_mask

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small objects from segmentation mask."""
        from scipy import ndimage

        processed_mask = np.zeros_like(mask)
        for class_id in np.unique(mask):
            if class_id == 0:  # Background
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            cleaned_mask = ndimage.binary_opening(
                class_mask, structure=np.ones((3, 3, 3))
            )
            processed_mask[cleaned_mask] = class_id

        return processed_mask

    def _resize_mask(
        self, mask: np.ndarray, target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Resize mask to target shape."""
        from scipy.ndimage import zoom

        zoom_factors = [target_shape[i] / mask.shape[i] for i in range(len(mask.shape))]
        resized_mask = zoom(
            mask, zoom_factors, order=0
        )  # Nearest neighbor interpolation

        return resized_mask

    def extract_embeddings(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the model.

        Args:
            image_tensor: Preprocessed image tensor

        Returns:
            Feature embeddings
        """
        self.model.eval()

        with torch.no_grad():
            embeddings = self.model.encoder(image_tensor)

        return embeddings

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_type": "SwinUNETR",
            "input_size": self.config["model"].get("input_size", [96, 96, 96]),
            "in_channels": self.config["model"]["in_channels"],
            "out_channels": self.config["model"]["out_channels"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }
