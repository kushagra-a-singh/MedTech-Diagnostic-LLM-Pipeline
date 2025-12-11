"""
Swin UNETR model implementation using MONAI.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            img_size=model_config.get("input_size", (96, 96, 96)),
            spatial_dims=3,
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
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
        input_size = self.config["model"].get("input_size", [96, 96, 96])

        # Build transforms list
        transform_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
        ]

        transform_list.extend(
            [
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=preprocess_config["normalize_params"]["lower"],
                    a_max=preprocess_config["normalize_params"]["upper"],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(
                    keys=["image"], source_key="image", allow_missing_keys=True
                ),
            ]
        )

        # Note: SpatialPadd caused failures for 2D inputs when spatial_size length
        # mismatched. We handle resizing/padding in the fallback preprocessing path
        # to ensure compatibility for both 2D and 3D inputs.

        transform_list.append(ToTensord(keys=["image"]))

        # Note: Orientation and Spacing are skipped here to avoid 2D dimension
        # mismatches. Fallback preprocessing handles resizing/orientation safely
        # for both 2D and 3D inputs.
        transforms = Compose(transform_list)

        return transforms

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocess medical image.

        Args:
            image_path: Path to medical image

        Returns:
            Preprocessed tensor
        """
        try:
            # Route DICOM inputs directly to robust fallback handling to avoid
            # 2D transform mismatches in MONAI pipelines.
            if str(image_path).lower().endswith(".dcm"):
                raise RuntimeError("Use fallback preprocessing for DICOM")

            data = {"image": image_path}
            transformed = self.transforms(data)
            image_tensor = transformed["image"]

            # Ensure we have the right shape: (C, H, W, D) or (C, H, W)
            # Add batch dimension if needed
            if len(image_tensor.shape) == 3:
                # 2D image: (C, H, W) -> (1, C, H, W)
                image_tensor = image_tensor.unsqueeze(0)
            elif len(image_tensor.shape) == 4:
                # 3D image: (C, H, W, D) -> (1, C, H, W, D)
                image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {e}")
            # Try to handle 2D images by converting to 3D
            try:
                import numpy as np
                from monai.data import MetaTensor

                # Load image manually
                from .preprocessing import MedicalImagePreprocessor

                preprocessor = MedicalImagePreprocessor(self.config)
                image_data = preprocessor.load_image(image_path)

                # Normalize to standard shape expectations.
                # Supported inputs: (H, W) 2D; (D, H, W) 3D; with optional channel first.
                arr = image_data
                # If 2D, make it (1, H, W) depth-first
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)  # (1, H, W)
                # If 3D (D, H, W) ensure it's depth-first; if already (H, W, D) swap
                if arr.ndim == 3 and arr.shape[0] not in (1, 2, 3, 4, 8, 16, 32):
                    # Likely (H, W, D): transpose to (D, H, W)
                    arr = np.transpose(arr, (2, 0, 1))

                # Add channel dim at front -> (C=1, D, H, W)
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, axis=0)

                # Convert to tensor: currently (C, D, H, W)
                image_tensor = torch.from_numpy(arr).float()
                # Reorder to (C, H, W, D) for consistent downstream handling
                image_tensor = image_tensor.permute(0, 2, 3, 1)  # (C, H, W, D)
                # If depth is 1 (2D scan), tile to 32 so dims are
                # divisible by 2**5 for SwinUNETR downsampling.
                if image_tensor.shape[-1] == 1:
                    image_tensor = image_tensor.repeat(1, 1, 1, 32)

                # Downsample large inputs to reduce memory footprint while keeping
                # dimensions compatible with SwinUNETR (multiples of 32).
                H, W, D = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
                max_hw = int(self.config.get("model", {}).get("max_hw", 256))
                target_H = min(H, max_hw)
                target_W = min(W, max_hw)
                # enforce multiples of 32 and minimum 32
                target_H = max(32, (target_H // 32) * 32)
                target_W = max(32, (target_W // 32) * 32)
                target_D = max(32, (D // 32) * 32)

                if (H != target_H) or (W != target_W) or (D != target_D):
                    # Current tensor is (C, H, W, D). Convert explicitly to (B, C, D, H, W)
                    x = image_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (B, C, D, H, W)
                    x = F.interpolate(
                        x,
                        size=(target_D, target_H, target_W),
                        mode="trilinear",
                        align_corners=False,
                    )
                    # Back to (C, H, W, D)
                    image_tensor = x.permute(0, 1, 3, 4, 2).squeeze(0)

                # Add batch dimension -> (B, C, H, W, D)
                return image_tensor.unsqueeze(0).to(self.device)
            except Exception as fallback_error:
                logger.error(f"Fallback preprocessing also failed: {fallback_error}")
                raise e

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
            spatial = list(image_tensor.shape[-3:])  # (H, W, D)
            # Choose ROI capped by config and enforce multiples of 32
            roi_size = [
                max(32, ((min(spatial[i], input_size[i]) // 32) * 32))
                for i in range(3)
            ]
            if any(spatial[i] > roi_size[i] for i in range(3)):
                sw_bs = int(inference_config.get("sw_batch_size", 1))
                # On CPU, keep sw_batch_size small to avoid OOM
                if str(self.device) == "cpu":
                    sw_bs = 1
                overlap = float(inference_config.get("overlap", 0.25))
                prediction = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=roi_size,
                    sw_batch_size=sw_bs,
                    overlap=overlap,
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

        # Remove batch dimension if present
        if len(mask.shape) == 4:
            mask = mask[0]  # (B, H, W, D) -> (H, W, D)

        # Ensure mask axes order matches original image data (D, H, W)
        if len(mask.shape) == 3:
            mask = np.transpose(mask, (2, 0, 1))  # (H, W, D) -> (D, H, W)

        # Apply postprocessing
        if postprocess_config["connected_component"]:
            mask = self._apply_connected_components(mask)

        if postprocess_config["fill_holes"]:
            mask = self._fill_holes(mask)

        if postprocess_config["remove_small_objects"]:
            mask = self._remove_small_objects(mask, postprocess_config["min_size"])

        # Handle 2D vs 3D shape mismatch
        # If original was 2D but mask is 3D, remove depth dimension
        if len(original_shape) == 2 and len(mask.shape) == 3:
            # Take middle slice or first slice
            mask = mask[0] if mask.shape[0] == 1 else mask[mask.shape[0] // 2]
        elif len(original_shape) == 3 and len(mask.shape) == 2:
            # Add depth dimension back
            mask = np.expand_dims(mask, axis=0)

        # Resize to original shape if needed
        if mask.shape != original_shape:
            mask = self._resize_mask(mask, original_shape)

        return mask

    def _apply_connected_components(self, mask: np.ndarray) -> np.ndarray:
        """Apply connected component analysis."""
        from scipy import ndimage

        processed_mask = np.zeros_like(mask)

        # Determine structure based on mask dimensions
        if len(mask.shape) == 2:
            structure = np.ones((3, 3))
        else:
            structure = np.ones((3, 3, 3))

        for class_id in np.unique(mask):
            if class_id == 0:  # Background
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            labeled, num_features = ndimage.label(class_mask, structure=structure)
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
            # binary_fill_holes works for both 2D and 3D
            filled_mask = ndimage.binary_fill_holes(class_mask)
            processed_mask[filled_mask] = class_id

        return processed_mask

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small objects from segmentation mask."""
        from scipy import ndimage

        processed_mask = np.zeros_like(mask)

        # Determine structure based on mask dimensions
        if len(mask.shape) == 2:
            structure = np.ones((3, 3))
        else:
            structure = np.ones((3, 3, 3))

        for class_id in np.unique(mask):
            if class_id == 0:  # Background
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            cleaned_mask = ndimage.binary_opening(class_mask, structure=structure)
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
            features_list = self.model.swinViT(image_tensor, normalize=True)
            # Use the last feature map (deepest representation)
            embedding_map = features_list[-1]
            
            # Global Average Pooling to get (B, C)
            if len(embedding_map.shape) == 5:
                # (B, C, D, H, W) -> (B, C)
                embeddings = torch.mean(embedding_map, dim=(2, 3, 4))
            elif len(embedding_map.shape) == 4:
                # (B, C, H, W) -> (B, C)
                embeddings = torch.mean(embedding_map, dim=(2, 3))
            else:
                # Fallback, just flatten
                embeddings = torch.flatten(embedding_map, start_dim=1)

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
