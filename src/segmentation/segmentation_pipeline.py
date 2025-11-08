"""
Main segmentation pipeline for medical imaging.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .postprocessing import SegmentationPostprocessor
from .preprocessing import MedicalImagePreprocessor
from .swin_unetr_model import SwinUNETRModel

logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """
    Main segmentation pipeline that orchestrates the entire segmentation process.
    """

    def __init__(self, config_path: str):
        """
        Initialize segmentation pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = SwinUNETRModel(self.config)
        self.preprocessor = MedicalImagePreprocessor(self.config)
        self.postprocessor = SegmentationPostprocessor(self.config)

        os.makedirs(self.config["output"]["output_dir"], exist_ok=True)

        logger.info("Segmentation pipeline initialized")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config["segmentation"]

    def process_single_image(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process a single medical image through the segmentation pipeline.

        Args:
            image_path: Path to input image
            output_dir: Output directory (optional)

        Returns:
            Dictionary containing results
        """
        logger.info(f"Processing image: {image_path}")

        if output_dir is None:
            output_dir = self.config["output"]["output_dir"]

        image_data = self.preprocessor.load_image(image_path)
    
        preprocessed_tensor = self.model.preprocess(image_path)

        prediction = self.model.predict(preprocessed_tensor)

        original_shape = image_data.shape
        segmentation_mask = self.model.postprocess(prediction, original_shape)

        embeddings = self.model.extract_embeddings(preprocessed_tensor)

        results = self._save_results(
            image_path, segmentation_mask, embeddings, output_dir
        )

        metrics = self._calculate_metrics(segmentation_mask, image_path)

        results.update({"metrics": metrics, "model_info": self.model.get_model_info()})

        logger.info(f"Completed processing: {image_path}")
        return results

    def process_batch(
        self, image_paths: List[str], output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a batch of medical images.

        Args:
            image_paths: List of image paths
            output_dir: Output directory (optional)

        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing batch of {len(image_paths)} images")

        results = []
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.process_single_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append(
                    {"image_path": image_path, "error": str(e), "status": "failed"}
                )

        if output_dir:
            self._save_batch_summary(results, output_dir)

        logger.info(
            f"Completed batch processing. Success: {len([r for r in results if 'error' not in r])}/{len(image_paths)}"
        )
        return results

    def _save_results(
        self,
        image_path: str,
        segmentation_mask: np.ndarray,
        embeddings: torch.Tensor,
        output_dir: str,
    ) -> Dict:
        """
        Save segmentation results.

        Args:
            image_path: Original image path
            segmentation_mask: Segmentation mask
            embeddings: Feature embeddings
            output_dir: Output directory

        Returns:
            Results dictionary
        """
      
        base_name = Path(image_path).stem
        mask_path = os.path.join(output_dir, f"{base_name}_mask.nii.gz")
        embedding_path = os.path.join(output_dir, f"{base_name}_embeddings.npy")
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")

        self.preprocessor.save_mask(segmentation_mask, mask_path)

        np.save(embedding_path, embeddings.detach().cpu().numpy())

        metadata = {
            "image_path": image_path,
            "mask_path": mask_path,
            "embedding_path": embedding_path,
            "mask_shape": segmentation_mask.shape,
            "embedding_shape": embeddings.shape,
            "classes": self._get_class_names(),
            "processing_timestamp": str(np.datetime64("now")),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "embedding_path": embedding_path,
            "metadata_path": metadata_path,
            "status": "success",
        }

    def _calculate_metrics(
        self, segmentation_mask: np.ndarray, image_path: str
    ) -> Dict:
        """
        Calculate segmentation metrics.

        Args:
            segmentation_mask: Segmentation mask
            image_path: Image path (for finding ground truth)

        Returns:
            Metrics dictionary
        """
        metrics = {}

        unique_classes = np.unique(segmentation_mask)
        metrics["num_classes"] = len(unique_classes)
        metrics["class_distribution"] = {
            int(cls): int(np.sum(segmentation_mask == cls)) for cls in unique_classes
        }

        # Calculate volume statistics
        voxel_volume = np.prod(self.config["preprocessing"]["spacing"])
        metrics["volumes"] = {
            int(cls): float(np.sum(segmentation_mask == cls) * voxel_volume)
            for cls in unique_classes
            if cls != 0  # Exclude background
        }

        # Try to find ground truth for comparison
        ground_truth_path = self._find_ground_truth(image_path)
        if ground_truth_path:
            try:
                ground_truth = self.preprocessor.load_image(ground_truth_path)
                dice_scores = self._calculate_dice_scores(
                    segmentation_mask, ground_truth
                )
                metrics["dice_scores"] = dice_scores
                metrics["mean_dice"] = np.mean(list(dice_scores.values()))
            except Exception as e:
                logger.warning(f"Could not calculate Dice scores: {e}")

        return metrics

    def _find_ground_truth(self, image_path: str) -> Optional[str]:
        """
        Find ground truth file for given image.

        Args:
            image_path: Image path

        Returns:
            Ground truth path if found
        """
        # Common ground truth naming patterns
        base_path = Path(image_path)
        possible_gt_paths = [
            base_path.parent / f"{base_path.stem}_gt.nii.gz",
            base_path.parent / f"{base_path.stem}_label.nii.gz",
            base_path.parent / f"{base_path.stem}_mask.nii.gz",
            base_path.parent / "labels" / f"{base_path.stem}.nii.gz",
        ]

        for gt_path in possible_gt_paths:
            if gt_path.exists():
                return str(gt_path)

        return None

    def _calculate_dice_scores(
        self, prediction: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[int, float]:
        """
        Calculate Dice scores for each class.

        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask

        Returns:
            Dictionary of Dice scores per class
        """
        dice_scores = {}

        if prediction.shape != ground_truth.shape:
            logger.warning("Prediction and ground truth have different shapes")
            return dice_scores

        for class_id in np.unique(
            np.concatenate([np.unique(prediction), np.unique(ground_truth)])
        ):
            if class_id == 0:  # Skip background
                continue

            pred_mask = (prediction == class_id).astype(np.uint8)
            gt_mask = (ground_truth == class_id).astype(np.uint8)

            intersection = np.sum(pred_mask * gt_mask)
            union = np.sum(pred_mask) + np.sum(gt_mask)

            if union > 0:
                dice = 2.0 * intersection / union
                dice_scores[int(class_id)] = float(dice)
            else:
                dice_scores[int(class_id)] = 0.0

        return dice_scores

    def _get_class_names(self) -> Dict[int, str]:
        """
        Get class names for segmentation.

        Returns:
            Dictionary mapping class IDs to names
        """
       
        class_names = {
            0: "background",
            1: "liver",
            2: "kidney",
            3: "spleen",
            4: "pancreas",
            5: "aorta",
            6: "inferior_vena_cava",
            7: "portal_vein",
            8: "hepatic_vein",
            9: "gallbladder",
            10: "esophagus",
            11: "stomach",
            12: "duodenum",
            13: "colon",
        }

        return class_names

    def _save_batch_summary(self, results: List[Dict], output_dir: str):
        """
        Save batch processing summary.

        Args:
            results: List of processing results
            output_dir: Output directory
        """
        summary = {
            "total_images": len(results),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "results": results,
            "timestamp": str(np.datetime64("now")),
        }

        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Batch summary saved to: {summary_path}")

    def get_pipeline_info(self) -> Dict:
        """
        Get pipeline information.

        Returns:
            Pipeline information dictionary
        """
        return {
            "pipeline_type": "Medical Image Segmentation",
            "model_info": self.model.get_model_info(),
            "config": self.config,
            "output_directory": self.config["output"]["output_dir"],
        }
