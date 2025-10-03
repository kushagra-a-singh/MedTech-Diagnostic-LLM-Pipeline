"""
Segmentation postprocessing utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import ndimage
from skimage import morphology, measure

logger = logging.getLogger(__name__)


class SegmentationPostprocessor:
    """
    Postprocessing utilities for segmentation masks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize postprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.postprocessing_config = config.get("postprocessing", {})
        
        logger.info("Segmentation postprocessor initialized")
    
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing to segmentation mask.
        
        Args:
            mask: Raw segmentation mask
            
        Returns:
            Postprocessed mask
        """
        try:
            processed_mask = mask.copy()
            
            # Apply connected component analysis
            if self.postprocessing_config.get("connected_component", True):
                processed_mask = self.apply_connected_components(processed_mask)
            
            # Fill holes
            if self.postprocessing_config.get("fill_holes", True):
                processed_mask = self.fill_holes(processed_mask)
            
            # Remove small objects
            if self.postprocessing_config.get("remove_small_objects", True):
                min_size = self.postprocessing_config.get("min_size", 100)
                processed_mask = self.remove_small_objects(processed_mask, min_size)
            
            # Smooth boundaries
            if self.postprocessing_config.get("smooth_boundaries", False):
                processed_mask = self.smooth_boundaries(processed_mask)
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Failed to postprocess mask: {e}")
            return mask
    
    def apply_connected_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply connected component analysis to keep largest components.
        
        Args:
            mask: Input mask
            
        Returns:
            Processed mask with largest components
        """
        try:
            processed_mask = np.zeros_like(mask)
            
            # Process each class separately
            for class_id in np.unique(mask):
                if class_id == 0:  
                    continue
                
                # Extract class mask
                class_mask = (mask == class_id).astype(np.uint8)
                
                # Label connected components
                labeled, num_features = ndimage.label(class_mask)
                
                if num_features > 0:
                    # Find largest component
                    component_sizes = ndimage.sum(class_mask, labeled, range(1, num_features + 1))
                    largest_component = np.argmax(component_sizes) + 1
                    
                    # Keep only largest component
                    largest_mask = (labeled == largest_component)
                    processed_mask[largest_mask] = class_id
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Failed to apply connected components: {e}")
            return mask
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in segmentation mask.
        
        Args:
            mask: Input mask
            
        Returns:
            Mask with filled holes
        """
        try:
            processed_mask = np.zeros_like(mask)
            
            # Process each class separately
            for class_id in np.unique(mask):
                if class_id == 0:  # Skip background
                    continue
                
                # Extract class mask
                class_mask = (mask == class_id).astype(bool)
                
                # Fill holes
                filled_mask = ndimage.binary_fill_holes(class_mask)
                
                # Update processed mask
                processed_mask[filled_mask] = class_id
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Failed to fill holes: {e}")
            return mask
    
    def remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """
        Remove small objects from segmentation mask.
        
        Args:
            mask: Input mask
            min_size: Minimum object size in voxels
            
        Returns:
            Mask with small objects removed
        """
        try:
            processed_mask = np.zeros_like(mask)
            
            # Process each class separately
            for class_id in np.unique(mask):
                if class_id == 0:  # Skip background
                    continue
                
                # Extract class mask
                class_mask = (mask == class_id).astype(bool)
                
                # Remove small objects
                cleaned_mask = morphology.remove_small_objects(
                    class_mask, min_size=min_size, connectivity=1
                )
                
                # Update processed mask
                processed_mask[cleaned_mask] = class_id
            
            return processed_mask
            
        except Exception as e:
            logger.error(f"Failed to remove small objects: {e}")
            return mask
    
    def smooth_boundaries(self, mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Smooth mask boundaries using Gaussian filtering.
        
        Args:
            mask: Input mask
            sigma: Gaussian kernel sigma
            
        Returns:
            Mask with smoothed boundaries
        """
        try:
            processed_mask = np.zeros_like(mask)
            
            # Process each class separately
            for class_id in np.unique(mask):
                if class_id == 0:  # Skip background
                    continue
                
                # Extract class mask
                class_mask = (mask == class_id).astype(float)
                
                # Apply Gaussian smoothing
                smoothed = ndimage.gaussian_filter(class_mask, sigma=sigma)
                
                # Threshold to get binary mask
                smoothed_mask = smoothed > 0.5
                
                # Update processed mask
                processed_mask[smoothed_mask] = class_id
            
            return processed_mask.astype(mask.dtype)
            
        except Exception as e:
            logger.error(f"Failed to smooth boundaries: {e}")
            return mask
    
    def calculate_volume_statistics(self, mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> Dict:
        """
        Calculate volume statistics for each class.
        
        Args:
            mask: Segmentation mask
            voxel_size: Voxel size in mm (x, y, z)
            
        Returns:
            Volume statistics dictionary
        """
        try:
            voxel_volume = np.prod(voxel_size)  # Volume of single voxel in mm³
            
            statistics = {}
            
            for class_id in np.unique(mask):
                if class_id == 0:  # Skip background
                    continue
                
                # Count voxels for this class
                voxel_count = np.sum(mask == class_id)
                
                # Calculate volume in mm³ and cm³
                volume_mm3 = voxel_count * voxel_volume
                volume_cm3 = volume_mm3 / 1000.0
                
                statistics[int(class_id)] = {
                    "voxel_count": int(voxel_count),
                    "volume_mm3": float(volume_mm3),
                    "volume_cm3": float(volume_cm3)
                }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to calculate volume statistics: {e}")
            return {}
    
    def calculate_surface_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray, 
                                 voxel_size: Tuple[float, float, float]) -> Dict:
        """
        Calculate surface-based metrics.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            voxel_size: Voxel size in mm
            
        Returns:
            Surface metrics dictionary
        """
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            metrics = {}
            
            for class_id in np.unique(np.concatenate([np.unique(prediction), np.unique(ground_truth)])):
                if class_id == 0:  # Skip background
                    continue
                
                # Extract class masks
                pred_mask = (prediction == class_id)
                gt_mask = (ground_truth == class_id)
                
                if not np.any(pred_mask) or not np.any(gt_mask):
                    continue
                
                # Get surface points
                pred_surface = self._get_surface_points(pred_mask, voxel_size)
                gt_surface = self._get_surface_points(gt_mask, voxel_size)
                
                if len(pred_surface) == 0 or len(gt_surface) == 0:
                    continue
                
                # Calculate Hausdorff distance
                hausdorff_dist = max(
                    directed_hausdorff(pred_surface, gt_surface)[0],
                    directed_hausdorff(gt_surface, pred_surface)[0]
                )
                
                # Calculate average surface distance
                avg_surface_dist = self._calculate_average_surface_distance(pred_surface, gt_surface)
                
                metrics[int(class_id)] = {
                    "hausdorff_distance": float(hausdorff_dist),
                    "average_surface_distance": float(avg_surface_dist)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate surface metrics: {e}")
            return {}
    
    def _get_surface_points(self, mask: np.ndarray, voxel_size: Tuple[float, float, float]) -> np.ndarray:
        """Extract surface points from binary mask."""
        try:
            # Find surface using morphological operations
            eroded = ndimage.binary_erosion(mask)
            surface = mask & ~eroded
            
            # Get coordinates of surface points
            surface_coords = np.where(surface)
            
            # Convert to physical coordinates
            surface_points = np.column_stack([
                surface_coords[0] * voxel_size[0],
                surface_coords[1] * voxel_size[1], 
                surface_coords[2] * voxel_size[2]
            ])
            
            return surface_points
            
        except Exception as e:
            logger.error(f"Failed to extract surface points: {e}")
            return np.array([])
    
    def _calculate_average_surface_distance(self, surface1: np.ndarray, surface2: np.ndarray) -> float:
        """Calculate average surface distance between two surfaces."""
        try:
            from scipy.spatial.distance import cdist
            
            # Calculate distances from surface1 to surface2
            distances1 = np.min(cdist(surface1, surface2), axis=1)
            
            # Calculate distances from surface2 to surface1
            distances2 = np.min(cdist(surface2, surface1), axis=1)
            
            # Average surface distance
            avg_dist = (np.mean(distances1) + np.mean(distances2)) / 2.0
            
            return avg_dist
            
        except Exception as e:
            logger.error(f"Failed to calculate average surface distance: {e}")
            return 0.0
    
    def generate_quality_report(self, mask: np.ndarray, original_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Generate quality assessment report for processed mask.
        
        Args:
            mask: Processed mask
            original_mask: Original mask before postprocessing (optional)
            
        Returns:
            Quality report dictionary
        """
        try:
            report = {
                "mask_shape": mask.shape,
                "num_classes": len(np.unique(mask)) - 1,  # Exclude background
                "class_distribution": {},
                "postprocessing_effects": {}
            }
            
            # Class distribution
            for class_id in np.unique(mask):
                if class_id == 0:
                    continue
                voxel_count = np.sum(mask == class_id)
                report["class_distribution"][int(class_id)] = int(voxel_count)
            
            # Compare with original if provided
            if original_mask is not None:
                report["postprocessing_effects"] = self._compare_masks(original_mask, mask)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {"error": str(e)}
    
    def _compare_masks(self, original: np.ndarray, processed: np.ndarray) -> Dict:
        """Compare original and processed masks."""
        try:
            comparison = {}
            
            for class_id in np.unique(np.concatenate([np.unique(original), np.unique(processed)])):
                if class_id == 0:
                    continue
                
                orig_count = np.sum(original == class_id)
                proc_count = np.sum(processed == class_id)
                
                comparison[int(class_id)] = {
                    "original_voxels": int(orig_count),
                    "processed_voxels": int(proc_count),
                    "change_ratio": float(proc_count / orig_count) if orig_count > 0 else 0.0
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare masks: {e}")
            return {}