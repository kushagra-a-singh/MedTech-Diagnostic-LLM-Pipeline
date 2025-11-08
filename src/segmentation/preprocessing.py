"""
Medical image preprocessing utilities.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Medical image preprocessing for DICOM and NIfTI files.
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get("preprocessing", {})

        logger.info("Medical image preprocessor initialized")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load medical image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image data as numpy array
        """
        try:
            image_path = Path(image_path)

            if image_path.suffix.lower() in [".nii", ".gz"]:
              
                return self._load_nifti(str(image_path))
            elif image_path.suffix.lower() == ".dcm":
               
                return self._load_dicom(str(image_path))
            elif image_path.is_dir():
               
                return self._load_dicom_series(str(image_path))
            else:
                raise ValueError(f"Unsupported file format: {image_path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def _load_nifti(self, file_path: str) -> np.ndarray:
        """Load NIfTI file."""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            logger.info(f"Loaded NIfTI image: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load NIfTI file {file_path}: {e}")
            raise

    def _load_dicom(self, file_path: str) -> np.ndarray:
        """Load single DICOM file."""
        try:
            ds = pydicom.dcmread(file_path)
            data = ds.pixel_array

            # Apply rescale slope and intercept if available
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                data = data * ds.RescaleSlope + ds.RescaleIntercept

            logger.info(f"Loaded DICOM image: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load DICOM file {file_path}: {e}")
            raise

    def _load_dicom_series(self, directory_path: str) -> np.ndarray:
        """Load DICOM series from directory."""
        try:
            # Use SimpleITK for series reading
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(directory_path)

            if not dicom_names:
                raise ValueError(f"No DICOM files found in {directory_path}")

            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            data = sitk.GetArrayFromImage(image)

            logger.info(f"Loaded DICOM series: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load DICOM series from {directory_path}: {e}")
            raise

    def preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """
        Preprocess image data.

        Args:
            image_data: Raw image data

        Returns:
            Preprocessed image data
        """
        try:
            processed_data = image_data.copy()

            if self.preprocessing_config.get("normalize", True):
                processed_data = self._normalize_image(processed_data)
            # Additional preprocessing steps can be added here
            return processed_data

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise

    def _normalize_image(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image intensity."""
        try:
            normalize_mode = self.preprocessing_config.get(
                "normalize_mode", "percentile"
            )

            if normalize_mode == "percentile":
                params = self.preprocessing_config.get(
                    "normalize_params", {"lower": 1.0, "upper": 99.0}
                )
                lower_percentile = np.percentile(image_data, params["lower"])
                upper_percentile = np.percentile(image_data, params["upper"])

                # Clip and normalize to [0, 1]
                normalized = np.clip(image_data, lower_percentile, upper_percentile)
                normalized = (normalized - lower_percentile) / (
                    upper_percentile - lower_percentile
                )

            elif normalize_mode == "zscore":
                mean = np.mean(image_data)
                std = np.std(image_data)
                normalized = (image_data - mean) / (std + 1e-8)

            elif normalize_mode == "minmax":
                min_val = np.min(image_data)
                max_val = np.max(image_data)
                normalized = (image_data - min_val) / (max_val - min_val + 1e-8)

            else:
                raise ValueError(f"Unsupported normalization mode: {normalize_mode}")

            return normalized

        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            return image_data

    def save_mask(
        self, mask: np.ndarray, output_path: str, reference_image: Optional[str] = None
    ):
        """
        Save segmentation mask.

        Args:
            mask: Segmentation mask
            output_path: Output file path
            reference_image: Reference image for header information
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() in [".nii", ".gz"]:
             
                if reference_image and Path(reference_image).suffix.lower() in [
                    ".nii",
                    ".gz",
                ]:
                    # Use reference image header
                    ref_img = nib.load(reference_image)
                    mask_img = nib.Nifti1Image(
                        mask.astype(np.int16), ref_img.affine, ref_img.header
                    )
                else:
                    # Create new header
                    mask_img = nib.Nifti1Image(mask.astype(np.int16), np.eye(4))

                nib.save(mask_img, str(output_path))

            else:
                
                np.save(str(output_path), mask)

            logger.info(f"Saved mask to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save mask to {output_path}: {e}")
            raise

    def get_image_metadata(self, image_path: str) -> Dict:
        """
        Extract metadata from medical image.

        Args:
            image_path: Path to image file

        Returns:
            Metadata dictionary
        """
        try:
            metadata = {
                "file_path": image_path,
                "file_size": os.path.getsize(image_path),
                "modality": "unknown",
                "patient_id": "unknown",
                "study_date": "unknown",
                "series_description": "unknown",
            }

            image_path = Path(image_path)

            if image_path.suffix.lower() in [".nii", ".gz"]:
                # NIfTI metadata
                img = nib.load(str(image_path))
                metadata.update(
                    {
                        "shape": img.shape,
                        "voxel_size": img.header.get_zooms(),
                        "data_type": str(img.get_data_dtype()),
                    }
                )

            elif image_path.suffix.lower() == ".dcm":
                # DICOM metadata
                ds = pydicom.dcmread(str(image_path))
                metadata.update(
                    {
                        "modality": getattr(ds, "Modality", "unknown"),
                        "patient_id": getattr(ds, "PatientID", "unknown"),
                        "study_date": getattr(ds, "StudyDate", "unknown"),
                        "series_description": getattr(
                            ds, "SeriesDescription", "unknown"
                        ),
                        "shape": (
                            ds.pixel_array.shape if hasattr(ds, "pixel_array") else None
                        ),
                    }
                )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from {image_path}: {e}")
            return {"file_path": image_path, "error": str(e)}
