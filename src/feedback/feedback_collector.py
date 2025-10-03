"""
Expert feedback collection and processing for model improvement.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collect and process expert feedback for model improvement.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feedback collector.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.feedback_dir = Path(self.config.get("feedback_dir", "data/feedback"))
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback_data = self._load_feedback()
        
        logger.info("Feedback collector initialized")
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback data."""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
            return []
    
    def _save_feedback(self):
        """Save feedback data to file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2, default=str)
            logger.info(f"Saved feedback to {self.feedback_file}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def add_feedback(self, result_id: str, feedback: Dict[str, Any]):
        """
        Add expert feedback for a specific result.
        
        Args:
            result_id: Unique identifier for the result
            feedback: Feedback dictionary
        """
        try:
            feedback_entry = {
                "result_id": result_id,
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "feedback_id": len(self.feedback_data)
            }
            
            validated_feedback = self._validate_feedback(feedback)
            feedback_entry["feedback"] = validated_feedback
            
            self.feedback_data.append(feedback_entry)
            self._save_feedback()
            
            logger.info(f"Added feedback for result {result_id}")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
    
    def _validate_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and structure feedback data.
        
        Args:
            feedback: Raw feedback dictionary
            
        Returns:
            Validated feedback dictionary
        """
        validated = {
            "overall_quality": feedback.get("overall_quality", 0),  # 1-5 scale
            "segmentation_quality": feedback.get("segmentation_quality", 0),
            "report_quality": feedback.get("report_quality", 0),
            "clinical_accuracy": feedback.get("clinical_accuracy", 0),
            "comments": feedback.get("comments", ""),
            "corrections": feedback.get("corrections", {}),
            "expert_id": feedback.get("expert_id", "anonymous"),
            "expertise_level": feedback.get("expertise_level", "unknown")
        }
        
        if "corrections" in feedback:
            validated["corrections"] = self._validate_corrections(feedback["corrections"])
        
        return validated
    
    def _validate_corrections(self, corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Validate corrections structure."""
        validated_corrections = {}
        
        if "segmentation" in corrections:
            validated_corrections["segmentation"] = {
                "corrected_mask_path": corrections["segmentation"].get("corrected_mask_path"),
                "class_corrections": corrections["segmentation"].get("class_corrections", {}),
                "boundary_corrections": corrections["segmentation"].get("boundary_corrections", [])
            }
        
        if "report" in corrections:
            validated_corrections["report"] = {
                "corrected_text": corrections["report"].get("corrected_text", ""),
                "section_corrections": corrections["report"].get("section_corrections", {}),
                "terminology_corrections": corrections["report"].get("terminology_corrections", [])
            }
        
        if "clinical_context" in corrections:
            validated_corrections["clinical_context"] = corrections["clinical_context"]
        
        return validated_corrections
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected feedback.
        
        Returns:
            Feedback summary dictionary
        """
        try:
            if not self.feedback_data:
                return {"total_feedback": 0, "message": "No feedback collected"}
            
            df = pd.DataFrame([f["feedback"] for f in self.feedback_data])
            
            summary = {
                "total_feedback": len(self.feedback_data),
                "average_ratings": {
                    "overall_quality": df["overall_quality"].mean(),
                    "segmentation_quality": df["segmentation_quality"].mean(),
                    "report_quality": df["report_quality"].mean(),
                    "clinical_accuracy": df["clinical_accuracy"].mean()
                },
                "rating_distribution": {
                    "overall_quality": df["overall_quality"].value_counts().to_dict(),
                    "segmentation_quality": df["segmentation_quality"].value_counts().to_dict(),
                    "report_quality": df["report_quality"].value_counts().to_dict(),
                    "clinical_accuracy": df["clinical_accuracy"].value_counts().to_dict()
                },
                "expert_distribution": df["expert_id"].value_counts().to_dict(),
                "corrections_count": len([f for f in self.feedback_data if f["feedback"].get("corrections")])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate feedback summary: {e}")
            return {"error": str(e)}
    
    def get_feedback_for_training(self, min_quality_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Get feedback data formatted for model training.
        
        Args:
            min_quality_threshold: Minimum quality threshold for including feedback
            
        Returns:
            List of training-ready feedback entries
        """
        try:
            training_data = []
            
            for entry in self.feedback_data:
                feedback = entry["feedback"]
                
                if feedback.get("overall_quality", 0) >= min_quality_threshold:
                    training_entry = {
                        "result_id": entry["result_id"],
                        "feedback_id": entry["feedback_id"],
                        "quality_scores": {
                            "overall": feedback.get("overall_quality", 0),
                            "segmentation": feedback.get("segmentation_quality", 0),
                            "report": feedback.get("report_quality", 0),
                            "clinical": feedback.get("clinical_accuracy", 0)
                        },
                        "corrections": feedback.get("corrections", {}),
                        "expert_metadata": {
                            "expert_id": feedback.get("expert_id", "anonymous"),
                            "expertise_level": feedback.get("expertise_level", "unknown")
                        }
                    }
                    training_data.append(training_entry)
            
            logger.info(f"Prepared {len(training_data)} feedback entries for training")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return []
    
    def generate_preference_pairs(self) -> List[Dict[str, Any]]:
        """
        Generate preference pairs for DPO training.
        
        Returns:
            List of preference pairs
        """
        try:
            preference_pairs = []
            
            # Group feedback by result pairs
            feedback_by_result = {}
            for entry in self.feedback_data:
                result_id = entry["result_id"]
                if result_id not in feedback_by_result:
                    feedback_by_result[result_id] = []
                feedback_by_result[result_id].append(entry)
            
            # Create preference pairs from results with multiple feedback
            for result_id, feedbacks in feedback_by_result.items():
                if len(feedbacks) >= 2:
                    # Sort by overall quality
                    sorted_feedbacks = sorted(feedbacks, 
                                            key=lambda x: x["feedback"].get("overall_quality", 0), 
                                            reverse=True)
                    
                    # Create pairs (better vs worse)
                    for i in range(len(sorted_feedbacks) - 1):
                        better = sorted_feedbacks[i]
                        worse = sorted_feedbacks[i + 1]
                        
                        if better["feedback"].get("overall_quality", 0) > worse["feedback"].get("overall_quality", 0):
                            pair = {
                                "result_id": result_id,
                                "preferred": {
                                    "feedback_id": better["feedback_id"],
                                    "quality_score": better["feedback"].get("overall_quality", 0),
                                    "corrections": better["feedback"].get("corrections", {})
                                },
                                "rejected": {
                                    "feedback_id": worse["feedback_id"],
                                    "quality_score": worse["feedback"].get("overall_quality", 0),
                                    "corrections": worse["feedback"].get("corrections", {})
                                }
                            }
                            preference_pairs.append(pair)
            
            logger.info(f"Generated {len(preference_pairs)} preference pairs")
            return preference_pairs
            
        except Exception as e:
            logger.error(f"Failed to generate preference pairs: {e}")
            return []
    
    def export_feedback_dataset(self, output_path: str, format: str = "json"):
        """
        Export feedback dataset for external use.
        
        Args:
            output_path: Output file path
            format: Export format ("json", "csv", "parquet")
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(self.feedback_data, f, indent=2, default=str)
                    
            elif format == "csv":
                flattened_data = []
                for entry in self.feedback_data:
                    flat_entry = {
                        "result_id": entry["result_id"],
                        "timestamp": entry["timestamp"],
                        "feedback_id": entry["feedback_id"]
                    }
                    
                    for key, value in entry["feedback"].items():
                        if not isinstance(value, (dict, list)):
                            flat_entry[f"feedback_{key}"] = value
                    
                    flattened_data.append(flat_entry)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(output_path, index=False)
                
            elif format == "parquet":
                flattened_data = []
                for entry in self.feedback_data:
                    flat_entry = {
                        "result_id": entry["result_id"],
                        "timestamp": entry["timestamp"],
                        "feedback_id": entry["feedback_id"]
                    }
                    
                    for key, value in entry["feedback"].items():
                        if not isinstance(value, (dict, list)):
                            flat_entry[f"feedback_{key}"] = value
                    
                    flattened_data.append(flat_entry)
                
                df = pd.DataFrame(flattened_data)
                df.to_parquet(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported feedback dataset to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export feedback dataset: {e}")
    
    def get_feedback_by_result(self, result_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a specific result.
        
        Args:
            result_id: Result identifier
            
        Returns:
            List of feedback entries
        """
        return [entry for entry in self.feedback_data if entry["result_id"] == result_id]
    
    def get_feedback_by_expert(self, expert_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback from a specific expert.
        
        Args:
            expert_id: Expert identifier
            
        Returns:
            List of feedback entries
        """
        return [entry for entry in self.feedback_data 
                if entry["feedback"].get("expert_id") == expert_id]
    
    def delete_feedback(self, feedback_id: int):
        """
        Delete feedback entry.
        
        Args:
            feedback_id: Feedback identifier
        """
        try:
            self.feedback_data = [entry for entry in self.feedback_data 
                                if entry["feedback_id"] != feedback_id]
            self._save_feedback()
            logger.info(f"Deleted feedback {feedback_id}")
        except Exception as e:
            logger.error(f"Failed to delete feedback: {e}")
    
    def clear_all_feedback(self):
        """Clear all feedback data."""
        try:
            self.feedback_data = []
            self._save_feedback()
            logger.info("Cleared all feedback data")
        except Exception as e:
            logger.error(f"Failed to clear feedback: {e}")