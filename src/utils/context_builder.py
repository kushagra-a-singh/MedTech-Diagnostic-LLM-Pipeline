"""
Context builder for assembling medical imaging context for LLM prompts.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MedicalContextBuilder:
    """
    Builds comprehensive context for medical LLM prompts from various sources.
    """

    def __init__(self):
        """Initialize the context builder."""
        pass

    def build_context(
        self,
        segmentation_results: Optional[Dict] = None,
        similar_cases: Optional[List[Dict]] = None,
        clinical_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        patient_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for LLM prompt.

        Args:
            segmentation_results: Segmentation pipeline results
            similar_cases: Retrieved similar cases from FAISS
            clinical_context: Clinical context from FHIR/local storage
            conversation_history: Previous conversation messages
            patient_info: Patient information

        Returns:
            Comprehensive context dictionary
        """
        context = {
            "segmentation_findings": self._format_segmentation_findings(
                segmentation_results
            ),
            "similar_cases": self._format_similar_cases(similar_cases),
            "clinical_context": clinical_context or {},
            "conversation_history": self._format_conversation_history(
                conversation_history
            ),
            "patient_info": patient_info or {},
        }

        return context

    def _format_segmentation_findings(self, seg_results: Optional[Dict]) -> str:
        """Format segmentation findings for prompt."""
        if not seg_results:
            return "No segmentation results available."

        findings = []
        findings.append("=== SEGMENTATION FINDINGS ===\n")

        # Image information
        if "image_path" in seg_results:
            findings.append(f"Image: {seg_results['image_path']}\n")

        # Metrics
        if "metrics" in seg_results:
            metrics = seg_results["metrics"]

            if "num_classes" in metrics:
                findings.append(f"Detected classes: {metrics['num_classes']}\n")

            if "volumes" in metrics:
                findings.append("\nDetected volumes:")
                for class_id, volume in metrics["volumes"].items():
                    findings.append(f"  - Class {class_id}: {volume:.2f} cc")

            if "dice_scores" in metrics:
                findings.append("\nSegmentation quality (Dice scores):")
                for class_id, dice in metrics["dice_scores"].items():
                    findings.append(f"  - Class {class_id}: {dice:.3f}")

            if "mean_dice" in metrics:
                findings.append(f"\nMean Dice Score: {metrics['mean_dice']:.3f}")

            if "class_distribution" in metrics:
                findings.append("\nClass distribution:")
                for class_id, count in metrics["class_distribution"].items():
                    findings.append(f"  - Class {class_id}: {count} voxels")

        # Model information
        if "model_info" in seg_results:
            model_info = seg_results["model_info"]
            findings.append(f"\nModel: {model_info.get('model_name', 'Unknown')}")

        return (
            "\n".join(findings) if findings else "No segmentation findings available."
        )

    def _format_similar_cases(self, similar_cases: Optional[List[Dict]]) -> str:
        """Format similar cases for prompt."""
        if not similar_cases or len(similar_cases) == 0:
            return "No similar cases found in the database."

        formatted = []
        formatted.append("=== SIMILAR CASES (Retrieved from FAISS) ===\n")

        # Handle both list of lists and flat list
        cases_to_format = similar_cases
        if similar_cases and isinstance(similar_cases[0], list):
            cases_to_format = similar_cases[0] if len(similar_cases) > 0 else []

        for i, case in enumerate(cases_to_format[:5], 1):  # Top 5 cases
            metadata = case.get("metadata", {})
            similarity = case.get("similarity", case.get("distance", 0.0))

            formatted.append(f"Case {i} (similarity: {similarity:.3f}):")

            if "image_path" in metadata:
                formatted.append(f"  Image: {metadata['image_path']}")

            if "segmentation_metrics" in metadata:
                metrics = metadata["segmentation_metrics"]
                if "volumes" in metrics:
                    formatted.append(
                        f"  Volumes: {len(metrics.get('volumes', {}))} classes detected"
                    )

            formatted.append("")  # Empty line between cases

        return "\n".join(formatted)

    def _format_conversation_history(self, history: Optional[List[Dict]]) -> str:
        """Format conversation history for prompt."""
        if not history or len(history) == 0:
            return "No previous conversation history."

        formatted = []
        formatted.append("=== CONVERSATION HISTORY ===\n")

        for msg in history[-10:]:  # Last 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")

            formatted.append("")

        return "\n".join(formatted)

    def build_chat_prompt(
        self,
        user_message: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build a chat prompt from user message and context.

        Args:
            user_message: Current user message
            context: Comprehensive context dictionary
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # System prompt
        if system_prompt:
            prompt_parts.append(system_prompt)
        else:
            prompt_parts.append(
                "You are a medical imaging diagnostic assistant. "
                "You help radiologists and clinicians analyze medical images (MRI/CT scans) "
                "by providing insights based on segmentation results, similar cases, and clinical context. "
                "Always be professional, accurate, and emphasize that your analysis should be reviewed by qualified medical professionals."
            )

        prompt_parts.append("\n")

        # Patient and clinical context
        if context.get("patient_info"):
            patient = context["patient_info"]
            prompt_parts.append("=== PATIENT INFORMATION ===")
            if "patient_id" in patient:
                prompt_parts.append(f"Patient ID: {patient['patient_id']}")
            if "age" in patient:
                prompt_parts.append(f"Age: {patient['age']}")
            if "gender" in patient:
                prompt_parts.append(f"Gender: {patient['gender']}")
            prompt_parts.append("")

        # Clinical context
        clinical_ctx = context.get("clinical_context", {})
        if clinical_ctx:
            prompt_parts.append("=== CLINICAL CONTEXT ===")
            if "modality" in clinical_ctx:
                prompt_parts.append(f"Modality: {clinical_ctx['modality']}")
            if "body_region" in clinical_ctx:
                prompt_parts.append(f"Body Region: {clinical_ctx['body_region']}")
            if "study_date" in clinical_ctx:
                prompt_parts.append(f"Study Date: {clinical_ctx['study_date']}")
            prompt_parts.append("")

        # Segmentation findings
        prompt_parts.append(
            context.get("segmentation_findings", "No segmentation findings.")
        )
        prompt_parts.append("")

        # Similar cases
        prompt_parts.append(context.get("similar_cases", "No similar cases."))
        prompt_parts.append("")

        # Conversation history
        if (
            context.get("conversation_history")
            and len(context["conversation_history"]) > 0
        ):
            prompt_parts.append(context["conversation_history"])
            prompt_parts.append("")

        # Current user message
        prompt_parts.append("=== CURRENT USER QUESTION ===")
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("")
        prompt_parts.append("=== ASSISTANT RESPONSE ===")
        prompt_parts.append(
            "Please provide a helpful, professional response to the user's question based on the above context:"
        )

        return "\n".join(prompt_parts)
