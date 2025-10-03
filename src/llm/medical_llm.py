"""
Medical LLM implementation for report generation.
"""

import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class MedicalLLM:
    """
    Medical LLM wrapper for report generation.
    """

    def __init__(self, config: Dict):
        """
        Initialize medical LLM.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_type = config.get("model_type", "biomistral")
        self.model = None
        self.tokenizer = None

        self._initialize_model()

        logger.info(f"Medical LLM initialized: {self.model_type}")

    def _initialize_model(self):
        """Initialize the LLM model."""
        try:
            if self.model_type == "biomistral":
                model_config = self.config["biomistral"]
            elif self.model_type == "hippo":
                model_config = self.config["hippo"]
            elif self.model_type == "falcon":
                model_config = self.config["falcon"]
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            use_hf_hub = self.config.get("use_hf_hub", True)
            trust_remote_code = self.config.get("trust_remote_code", False)
            auth_env_var = self.config.get("auth_env_var", "HUGGINGFACE_HUB_TOKEN")
            hf_token = os.getenv(auth_env_var) or os.getenv("HF_TOKEN")

            model_id_or_path = model_config.get("model_name")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                use_auth_token=hf_token if use_hf_hub and hf_token else None,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                load_in_8bit=model_config.get("load_in_8bit", False),
                load_in_4bit=model_config.get("load_in_4bit", False),
                device_map=model_config.get("device_map", "auto"),
                trust_remote_code=trust_remote_code,
                use_auth_token=hf_token if use_hf_hub and hf_token else None,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        logger.warning("Using dummy model for testing")
        self.model = None
        self.tokenizer = None

    def generate_report(self, context: Dict) -> Dict:
        """
        Generate medical report from context.

        Args:
            context: Context dictionary with segmentation findings, similar cases, etc.

        Returns:
            Generated report
        """
        try:
            prompt = self._prepare_prompt(context)

            if self.model is not None:
                report = self._generate_text(prompt)
            else:
                report = self._generate_dummy_report(context)

            return {
                "report": report,
                "model_type": self.model_type,
                "prompt_length": len(prompt),
            }

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "report": f"Error generating report: {str(e)}",
                "model_type": self.model_type,
                "error": str(e),
            }

    def _prepare_prompt(self, context: Dict) -> str:
        """Prepare prompt from context."""
        prompt_template = self.config["prompts"]["report_generation"]

        segmentation_findings = context.get("segmentation_findings", "No findings")
        similar_cases = context.get("similar_cases", "No similar cases")
        clinical_context = context.get("clinical_context", {})

        modality = clinical_context.get("modality", "Unknown")
        body_region = clinical_context.get("body_region", "Unknown")
        clinical_history = clinical_context.get(
            "clinical_history", "No clinical history provided"
        )

        prompt = prompt_template.format(
            modality=modality,
            body_region=body_region,
            clinical_history=clinical_history,
            segmentation_findings=segmentation_findings,
            similar_cases=similar_cases,
        )

        return prompt

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the model."""
       
        if self.model_type == "biomistral":
            gen_config = self.config["biomistral"]
        elif self.model_type == "hippo":
            gen_config = self.config["hippo"]
        elif self.model_type == "falcon":
            gen_config = self.config["falcon"]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=gen_config["max_length"],
        )
        if self.model is not None:
            try:
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception:
                pass

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=gen_config["max_length"],
                temperature=gen_config["temperature"],
                top_p=gen_config["top_p"],
                top_k=gen_config["top_k"],
                repetition_penalty=gen_config["repetition_penalty"],
                do_sample=gen_config["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        report = generated_text[len(prompt) :].strip()

        return report

    def _generate_dummy_report(self, context: Dict) -> str:
        """Generate a dummy report for testing."""
        segmentation_findings = context.get("segmentation_findings", "No findings")
        similar_cases = context.get("similar_cases", "No similar cases")

        report = f"""
DUMMY MEDICAL REPORT

Based on the provided segmentation findings and similar cases, here is a sample radiological report:

TECHNIQUE:
- Imaging modality: CT/MRI
- Body region: Abdomen
- Contrast: IV contrast administered

FINDINGS:
{segmentation_findings}

SIMILAR CASES REFERENCE:
{similar_cases}

IMPRESSION:
This is a dummy report generated for testing purposes. The segmentation analysis shows various anatomical structures have been identified. Further clinical correlation is recommended.

RECOMMENDATIONS:
1. Clinical correlation with patient history
2. Follow-up imaging as clinically indicated
3. Consultation with appropriate specialist if needed

Note: This is a test report and should not be used for clinical decision-making.
"""

        return report.strip()

    def answer_question(self, question: str, context: Dict) -> str:
        """
        Answer medical questions based on context.

        Args:
            question: Medical question
            context: Context information

        Returns:
            Answer to the question
        """
        try:
            qa_template = self.config["prompts"]["qa_template"]

            imaging_data = context.get("imaging_data", "No imaging data")
            clinical_context = context.get("clinical_context", "No clinical context")

            prompt = qa_template.format(
                question=question,
                imaging_data=imaging_data,
                clinical_context=clinical_context,
            )

            if self.model is not None:
                answer = self._generate_text(prompt)
            else:
                answer = f"Dummy answer to: {question}\n\nThis is a test response and should not be used for clinical purposes."

            return answer

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return f"Error answering question: {str(e)}"

    def summarize_report(self, report: str) -> str:
        """
        Summarize a medical report.

        Args:
            report: Medical report to summarize

        Returns:
            Summary of the report
        """
        try:
            summary_template = self.config["prompts"]["summarization_template"]
            prompt = summary_template.format(report=report)

            if self.model is not None:
                summary = self._generate_text(prompt)
            else:
                summary = f"Summary of report:\n\n{report[:200]}...\n\nThis is a test summary."

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize report: {e}")
            return f"Error summarizing report: {str(e)}"

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Model information dictionary
        """
        info = {
            "model_type": self.model_type,
            "model_name": self.config.get(self.model_type, {}).get(
                "model_name", "Unknown"
            ),
            "max_length": self.config.get(self.model_type, {}).get("max_length", 2048),
            "temperature": self.config.get(self.model_type, {}).get("temperature", 0.7),
        }

        if self.model is not None:
            info["model_loaded"] = True
            info["model_parameters"] = sum(p.numel() for p in self.model.parameters())
        else:
            info["model_loaded"] = False
            info["model_parameters"] = 0

        return info
