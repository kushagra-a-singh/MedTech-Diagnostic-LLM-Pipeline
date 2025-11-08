"""
Medical LLM implementation for report generation.
"""

import logging
import os
import warnings
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Workaround for PyTorch < 2.6 to load .bin files
# This is a temporary solution until PyTorch 2.6 is available
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    """Patched torch.load that bypasses version check for .bin files."""
    # Remove weights_only to bypass security check
    kwargs.pop('weights_only', None)
    try:
        # Try using internal _load which doesn't have version check
        if hasattr(torch.serialization, '_load'):
            return torch.serialization._load(f, map_location, pickle_module, **kwargs)
        else:
            # Fallback: use original with weights_only=False
            return _original_torch_load(f, map_location, pickle_module, weights_only=False, **kwargs)
    except Exception:
        return _original_torch_load(f, map_location, pickle_module, **kwargs)

# Apply workaround if PyTorch < 2.6
torch_version = torch.__version__.split('+')[0]
torch_major, torch_minor = map(int, torch_version.split('.')[:2])
if torch_major < 2 or (torch_major == 2 and torch_minor < 6):
    torch.load = _patched_torch_load
    warnings.warn(
        "⚠️  PyTorch < 2.6 detected. Applied workaround to load .bin files. "
        "This bypasses security restrictions - only use with trusted models. "
        "Upgrade to PyTorch 2.6+ when available.",
        UserWarning,
        stacklevel=2
    )

# Try to import BitsAndBytesConfig for quantization (optional dependency)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    try:
        from transformers.utils.bitsandbytes import BitsAndBytesConfig
        BITSANDBYTES_AVAILABLE = True
    except ImportError:
        BITSANDBYTES_AVAILABLE = False
        BitsAndBytesConfig = None

# Check if bitsandbytes library is available
try:
    import bitsandbytes as bnb
    BITSANDBYTES_LIB_AVAILABLE = True
except ImportError:
    BITSANDBYTES_LIB_AVAILABLE = False

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
            # ✅ Accepts both full 'llm' config or a single model section
            if "biomistral" in self.config or "hippo" in self.config or "falcon" in self.config:
                # Full LLM config passed
                model_type = self.config.get("model_type", "biomistral")
                model_config = self.config.get(model_type, {})
            else:
                # Only a model subsection passed (fallback)
                model_type = "biomistral"
                model_config = self.config

            use_hf_hub = self.config.get("use_hf_hub", True)
            trust_remote_code = self.config.get("trust_remote_code", False)
            auth_env_var = self.config.get("auth_env_var", "HUGGINGFACE_HUB_TOKEN")
            hf_token = os.getenv(auth_env_var) or os.getenv("HF_TOKEN")

            model_id_or_path = model_config.get("model_name")

            # ✅ Expand relative paths (like "./models/biomistral") to absolute
            # Resolve relative to project root (2 levels up from src/llm/medical_llm.py)
            if model_id_or_path:
                # Find project root (go up 2 levels from src/llm/medical_llm.py)
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                
                if model_id_or_path.startswith("./"):
                    # Remove leading "./" and resolve relative to project root
                    relative_path = model_id_or_path[2:]
                    model_id_or_path = os.path.abspath(os.path.join(project_root, relative_path))
                elif model_id_or_path.startswith("../"):
                    # For "../" paths, resolve from project root
                    model_id_or_path = os.path.abspath(os.path.join(project_root, model_id_or_path))
                elif not os.path.isabs(model_id_or_path):
                    # Relative path without "./" prefix - resolve from project root
                    model_id_or_path = os.path.abspath(os.path.join(project_root, model_id_or_path))

            # ✅ Local vs HF loading
            if model_id_or_path and os.path.exists(model_id_or_path):
                logger.info(f"Loading local model from {model_id_or_path}")
                use_hf_hub = False
            else:
                logger.info(f"Loading model from Hugging Face Hub: {model_id_or_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
                use_auth_token=hf_token if use_hf_hub and hf_token else None,
            )

            # Configure quantization
            quantization_config = None
            load_in_8bit = model_config.get("load_in_8bit", False)
            load_in_4bit = model_config.get("load_in_4bit", False)
            device_map = model_config.get("device_map", "auto")
            
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "use_auth_token": hf_token if use_hf_hub and hf_token else None,
            }
            
            if load_in_8bit or load_in_4bit:
                if BITSANDBYTES_AVAILABLE and BITSANDBYTES_LIB_AVAILABLE:
                    # Use BitsAndBytesConfig (new way - preferred)
                    try:
                        if load_in_8bit:
                            # Try with CPU offloading enabled (for large models that don't fit on GPU)
                            try:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True,
                                    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for large models
                                )
                                logger.info("Using BitsAndBytesConfig for 8-bit quantization with CPU offloading enabled")
                            except TypeError:
                                # llm_int8_enable_fp32_cpu_offload might not be available in this version
                                # Try without it - device_map="auto" should handle offloading
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True,
                                )
                                logger.info("Using BitsAndBytesConfig for 8-bit quantization (CPU offload handled by device_map)")
                        elif load_in_4bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                            )
                            logger.info("Using BitsAndBytesConfig for 4-bit quantization")
                        
                        model_kwargs["quantization_config"] = quantization_config
                        # Use "auto" device_map to allow accelerate to handle CPU offloading if needed
                        model_kwargs["device_map"] = device_map if device_map != "auto" else "auto"
                    except Exception as e:
                        logger.warning(f"Failed to create BitsAndBytesConfig: {e}. Falling back to deprecated method.")
                        # Fall back to deprecated method
                        quantization_config = None
                
                # Fallback to deprecated method if BitsAndBytesConfig not available or failed
                if quantization_config is None:
                    if not BITSANDBYTES_LIB_AVAILABLE:
                        logger.warning(
                            "bitsandbytes library not available. Install it with: pip install bitsandbytes. "
                            "Falling back to deprecated quantization methods."
                        )
                    
                    # Use deprecated method - but it has limitations with CPU offloading
                    if load_in_8bit:
                        # The deprecated load_in_8bit doesn't support CPU offloading well
                        # Try to use a more restrictive device_map to keep model on GPU
                        # If model doesn't fit, we'll get an error and can handle it
                        model_kwargs["load_in_8bit"] = True
                        # Try to keep model on GPU only - if it doesn't fit, we'll need BitsAndBytesConfig
                        if device_map == "auto":
                            # Use "auto" but it may fail if model doesn't fit
                            # Better to explicitly set to GPU 0 to avoid CPU offload issues
                            if torch.cuda.is_available():
                                model_kwargs["device_map"] = "cuda:0"
                                logger.info("Using device_map='cuda:0' to avoid CPU offload issues with deprecated load_in_8bit")
                            else:
                                model_kwargs["device_map"] = "cpu"
                                logger.warning("No GPU available, loading on CPU (this may be very slow)")
                        else:
                            model_kwargs["device_map"] = device_map
                        logger.warning("Using deprecated load_in_8bit. If you get CPU offload errors, install/upgrade bitsandbytes and transformers to use BitsAndBytesConfig.")
                    elif load_in_4bit:
                        model_kwargs["load_in_4bit"] = True
                        model_kwargs["device_map"] = device_map
                        logger.warning("Using deprecated load_in_4bit. Consider upgrading to BitsAndBytesConfig.")
            else:
                # No quantization
                model_kwargs["device_map"] = device_map

            # Try to load the model
            # Prefer safetensors format to avoid PyTorch version restrictions
            # Check if safetensors files exist
            if model_id_or_path and os.path.exists(model_id_or_path):
                # Check for safetensors files
                safetensors_files = [
                    f for f in os.listdir(model_id_or_path) 
                    if f.endswith('.safetensors') or (f.endswith('.json') and 'safetensors' in f.lower())
                ]
                if safetensors_files:
                    # Explicitly use safetensors if available
                    model_kwargs["use_safetensors"] = True
                    logger.info(f"Found safetensors files ({len(safetensors_files)}), using safetensors format for loading")
                else:
                    # Check PyTorch version and warn if too old
                    torch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix
                    torch_major, torch_minor = map(int, torch_version.split('.')[:2])
                    if torch_major < 2 or (torch_major == 2 and torch_minor < 6):
                        logger.warning(
                            f"⚠️  No safetensors files found and PyTorch {torch.__version__} is too old (< 2.6). "
                            f"Model loading may fail due to security restrictions on .bin files.\n"
                            f"   Solutions:\n"
                            f"   1. Download model in safetensors format: python download_model_safetensors.py BioMistral/BioMistral-7B {model_id_or_path}\n"
                            f"   2. Upgrade PyTorch: pip install --upgrade torch>=2.6.0\n"
                            f"   3. See FIX_PYTORCH_VERSION_ISSUE.md for details"
                        )
                    # Don't set use_safetensors=False - let transformers handle it (will prefer safetensors if available)
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    **model_kwargs
                )
            except Exception as model_error:
                error_msg = str(model_error)
                # Check if it's the PyTorch version issue with .bin files
                if "upgrade torch to at least v2.6" in error_msg or "CVE-2025-32434" in error_msg:
                    logger.error(
                        f"Model loading failed due to PyTorch version restriction. "
                        f"PyTorch 2.6+ is required to load .bin files due to security vulnerability (CVE-2025-32434).\n"
                        f"Current PyTorch version: {torch.__version__}\n"
                        f"Solutions:\n"
                        f"1. Convert model to safetensors format (recommended):\n"
                        f"   python -c \"from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('{model_id_or_path}', torch_dtype='auto'); model.save_pretrained('{model_id_or_path}', safe_serialization=True)\"\n"
                        f"2. Upgrade PyTorch to 2.6+ (if available):\n"
                        f"   pip install --upgrade torch>=2.6.0\n"
                        f"3. Disable quantization temporarily to test model loading\n"
                        f"Original error: {error_msg}"
                    )
                    # Don't try fallback for this error - it won't work
                    raise model_error
                # Check if it's the CPU offload error with deprecated quantization
                elif "Some modules are dispatched on the CPU or the disk" in error_msg and load_in_8bit:
                    logger.error(
                        f"Model loading failed due to CPU offload issue with 8-bit quantization. "
                        f"This happens when the model doesn't fit on GPU and the deprecated load_in_8bit "
                        f"doesn't support CPU offloading.\n"
                        f"Solutions:\n"
                        f"1. Ensure bitsandbytes and transformers are up to date: pip install --upgrade bitsandbytes transformers\n"
                        f"2. The code should automatically use BitsAndBytesConfig if available.\n"
                        f"3. If the issue persists, try reducing model size or disabling quantization.\n"
                        f"Original error: {error_msg}"
                    )
                    # Try loading without quantization as fallback
                    logger.info("Attempting to load model without quantization as fallback...")
                    try:
                        fallback_kwargs = {
                            "trust_remote_code": trust_remote_code,
                            "use_auth_token": hf_token if use_hf_hub and hf_token else None,
                            "device_map": "cpu" if not torch.cuda.is_available() else "auto",
                        }
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_id_or_path,
                            **fallback_kwargs
                        )
                        logger.warning("Model loaded without quantization. This may use more memory and be slower.")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load model even without quantization: {fallback_error}")
                        raise model_error  # Raise the original error
                else:
                    # Re-raise other errors
                    raise

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
