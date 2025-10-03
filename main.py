"""
Main MedTech Diagnostic LLM Pipeline orchestrator.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent / "src"))

from deployment import PipelineDeployer
from feedback import FeedbackCollector
from llm import MedicalLLM
from mcp_fhir import MCPFHIRClient
from segmentation import SegmentationPipeline
from vector_store import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class MedTechPipeline:
    """
    Main pipeline orchestrator for the MedTech Diagnostic LLM Pipeline.
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the MedTech pipeline.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = self._load_configs()

        self.segmentation_pipeline = None
        self.vector_store = None
        self.llm = None
        self.mcp_fhir = None
        self.feedback_collector = None
        self.deployer = None

        logger.info("MedTech Pipeline initialized")

    def _load_configs(self) -> Dict:
        """
        Load all configuration files.

        Returns:
            Dictionary of configurations
        """
        configs = {}

        config_files = {
            "segmentation": "segmentation_config.yaml",
            "vector_store": "vector_store_config.yaml",
            "llm": "llm_config.yaml",
            "mcp_fhir": "mcp_fhir_config.yaml",
            "deployment": "deployment_config.yaml",
        }

        for component, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                with open(config_path, "r") as f:
                    configs[component] = yaml.safe_load(f)
                logger.info(f"Loaded config: {filename}")
            else:
                logger.warning(f"Config file not found: {filename}")
                configs[component] = {}

        return configs

    def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        if self.configs.get("segmentation"):
            self.segmentation_pipeline = SegmentationPipeline(
                str(self.config_dir / "segmentation_config.yaml")
            )
            logger.info("Segmentation pipeline initialized")

        if self.configs.get("vector_store"):
            self.vector_store = FAISSIndex(self.configs["vector_store"])
            logger.info("Vector store initialized")

        if self.configs.get("llm"):
            self.llm = MedicalLLM(self.configs["llm"])
            logger.info("Medical LLM initialized")

        if self.configs.get("mcp_fhir"):
            self.mcp_fhir = MCPFHIRClient(self.configs["mcp_fhir"])
            logger.info("MCP-FHIR client initialized")

        self.feedback_collector = FeedbackCollector()
        logger.info("Feedback collector initialized")

        if self.configs.get("deployment"):
            self.deployer = PipelineDeployer(self.configs["deployment"])
            logger.info("Pipeline deployer initialized")

    def process_medical_image(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process a single medical image through the complete pipeline.

        Args:
            image_path: Path to medical image
            output_dir: Output directory (optional)

        Returns:
            Complete pipeline results
        """
        logger.info(f"Processing medical image: {image_path}")

        results = {"image_path": image_path, "status": "processing", "components": {}}

        try:
            #Step 1:segmentation
            if self.segmentation_pipeline:
                logger.info("Step 1: Performing segmentation...")
                seg_results = self.segmentation_pipeline.process_single_image(
                    image_path, output_dir
                )
                results["components"]["segmentation"] = seg_results

                if "embedding_path" in seg_results:
                    embeddings = self._load_embeddings(seg_results["embedding_path"])

                    #Step 2:vector store indexing
                    if self.vector_store and embeddings is not None:
                        logger.info("Step 2: Indexing embeddings...")
                        self._index_embeddings(embeddings, seg_results)

            #Step 3:retrieve similar cases
            similar_cases = []
            if self.vector_store and "embedding_path" in seg_results:
                logger.info("Step 3: Retrieving similar cases...")
                similar_cases = self._retrieve_similar_cases(embeddings)
                results["components"]["similar_cases"] = similar_cases

            #Step 4:get clinical context from FHIR
            clinical_context = {}
            if self.mcp_fhir:
                logger.info("Step 4: Retrieving clinical context...")
                clinical_context = self._get_clinical_context(image_path)
                results["components"]["clinical_context"] = clinical_context

            #Step 5:generate medical report
            if self.llm:
                logger.info("Step 5: Generating medical report...")
                report = self._generate_medical_report(
                    seg_results, similar_cases, clinical_context
                )
                results["components"]["medical_report"] = report

            results["status"] = "completed"
            logger.info("Pipeline processing completed successfully")

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _load_embeddings(self, embedding_path: str) -> Optional[np.ndarray]:
        """Load embeddings from file."""
        try:
            embeddings = np.load(embedding_path)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embedding_path}: {e}")
            return None

    def _index_embeddings(self, embeddings: np.ndarray, seg_results: Dict):
        """Index embeddings in vector store."""
        try:
            metadata = [
                {
                    "image_path": seg_results["image_path"],
                    "mask_path": seg_results["mask_path"],
                    "embedding_path": seg_results["embedding_path"],
                    "segmentation_metrics": seg_results.get("metrics", {}),
                    "processing_timestamp": seg_results.get("processing_timestamp", ""),
                }
            ]

            self.vector_store.add_embeddings(embeddings, metadata)

        except Exception as e:
            logger.error(f"Failed to index embeddings: {e}")

    def _retrieve_similar_cases(
        self, query_embeddings: np.ndarray, k: int = 5
    ) -> List[Dict]:
        """Retrieve similar cases from vector store."""
        try:
            results = self.vector_store.search_with_metadata(query_embeddings, k)
            return results[0] if results else []  
        except Exception as e:
            logger.error(f"Failed to retrieve similar cases: {e}")
            return []

    def _get_clinical_context(self, image_path: str) -> Dict:
        """Get clinical context from FHIR."""
        try:
            context = {
                "patient_id": "unknown",
                "study_date": "unknown",
                "modality": "unknown",
                "body_region": "unknown",
            }

            if self.mcp_fhir:
                fhir_context = self.mcp_fhir.get_patient_context(context["patient_id"])
                context.update(fhir_context)

            return context

        except Exception as e:
            logger.error(f"Failed to get clinical context: {e}")
            return {}

    def _generate_medical_report(
        self, seg_results: Dict, similar_cases: List[Dict], clinical_context: Dict
    ) -> Dict:
        """Generate medical report using LLM."""
        try:
            prompt_context = {
                "segmentation_findings": self._format_segmentation_findings(
                    seg_results
                ),
                "similar_cases": self._format_similar_cases(similar_cases),
                "clinical_context": clinical_context,
            }

            report = self.llm.generate_report(prompt_context)

            return {"report": report, "generation_timestamp": str(np.datetime64("now"))}

        except Exception as e:
            logger.error(f"Failed to generate medical report: {e}")
            return {"error": str(e)}

    def _format_segmentation_findings(self, seg_results: Dict) -> str:
        """Format segmentation findings for LLM prompt."""
        findings = []

        if "metrics" in seg_results:
            metrics = seg_results["metrics"]

            if "volumes" in metrics:
                findings.append("Detected volumes:")
                for class_id, volume in metrics["volumes"].items():
                    findings.append(f"- Class {class_id}: {volume:.2f} cc")

            if "dice_scores" in metrics:
                findings.append("Segmentation quality (Dice scores):")
                for class_id, dice in metrics["dice_scores"].items():
                    findings.append(f"- Class {class_id}: {dice:.3f}")

        return "\n".join(findings) if findings else "No segmentation findings available"

    def _format_similar_cases(self, similar_cases: List[Dict]) -> str:
        """Format similar cases for LLM prompt."""
        if not similar_cases:
            return "No similar cases found"

        cases = []
        for i, case in enumerate(similar_cases[:3]):  #limit to top 3
            metadata = case.get("metadata", {})
            similarity = case.get("similarity", 0.0)

            case_info = f"Case {i+1} (similarity: {similarity:.3f}):"
            if "image_path" in metadata:
                case_info += f" {metadata['image_path']}"
            cases.append(case_info)

        return "\n".join(cases)

    def process_batch(
        self, image_paths: List[str], output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a batch of medical images.

        Args:
            image_paths: List of image paths
            output_dir: Output directory (optional)

        Returns:
            List of pipeline results
        """
        logger.info(f"Processing batch of {len(image_paths)} images")

        results = []
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.process_medical_image(image_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(
                    {"image_path": image_path, "status": "failed", "error": str(e)}
                )

        if output_dir:
            self._save_batch_summary(results, output_dir)

        return results

    def _save_batch_summary(self, results: List[Dict], output_dir: str):
        """Save batch processing summary."""
        summary = {
            "total_images": len(results),
            "successful": len([r for r in results if r.get("status") == "completed"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "results": results,
            "timestamp": str(np.datetime64("now")),
        }

        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Batch summary saved to: {summary_path}")

    def collect_feedback(self, result_id: str, feedback: Dict):
        """
        Collect expert feedback for pipeline results.

        Args:
            result_id: Result identifier
            feedback: Feedback data
        """
        if self.feedback_collector:
            self.feedback_collector.add_feedback(result_id, feedback)
            logger.info(f"Feedback collected for result: {result_id}")

    def deploy_pipeline(self, deployment_type: str = "api"):
        """
        Deploy the pipeline.

        Args:
            deployment_type: Type of deployment ("api", "web", "docker")
        """
        if self.deployer:
            self.deployer.deploy(deployment_type)
            logger.info(f"Pipeline deployed as: {deployment_type}")

    def get_pipeline_status(self) -> Dict:
        """
        Get pipeline status and component information.

        Returns:
            Pipeline status dictionary
        """
        status = {"pipeline_status": "initialized", "components": {}}

        if self.segmentation_pipeline:
            status["components"]["segmentation"] = "active"
        else:
            status["components"]["segmentation"] = "inactive"

        if self.vector_store:
            status["components"]["vector_store"] = "active"
            status["vector_store_stats"] = self.vector_store.get_index_stats()
        else:
            status["components"]["vector_store"] = "inactive"

        if self.llm:
            status["components"]["llm"] = "active"
        else:
            status["components"]["llm"] = "inactive"

        if self.mcp_fhir:
            status["components"]["mcp_fhir"] = "active"
        else:
            status["components"]["mcp_fhir"] = "inactive"

        return status


def main():
    """Main entry point for the MedTech pipeline."""
    parser = argparse.ArgumentParser(description="MedTech Diagnostic LLM Pipeline")
    parser.add_argument(
        "--config-dir", default="configs", help="Configuration directory"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "deploy"],
        default="single",
        help="Pipeline mode",
    )
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument(
        "--deployment-type",
        choices=["api", "web", "docker"],
        default="api",
        help="Deployment type",
    )

    args = parser.parse_args()

    pipeline = MedTechPipeline(args.config_dir)
    pipeline.initialize_components()

    if args.mode == "single":
        result = pipeline.process_medical_image(args.input, args.output_dir)
        print(json.dumps(result, indent=2))

    elif args.mode == "batch":
        if os.path.isdir(args.input):
            image_paths = []
            for ext in [".nii.gz", ".nii", ".dcm"]:
                image_paths.extend(Path(args.input).glob(f"*{ext}"))
            image_paths = [str(p) for p in image_paths]
        else:
            image_paths = [args.input]

        results = pipeline.process_batch(image_paths, args.output_dir)
        print(f"Processed {len(results)} images")

    elif args.mode == "deploy":
        pipeline.deploy_pipeline(args.deployment_type)
        print(f"Pipeline deployed as {args.deployment_type}")


if __name__ == "__main__":
    main()
