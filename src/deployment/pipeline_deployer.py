"""
Pipeline deployment implementation.
"""

import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))


logger = logging.getLogger(__name__)


class PipelineDeployer:
    """
    Deploy the MedTech pipeline.
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline deployer.
        Args:
            config: Configuration dictionary
        """
        self.config = config

        logger.info("Pipeline deployer initialized")

    def deploy(self, deployment_type: str = "api"):
        """
        Deploy the pipeline.

        Args:
            deployment_type: Type of deployment ("api", "web", "docker")
        """
        logger.info(f"Deploying pipeline as: {deployment_type}")

        if deployment_type == "api":
            self._deploy_api()
        elif deployment_type == "web":
            self._deploy_web()
        elif deployment_type == "docker":
            self._deploy_docker()
        else:
            raise ValueError(f"Unsupported deployment type: {deployment_type}")

    def _deploy_api(self):
        """Deploy as API server using FastAPI."""
        logger.info("Deploying API server...")
        config_dir = self.config.get("config_dir", "configs")
        
        from main import MedTechPipeline

        pipeline = MedTechPipeline(config_dir)
        pipeline.initialize_components()

        app = FastAPI(title="MedTech Diagnostic LLM Pipeline API")

        #CORS
        api_cfg = self.config.get("api_server", {})
        cors_cfg = api_cfg.get("cors", {"enabled": True, "origins": ["*"]})
        if cors_cfg.get("enabled", True):
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_cfg.get("origins", ["*"]),
                allow_credentials=True,
                allow_methods=cors_cfg.get("methods", ["*"]),
                allow_headers=cors_cfg.get("headers", ["*"]),
            )

        class ProcessRequest(BaseModel):
            image_path: str
            output_dir: Optional[str] = None

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/status")
        def status():
            return pipeline.get_pipeline_status()

        @app.post("/process")
        def process(req: ProcessRequest):
            return pipeline.process_medical_image(req.image_path, req.output_dir)

        @app.post("/process-upload")
        async def process_upload(file: UploadFile = File(...)):
            uploads_dir = PROJECT_ROOT / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            dest_path = uploads_dir / file.filename
            with open(dest_path, "wb") as f:
                f.write(await file.read())
            return pipeline.process_medical_image(str(dest_path), str(uploads_dir))

        @app.get("/history")
        def history_list():
            outputs = PROJECT_ROOT / "data" / "segmentation_outputs"
            outputs.mkdir(parents=True, exist_ok=True)
            items = []
            for p in outputs.glob("*_metadata.json"):
                try:
                    import json as _json

                    with open(p, "r") as f:
                        meta = _json.load(f)
                    items.append(
                        {"id": p.stem.replace("_metadata", ""), "metadata": meta}
                    )
                except Exception:
                    continue
            items.sort(
                key=lambda x: x["metadata"].get("processing_timestamp", ""),
                reverse=True,
            )
            return {"items": items}

        @app.get("/history/{item_id}")
        def history_item(item_id: str):
            outputs = PROJECT_ROOT / "data" / "segmentation_outputs"
            meta = outputs / f"{item_id}_metadata.json"
            mask = outputs / f"{item_id}_mask.nii.gz"
            emb = outputs / f"{item_id}_embeddings.npy"
            if not meta.exists():
                return {"error": "not_found"}
            import json as _json

            with open(meta, "r") as f:
                meta_json = _json.load(f)
            return {
                "id": item_id,
                "metadata": meta_json,
                "mask_path": str(mask) if mask.exists() else None,
                "embedding_path": str(emb) if emb.exists() else None,
            }

        api_cfg = self.config.get("api_server", {})
        host = api_cfg.get("host", "0.0.0.0")
        port = int(api_cfg.get("port", 8000))
        uvicorn.run(app, host=host, port=port)

    def _deploy_web(self):
        """Deploy as web application."""
        logger.info("Deploying web application...")
        

    def _deploy_docker(self):
        """Deploy as Docker container."""
        logger.info("Deploying Docker container...")
        
