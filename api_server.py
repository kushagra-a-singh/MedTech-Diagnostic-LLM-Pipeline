"""
FastAPI backend server for MedTech Diagnostic LLM Pipeline.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

sys.path.append(str(Path(__file__).parent / "src"))

import warnings

from main import MedTechPipeline

# Suppress transformers deprecation warning
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.hub"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_server.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

pipeline: Optional[MedTechPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global pipeline
    try:
        logger.info("Initializing MedTech Pipeline...")
        pipeline = MedTechPipeline("configs")
        pipeline.initialize_components()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None

    yield

    logger.info("Shutting down API server...")


app = FastAPI(
    title="MedTech Diagnostic LLM Pipeline API",
    description="API for medical image segmentation, vector retrieval, and LLM-based report generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ProcessRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None


class ProcessResponse(BaseModel):
    status: str
    image_path: str
    components: Dict[str, Any]
    error: Optional[str] = None


class StatusResponse(BaseModel):
    pipeline_status: str
    components: Dict[str, str]
    vector_store_stats: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    result_id: str
    feedback: Dict[str, Any]


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None


processing_tasks: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MedTech Diagnostic LLM Pipeline API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {"status": "healthy", "timestamp": str(np.datetime64("now"))}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get pipeline status."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        status = pipeline.get_pipeline_status()
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=ProcessResponse)
async def process_image(request: ProcessRequest):
    """Process medical image through the pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Validate image path
        if not os.path.exists(request.image_path):
            raise HTTPException(
                status_code=404, detail=f"Image not found: {request.image_path}"
            )

        logger.info(f"Processing image: {request.image_path}")
        result = pipeline.process_medical_image(request.image_path, request.output_dir)

        return ProcessResponse(
            status=result.get("status", "unknown"),
            image_path=result.get("image_path", request.image_path),
            components=result.get("components", {}),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        return ProcessResponse(
            status="failed", image_path=request.image_path, components={}, error=str(e)
        )


@app.post("/process-upload", response_model=ProcessResponse)
async def process_upload(file: UploadFile = File(...)):
    """Process uploaded medical image."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Processing uploaded file: {file.filename}")
        result = pipeline.process_medical_image(str(file_path), str(uploads_dir))

        return ProcessResponse(
            status=result.get("status", "unknown"),
            image_path=str(file_path),
            components=result.get("components", {}),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Failed to process upload: {e}")
        return ProcessResponse(
            status="failed",
            image_path=file.filename if file else "unknown",
            components={},
            error=str(e),
        )


@app.post("/process-async")
async def process_image_async(
    request: ProcessRequest, background_tasks: BackgroundTasks
):
    """Process medical image asynchronously."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import uuid

    task_id = str(uuid.uuid4())

    processing_tasks[task_id] = {
        "status": "queued",
        "image_path": request.image_path,
        "created_at": str(np.datetime64("now")),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(process_image_background, task_id, request)

    return {"task_id": task_id, "status": "queued"}


async def process_image_background(task_id: str, request: ProcessRequest):
    """Background task for image processing."""
    try:
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["started_at"] = str(np.datetime64("now"))

        result = pipeline.process_medical_image(request.image_path, request.output_dir)

        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["completed_at"] = str(np.datetime64("now"))
        processing_tasks[task_id]["result"] = result

    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["failed_at"] = str(np.datetime64("now"))


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of async processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return processing_tasks[task_id]


@app.get("/tasks")
async def list_tasks():
    """List all processing tasks."""
    return {"tasks": processing_tasks}


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit expert feedback."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.collect_feedback(request.result_id, request.feedback)
        return {"status": "success", "message": "Feedback submitted"}
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_similar_cases(request: QueryRequest):
    """Search for similar cases."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if pipeline.vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        if hasattr(pipeline.vector_store, "embedding_extractor"):
            query_embeddings = (
                pipeline.vector_store.embedding_extractor.extract_text_embeddings(
                    request.query
                )
            )
        else:
            query_embeddings = np.random.randn(1, 768)

        results = pipeline.vector_store.search_with_metadata(
            query_embeddings, request.k
        )

        return {"query": request.query, "results": results}

    except Exception as e:
        logger.error(f"Failed to search similar cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_processing_history():
    """Get processing history."""
    try:
        recent_tasks = {k: v for k, v in list(processing_tasks.items())[-10:]}
        return {"history": recent_tasks}
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{result_id}")
async def get_result_details(result_id: str):
    """Get detailed results for a specific processing result."""
    if result_id in processing_tasks:
        return processing_tasks[result_id]
    else:
        raise HTTPException(status_code=404, detail="Result not found")


@app.post("/models/download")
async def download_models(background_tasks: BackgroundTasks):
    """Download required models."""
    background_tasks.add_task(download_models_background)
    return {"status": "started", "message": "Model download started in background"}


async def download_models_background():
    """Background task for downloading models."""
    try:
        logger.info("Starting model downloads...")

        from setup_models import download_all_models

        download_all_models()

        logger.info("Model downloads completed")

    except Exception as e:
        logger.error(f"Failed to download models: {e}")


@app.get("/models/status")
async def get_models_status():
    """Get status of required models."""
    try:
        models_dir = Path("models")

        model_status = {
            "swin_unetr": {
                "path": "models/swin_unetr_pretrained.pth",
                "exists": (models_dir / "swin_unetr_pretrained.pth").exists(),
                "size": None,
            },
            "biomistral": {
                "path": "models/biomistral",
                "exists": (models_dir / "biomistral").exists(),
                "size": None,
            },
            "hippo": {
                "path": "models/hippo",
                "exists": (models_dir / "hippo").exists(),
                "size": None,
            },
            "falcon": {
                "path": "models/falcon",
                "exists": (models_dir / "falcon").exists(),
                "size": None,
            },
        }

        for model_name, info in model_status.items():
            if info["exists"]:
                path = Path(info["path"])
                if path.is_file():
                    info["size"] = path.stat().st_size
                elif path.is_dir():
                    info["size"] = sum(
                        f.stat().st_size for f in path.rglob("*") if f.is_file()
                    )

        return {"models": model_status}

    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_configuration():
    """Get current pipeline configuration."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        return {"config": pipeline.configs}
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/reload")
async def reload_configuration():
    """Reload pipeline configuration."""
    global pipeline

    try:
        logger.info("Reloading pipeline configuration...")
        pipeline = MedTechPipeline("configs")
        pipeline.initialize_components()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get pipeline performance metrics."""
    try:
        metrics = {
            "total_processed": len(processing_tasks),
            "successful": len(
                [t for t in processing_tasks.values() if t["status"] == "completed"]
            ),
            "failed": len(
                [t for t in processing_tasks.values() if t["status"] == "failed"]
            ),
            "processing": len(
                [t for t in processing_tasks.values() if t["status"] == "processing"]
            ),
            "queued": len(
                [t for t in processing_tasks.values() if t["status"] == "queued"]
            ),
        }

        if pipeline and pipeline.vector_store:
            vector_stats = pipeline.vector_store.get_index_stats()
            metrics["vector_store"] = vector_stats

        return {"metrics": metrics}

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    del processing_tasks[task_id]
    return {"status": "success", "message": "Task deleted"}


@app.delete("/tasks")
async def clear_all_tasks():
    """Clear all processing tasks."""
    processing_tasks.clear()
    return {"status": "success", "message": "All tasks cleared"}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    scan_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for conversational interaction with the medical LLM.

    Supports:
    - Text-based questions
    - Context from uploaded scans
    - Follow-up questions with conversation history
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")

    try:
        import uuid
        from pathlib import Path

        sys.path.append(str(Path(__file__).parent / "src"))
        from utils.context_builder import MedicalContextBuilder
        from utils.patient_storage import PatientStorage

        session_id = request.session_id or f"session_{uuid.uuid4()}"

        # Build context
        context_builder = MedicalContextBuilder()
        patient_storage = PatientStorage()

        # Get scan context if available
        segmentation_results = None
        similar_cases = None
        clinical_context = {}

        if request.scan_context:
            segmentation_results = request.scan_context.get("segmentation")
            similar_cases = request.scan_context.get("similar_cases", [])

            # Try to extract patient info from image path
            image_path = request.scan_context.get("image_path", "")
            if image_path:
                # Simple patient ID extraction (can be improved)
                clinical_context = {
                    "modality": (
                        "CT"
                        if "ct" in image_path.lower()
                        else "MRI" if "mri" in image_path.lower() else "Unknown"
                    ),
                    "body_region": "Unknown",
                    "study_date": str(np.datetime64("now")),
                }

        # Build comprehensive context
        context = context_builder.build_context(
            segmentation_results=segmentation_results,
            similar_cases=similar_cases,
            clinical_context=clinical_context,
            conversation_history=request.conversation_history,
        )

        # Build prompt
        prompt = context_builder.build_chat_prompt(
            user_message=request.message,
            context=context,
        )

        # Generate response using LLM
        llm_context = {
            "segmentation_findings": context["segmentation_findings"],
            "similar_cases": context["similar_cases"],
            "clinical_context": context["clinical_context"],
        }

        # Use answer_question method for chat
        response_text = pipeline.llm.answer_question(request.message, llm_context)

        # If response is empty or error, try generate_report
        if not response_text or "Error" in response_text:
            report_result = pipeline.llm.generate_report(llm_context)
            if isinstance(report_result, dict) and "report" in report_result:
                response_text = report_result["report"]
            else:
                response_text = str(report_result)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=str(np.datetime64("now")),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MedTech Diagnostic LLM Pipeline API Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )
