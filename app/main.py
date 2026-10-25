"""
ClauseGuard — FastAPI Application Server.
Exposes REST and Server-Sent Events (SSE) streaming endpoints for contract risk audits.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Generator

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import AuditReportSchema
from app.service import ContractAuditService

RULES_PATH = PROJECT_ROOT / "legal-ai" / "rules" / "review_rules.json"


# -----------------------------------------------------------------------------
# Application Lifespan: Loads ML models once at server startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Loads LegalBERT, NuExtract 2.0, PDF parser, and rule detector into GPU VRAM
    once when the server boots up, avoiding re-loading per request.
    """
    print("=" * 60)
    print("[*] ClauseGuard API Server starting up...")
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[*] GPU device: {torch.cuda.get_device_name(0)}")
        print(f"[*] Initial VRAM allocated: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")
    else:
        print("[!] No GPU detected. Running in CPU mode (slower).")

    # Instantiate the singleton audit service
    app.state.audit_service = ContractAuditService()
    print("[✓] ContractAuditService successfully initialized and ready for requests.")
    print("=" * 60)

    yield

    # Clean shutdown
    print("[*] Shutting down ClauseGuard API Server...")
    if hasattr(app.state, "audit_service"):
        del app.state.audit_service
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[✓] Shutdown complete.")


# -----------------------------------------------------------------------------
# FastAPI App & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ClauseGuard Legal Contract Audit API",
    description="Automated legal contract review pipeline using LegalBERT classification, NuExtract 2.0 extraction, and deterministic policy rule checking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js web frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production domains or Cloudflare tunnel origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------
def get_audit_service(request: Request) -> ContractAuditService:
    """Dependency provider returning the shared ContractAuditService instance."""
    service = getattr(request.app.state, "audit_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ContractAuditService is not yet loaded or initialized.",
        )
    return service


def save_upload_to_temp(file: UploadFile) -> str:
    """Validates that the upload is a PDF and writes it to a temporary file."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Received: {filename}",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        shutil.copyfileobj(file.file, temp_file)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Root status endpoint."""
    return {
        "app": "ClauseGuard Contract Audit API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check returning system status, GPU device info, and current VRAM memory.
    """
    cuda_available = torch.cuda.is_available()
    vram_alloc = round(torch.cuda.memory_allocated() / (1024**2), 2) if cuda_available else 0
    vram_res = round(torch.cuda.memory_reserved() / (1024**2), 2) if cuda_available else 0

    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "vram_allocated_mb": vram_alloc,
        "vram_reserved_mb": vram_res,
        "models_loaded": hasattr(app.state, "audit_service"),
    }


@app.get("/api/rules", tags=["Rules"])
async def get_review_rules():
    """
    Returns the active set of deterministic contract compliance rules configured in the system.
    """
    if not RULES_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rules definition file not found at {RULES_PATH}",
        )

    try:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        return {
            "total_rules": len(rules_data),
            "rules": rules_data,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read rules: {exc}",
        )


@app.post(
    "/api/audit",
    response_model=AuditReportSchema,
    tags=["Audit"],
    summary="Synchronous Contract Audit",
    description="Upload a contract PDF. Returns the full structured AuditReportSchema when processing finishes.",
)
async def audit_contract_sync(
    file: UploadFile = File(..., description="Contract PDF to analyze"),
    service: ContractAuditService = Depends(get_audit_service),
):
    """Executes the complete audit pipeline synchronously and returns the report."""
    temp_path = save_upload_to_temp(file)
    try:
        report = service.analyze_document(temp_path)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit failed during processing: {str(exc)}",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post(
    "/api/audit/stream",
    tags=["Audit"],
    summary="Real-Time Streaming Contract Audit (SSE)",
    description=(
        "Upload a contract PDF. Returns an HTTP Server-Sent Events (SSE) stream yielding "
        "step-by-step progress events (15%, 30%, 45%, 60%, 75%, 90%, 100%) and the final report."
    ),
)
async def audit_contract_stream(
    file: UploadFile = File(..., description="Contract PDF to analyze"),
    service: ContractAuditService = Depends(get_audit_service),
):
    """Streams live step progress events via Server-Sent Events (SSE)."""
    temp_path = save_upload_to_temp(file)

    def sse_event_generator() -> Generator[str, None, None]:
        try:
            for event in service.analyze_document_stream(temp_path):
                # Format as standard Server-Sent Event: "data: <json>\n\n"
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            error_payload = {
                "step": "error",
                "progress": 0,
                "message": f"Pipeline execution error: {str(exc)}",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering (Cloudflare / Nginx)
        },
    )


# -----------------------------------------------------------------------------
# Direct Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("[*] Launching ClauseGuard server on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
