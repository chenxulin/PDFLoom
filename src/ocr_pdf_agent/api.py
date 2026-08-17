"""FastAPI upload, status, and artifact endpoints."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager, suppress
from hmac import compare_digest
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .config import Settings
from .jobs import JobManager

settings = Settings()
manager = JobManager(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    yield
    await manager.shutdown()


app = FastAPI(
    title="OCR PDF Agent",
    version="0.1.0",
    description=(
        "PaddleOCR + PDFMathTranslate v1 + serial LLM + typography and "
        "vector-table redraw service"
    ),
    lifespan=lifespan,
)


async def require_service_token(
    supplied: Annotated[str | None, Header(alias="X-OCR-PDF-Agent-Token")] = None,
) -> None:
    expected = settings.access_token
    if expected and (not supplied or not compare_digest(supplied, expected)):
        raise HTTPException(401, "Invalid service token")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ocr_pdf_agent",
        "paddleocr_configured": bool(settings.paddleocr_enabled and settings.ocr_token),
        "llm_configured": bool(settings.llm_key),
    }


@app.post(
    "/v1/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
async def create_job(
    file: Annotated[UploadFile, File(...)],
    target_language: Annotated[str | None, Form()] = None,
    source_language: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    filename = Path(file.filename or "upload.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Only PDF uploads are accepted")
    job_id = uuid4().hex
    overrides: dict[str, str] = {}
    language_pattern = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,8})?$")
    for field_name, value in (
        ("target_language", target_language),
        ("source_language", source_language),
    ):
        if value is None:
            continue
        normalized = value.strip()
        if field_name == "source_language" and normalized.casefold() == "auto":
            overrides[field_name] = "auto"
        elif not language_pattern.fullmatch(normalized):
            raise HTTPException(400, f"Invalid {field_name}")
        else:
            overrides[field_name] = normalized
    job_dir = manager.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    source = job_dir / "source.pdf"
    limit = settings.max_upload_mib * 1024 * 1024
    size = 0
    try:
        with source.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"PDF exceeds {settings.max_upload_mib} MiB limit")
                destination.write(chunk)
        with source.open("rb") as uploaded:
            magic = uploaded.read(5)
        if size < 5 or magic != b"%PDF-":
            raise HTTPException(400, "Uploaded content is not a valid PDF")
        (job_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0.0,
                    "message": "Waiting for a worker",
                    "input": {"filename": filename, "size_bytes": size},
                    "options": {
                        "source_language": overrides.get("source_language", settings.source_language),
                        "target_language": overrides.get("target_language", settings.target_language),
                    },
                    "artifacts": {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        source.unlink(missing_ok=True)
        (job_dir / "manifest.json").unlink(missing_ok=True)
        with suppress(OSError):
            job_dir.rmdir()
        raise
    finally:
        await file.close()
    manager.dispatch(job_id, source, settings_overrides=overrides)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/v1/jobs/{job_id}",
    }


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_service_token)])
async def get_job(job_id: str) -> dict[str, Any]:
    try:
        return manager.read(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Job not found") from None


@app.get(
    "/v1/jobs/{job_id}/artifacts/{artifact_name}",
    dependencies=[Depends(require_service_token)],
)
async def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
    allowed = {
        "translated": "translated.pdf",
        "bilingual": "bilingual.pdf",
        "manifest": "manifest.json",
        "ocr": "ocr_ppstructurev3.json",
        "ocr-input": "ocr_pdfmathtranslate_input.pdf",
        "ledger": "translation_ledger.json",
        "source": "source.pdf",
    }
    filename = allowed.get(artifact_name)
    if filename is None:
        raise HTTPException(404, "Artifact not found")
    try:
        path = manager.job_dir(job_id) / filename
    except ValueError:
        raise HTTPException(404, "Job not found") from None
    if not path.is_file():
        raise HTTPException(404, "Artifact not available")
    media_type = "application/json" if path.suffix == ".json" else "application/pdf"
    return FileResponse(path, media_type=media_type, filename=path.name)
