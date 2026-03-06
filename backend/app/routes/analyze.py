import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile, status

from ..config import get_settings
from ..extractor import extract_pdf_text
from ..flows.analyze_flow import analyze_paper_flow
from ..services.aggregation import build_aggregation
from ..models import (
    Aggregation,
    AnalyzeMeta,
    AnalyzeResponse,
    AnalyzeUrlRequest,
    EvaluatorSummary,
    ExtractionInfo,
    ClassificationResult,
    Evaluations,
    PracticalImpactSummary,
)
from ..storage import load_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])

MAX_BYTES = 25 * 1024 * 1024  # 25MB




def _tmp_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    tmp = base_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _save_upload_to_tmp(upload: UploadFile) -> AnalyzeMeta:
    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    content_type = upload.content_type or ""
    if "pdf" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF.",
        )

    tmp_dir = _tmp_dir()
    request_id = uuid4()
    safe_name = f"upload_{request_id}.pdf"
    target = tmp_dir / safe_name

    logger.info("Saving uploaded PDF to %s", target)

    total_bytes = 0
    with target.open("wb") as f:
        while True:
            chunk = upload.file.read(8192)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_BYTES:
                f.close()
                target.unlink(missing_ok=True)
                logger.warning("Uploaded file exceeded 25MB limit (%d bytes).", total_bytes)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded PDF is too large (limit 25MB).",
                )
            f.write(chunk)

    logger.info("Upload saved: %s bytes=%d content_type=%s", target, total_bytes, content_type)

    now = datetime.now(timezone.utc)
    saved_path = str(target)
    return AnalyzeMeta(
        mode="upload",
        filename=upload.filename,
        content_type=content_type,
        bytes=total_bytes,
        saved_path=saved_path,
        timestamp=now,
        request_id=request_id,
    )


def _download_pdf_from_store(paper_id: str) -> AnalyzeMeta:
    if not paper_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'paper_id' is required.",
        )

    store = load_store()
    discovered = store.get("discovered", [])
    pdf_url = None
    for item in discovered:
        if item.get("paper_id") == paper_id:
            pdf_url = item.get("pdf_url")
            break

    if not pdf_url:
        logger.warning("No pdf_url found in store for paper_id=%s", paper_id)
        # Transport-only error message as required
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to download PDF. Please upload manually.",
        )

    logger.info("Attempting single HTTP GET for pdf_url=%s (paper_id=%s)", pdf_url, paper_id)
    try:
        resp = requests.get(pdf_url, stream=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HTTP GET failed for pdf_url=%s error=%s", pdf_url, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to download PDF. Please upload manually.",
        ) from exc

    if resp.status_code != 200:
        logger.warning(
            "HTTP GET returned non-200 for pdf_url=%s status=%s", pdf_url, resp.status_code
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to download PDF. Please upload manually.",
        )

    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower():
        logger.warning(
            "Content-Type does not look like PDF for pdf_url=%s content_type=%s",
            pdf_url,
            content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to download PDF. Please upload manually.",
        )

    tmp_dir = _tmp_dir()
    request_id = uuid4()
    safe_name = f"url_{paper_id}_{request_id}.pdf"
    target = tmp_dir / safe_name

    logger.info("Saving downloaded PDF to %s", target)

    total_bytes = 0
    try:
        with target.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_BYTES:
                    f.close()
                    target.unlink(missing_ok=True)
                    logger.warning(
                        "Downloaded file exceeded 25MB limit (%d bytes) from pdf_url=%s",
                        total_bytes,
                        pdf_url,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Downloaded PDF is too large (limit 25MB).",
                    )
                f.write(chunk)
    finally:
        resp.close()

    logger.info(
        "Download saved: %s bytes=%d content_type=%s pdf_url=%s",
        target,
        total_bytes,
        content_type,
        pdf_url,
    )

    now = datetime.now(timezone.utc)
    saved_path = str(target)
    return AnalyzeMeta(
        mode="url",
        filename=os.path.basename(pdf_url),
        content_type=content_type,
        bytes=total_bytes,
        saved_path=saved_path,
        timestamp=now,
        request_id=request_id,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_entrypoint(
    request: Request,
    file: UploadFile | None = File(default=None),
    body: AnalyzeUrlRequest | None = Body(default=None),
) -> AnalyzeResponse:
    """
    PDF ingestion + text extraction for /analyze.

    This layer handles:
    - File/URL transport (already implemented)
    - Text extraction
    - Simple quality metrics & scanned detection
    """
    logger.info("POST /analyze received")

    warnings: list[str] = []
    settings = get_settings()

    content_type = request.headers.get("content-type", "")
    logger.info("Content-Type for /analyze: %s", content_type)

    try:
        # 1) Transport: get PDF saved to disk
        if file is not None:
            logger.info("Analyze mode: upload")
            meta = _save_upload_to_tmp(file)
        elif body is not None:
            if body.mode != "url":
                logger.warning("Invalid mode in JSON analyze request: %s", body.mode)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="For JSON analyze requests, mode must be 'url'.",
                )
            logger.info("Analyze mode: url for paper_id=%s", body.paper_id)
            meta = _download_pdf_from_store(body.paper_id)
        else:
            logger.warning("Neither file upload nor JSON body provided to /analyze.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either a PDF file upload or JSON body { 'paper_id': '...', 'mode': 'url' } is required.",
            )

        # 2) Text extraction
        logger.info("Starting extraction phase for saved_path=%s", meta.saved_path)
        try:
            extraction_result = extract_pdf_text(meta.saved_path)
        except FileNotFoundError as exc:
            logger.warning("Saved PDF not found during extraction: %s", exc)
            return AnalyzeResponse(
                meta=meta,
                extraction=None,
                preview=None,
                warnings=warnings,
                error="Saved PDF not found for extraction.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during PDF extraction: %s", exc)
            return AnalyzeResponse(
                meta=meta,
                extraction=None,
                preview=None,
                warnings=warnings,
                error="Failed to extract text from PDF.",
            )

        total_chars = extraction_result.total_characters
        total_words = extraction_result.total_words

        # 3) Low-text / scanned heuristics
        if total_chars == 0:
            warning_msg = "No extractable text found. PDF may be scanned."
            warnings.append(warning_msg)
            logger.info("Scanned detection: %s", warning_msg)
            return AnalyzeResponse(
                meta=meta,
                extraction=None,
                preview=None,
                warnings=warnings,
                error=warning_msg,
            )

        if total_chars < 1000 or total_words < 200:
            warning_msg = "Extracted text is very low. This PDF may be scanned or image-based."
            warnings.append(warning_msg)
            logger.info("Low-text warning added: %s", warning_msg)

        extraction_info = ExtractionInfo(
            pages=extraction_result.pages,
            total_characters=total_chars,
            total_words=total_words,
            average_chars_per_page=extraction_result.average_chars_per_page,
        )

        # 4) Optional DEV preview
        if settings.dev_mode:
            raw_preview = extraction_result.text[:1200]
            # Simple whitespace cleanup for readability
            preview = " ".join(raw_preview.split())
        else:
            preview = None

        logger.info(
            "Extraction metrics: pages=%s total_characters=%d total_words=%d avg_chars_per_page=%s",
            extraction_info.pages,
            extraction_info.total_characters,
            extraction_info.total_words,
            extraction_info.average_chars_per_page,
        )

        # 5) Readability gate before Prefect flow
        if total_chars < 1500 or total_words < 300:
            warning_msg = "Insufficient readable text for reliable classification."
            warnings.append(warning_msg)
            logger.info("Readability gate failed: %s", warning_msg)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=None,
                warnings=warnings,
                error="Extraction quality too low to classify.",
            )

        # 6) Run Prefect flow for the main analysis pipeline
        logger.info("Starting Prefect analysis flow")
        try:
            response = analyze_paper_flow(
                meta=meta,
                extraction_result=extraction_result,
                preview=preview,
                warnings=warnings,
            )
            
            # Check if flow returned an error
            if response.error:
                logger.warning("Prefect flow completed with error: %s", response.error)
            
            return response
            
        except Exception as exc:  # noqa: BLE001
            logger.error("Prefect flow failed unexpectedly: %s", exc)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                warnings=warnings,
                error="Analysis pipeline failed.",
            )
    except HTTPException as http_exc:
        # Map HTTP errors to the error field format
        logger.info("Returning error from /analyze: %s", http_exc.detail)
        return AnalyzeResponse(
            meta=None,
            extraction=None,
            preview=None,
            classification=None,
            paper_ir=None,
            evaluations=None,
            warnings=warnings,
            error=str(http_exc.detail),
        )

