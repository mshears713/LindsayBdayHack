import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile, status

from ..config import get_settings
from ..extractor import extract_pdf_text
from ..classification import classify_paper
from ..paper_ir import extract_paper_ir
from ..evaluators.statistical_rigor import evaluate_statistical_rigor
from ..evaluators.methodological_soundness import evaluate_methodological_soundness
from ..evaluators.clinical_relevance import evaluate_clinical_relevance
from ..evaluators.practical_impact_priority import evaluate_practical_impact
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


def _build_aggregation(evaluations: Evaluations) -> Aggregation:
    scores = []
    for e in (
        evaluations.statistical_rigor,
        evaluations.methodological_soundness,
        # bias_promotional_risk not yet implemented
        evaluations.clinical_relevance,
        evaluations.practical_impact_priority,
    ):
        if e is not None:
            scores.append(e.score)

    overall_score = round(sum(scores) / len(scores)) if scores else 0

    if overall_score >= 80:
        quality_band = "High Quality"
    elif overall_score >= 60:
        quality_band = "Moderate Quality"
    else:
        quality_band = "Low Quality"

    seen: set[str] = set()
    top_strengths: list[str] = []
    top_risks: list[str] = []

    for e in (
        evaluations.statistical_rigor,
        evaluations.methodological_soundness,
        evaluations.clinical_relevance,
        evaluations.practical_impact_priority,
    ):
        if e is None:
            continue
        for s in e.strengths:
            key = s.strip().lower()
            if key not in seen and len(top_strengths) < 5:
                seen.add(key)
                top_strengths.append(s)

    seen = set()
    for e in (
        evaluations.statistical_rigor,
        evaluations.methodological_soundness,
        evaluations.clinical_relevance,
        evaluations.practical_impact_priority,
    ):
        if e is None:
            continue
        for r in e.risks:
            key = r.strip().lower()
            if key not in seen and len(top_risks) < 5:
                seen.add(key)
                top_risks.append(r)

    practical_summary = None
    if evaluations.practical_impact_priority is not None:
        practical_summary = PracticalImpactSummary(
            score=evaluations.practical_impact_priority.score,
            priority_label=evaluations.practical_impact_priority.priority_label,
        )

    return Aggregation(
        overall_score=overall_score,
        quality_band=quality_band,
        evaluator_summary=EvaluatorSummary(
            statistical_rigor=evaluations.statistical_rigor.score if evaluations.statistical_rigor else None,
            methodological_soundness=evaluations.methodological_soundness.score if evaluations.methodological_soundness else None,
            bias_promotional_risk=None,
            clinical_relevance=evaluations.clinical_relevance.score if evaluations.clinical_relevance else None,
            practical_impact_priority=practical_summary,
        ),
        top_strengths=top_strengths,
        top_risks=top_risks,
    )


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

        # 5) Readability gate before classification
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

        # 6) Mini classification call
        try:
            cls = classify_paper(
                extracted_text=extraction_result.text,
                total_characters=total_chars,
                total_words=total_words,
            )
        except RuntimeError as exc:
            # Explicit schema / config failures
            msg = str(exc)
            logger.warning("Classification failed: %s", msg)
            error_msg = (
                "Classification schema validation failed."
                if "schema validation failed" in msg.lower()
                else msg
            )
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=None,
                warnings=warnings,
                error=error_msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during classification: %s", exc)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=None,
                warnings=warnings,
                error="Classification failed.",
            )

        classification_result = ClassificationResult(
            paper_type=cls.paper_type,
            population=cls.population,
            domain_focus=cls.domain_focus,
            funding_detected=cls.funding_detected,
        )

        logger.info(
            "Classification added to response: paper_type=%s population=%s funding_detected=%s",
            classification_result.paper_type,
            classification_result.population,
            classification_result.funding_detected,
        )

        # 7) Canonical IR extraction
        logger.info("IR extraction started")
        classification_context = {
            "paper_type": classification_result.paper_type,
            "population": classification_result.population,
            "domain_focus": classification_result.domain_focus,
            "funding_detected": classification_result.funding_detected,
        }
        try:
            paper_ir = extract_paper_ir(
                extracted_text=extraction_result.text,
                classification_context=classification_context,
            )
        except RuntimeError as exc:
            msg = str(exc)
            logger.warning("IR extraction failed: %s", msg)
            warning_msg = "IR extraction schema validation failed."
            if "schema validation failed" in msg.lower():
                warnings.append(warning_msg)
                error_msg = warning_msg
            else:
                error_msg = msg
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=None,
                warnings=warnings,
                error=error_msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during IR extraction: %s", exc)
            warnings.append("IR extraction failed.")
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=None,
                warnings=warnings,
                error="IR extraction failed.",
            )

        logger.info(
            "IR extraction completed. claims=%d results=%d limitations=%d",
            len(paper_ir.main_claims),
            len(paper_ir.key_numerical_results),
            len(paper_ir.stated_limitations),
        )

        # 8) Statistical Rigor evaluation
        try:
            rigor_eval = evaluate_statistical_rigor(paper_ir)
        except RuntimeError as exc:
            msg = str(exc)
            logger.warning("Statistical Rigor evaluation failed: %s", msg)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=None,
                warnings=warnings,
                error=msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during Statistical Rigor evaluation: %s", exc)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=None,
                warnings=warnings,
                error="Statistical Rigor evaluation failed.",
            )

        # 9) Methodological Soundness evaluation
        try:
            method_eval = evaluate_methodological_soundness(paper_ir)
        except RuntimeError as exc:
            msg = str(exc)
            logger.warning("Methodological Soundness evaluation failed: %s", msg)
            evals = Evaluations(statistical_rigor=rigor_eval)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error=msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during Methodological Soundness evaluation: %s", exc)
            evals = Evaluations(statistical_rigor=rigor_eval)
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error="Methodological Soundness evaluation failed.",
            )

        # 10) Clinical Relevance evaluation
        try:
            clinical_eval = evaluate_clinical_relevance(paper_ir)
        except RuntimeError as exc:
            msg = str(exc)
            logger.warning("Clinical Relevance evaluation failed: %s", msg)
            evals = Evaluations(
                statistical_rigor=rigor_eval,
                methodological_soundness=method_eval,
            )
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error=msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during Clinical Relevance evaluation: %s", exc)
            evals = Evaluations(
                statistical_rigor=rigor_eval,
                methodological_soundness=method_eval,
            )
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error="Clinical Relevance evaluation failed.",
            )

        # 11) Practical Impact / Priority evaluation
        try:
            practical_eval = evaluate_practical_impact(paper_ir)
        except RuntimeError as exc:
            msg = str(exc)
            logger.warning("Practical Impact evaluation failed: %s", msg)
            evals = Evaluations(
                statistical_rigor=rigor_eval,
                methodological_soundness=method_eval,
                clinical_relevance=clinical_eval,
            )
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error=msg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during Practical Impact evaluation: %s", exc)
            evals = Evaluations(
                statistical_rigor=rigor_eval,
                methodological_soundness=method_eval,
                clinical_relevance=clinical_eval,
            )
            return AnalyzeResponse(
                meta=meta,
                extraction=extraction_info,
                preview=preview,
                classification=classification_result,
                paper_ir=paper_ir,
                evaluations=evals,
                warnings=warnings,
                error="Practical Impact evaluation failed.",
            )

        evaluations = Evaluations(
            statistical_rigor=rigor_eval,
            methodological_soundness=method_eval,
            clinical_relevance=clinical_eval,
            practical_impact_priority=practical_eval,
        )

        return AnalyzeResponse(
            meta=meta,
            extraction=extraction_info,
            preview=preview,
            classification=classification_result,
            paper_ir=paper_ir,
            evaluations=evaluations,
            aggregation=_build_aggregation(evaluations),
            warnings=warnings,
            error=None,
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

