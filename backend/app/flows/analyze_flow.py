"""
Prefect flow orchestration for paper analysis pipeline.

This module defines the main Prefect flow that orchestrates the existing
analysis pipeline functions with proper task boundaries and parallel execution.
"""

import logging
from typing import Optional

from prefect import flow, task, get_run_logger

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
    ClassificationResult,
    Evaluations,
    ExtractionInfo,
)
from ..services.aggregation import build_aggregation

logger = logging.getLogger(__name__)


@flow(name="Analyze Paper Flow")
def analyze_paper_flow(
    meta: AnalyzeMeta,
    extraction_result,
    preview: Optional[str],
    warnings: list[str],
) -> AnalyzeResponse:
    """
    Main Prefect flow for paper analysis pipeline.
    
    This flow orchestrates the complete analysis pipeline from text extraction
    through evaluation and aggregation, with parallel execution of evaluators.
    
    Args:
        meta: Analysis metadata from upload/download
        extraction_result: Result from extract_pdf_text function
        preview: Optional text preview for dev mode
        warnings: Accumulated warning messages
        
    Returns:
        Complete AnalyzeResponse with all pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting Prefect analyze_paper_flow for request_id=%s", meta.request_id)
    
    try:
        # Create extraction info from extraction result
        extraction_info = ExtractionInfo(
            pages=extraction_result.pages,
            total_characters=extraction_result.total_characters,
            total_words=extraction_result.total_words,
            average_chars_per_page=extraction_result.average_chars_per_page,
        )
        
        # Step 1: Classification
        logger.info("Starting classification step")
        classification_result = classify_paper(
            extracted_text=extraction_result.text,
            total_characters=extraction_result.total_characters,
            total_words=extraction_result.total_words,
        )
        
        # Step 2: Paper IR extraction
        logger.info("Starting IR extraction step")
        classification_context = {
            "paper_type": classification_result.paper_type,
            "population": classification_result.population,
            "domain_focus": classification_result.domain_focus,
            "funding_detected": classification_result.funding_detected,
        }
        
        paper_ir = extract_paper_ir(
            extracted_text=extraction_result.text,
            classification_context=classification_context,
        )
        
        # Step 3: Parallel evaluator execution
        logger.info("Running statistical rigor evaluation")
        logger.info("Running methodological evaluation")
        logger.info("Running clinical relevance evaluation")
        logger.info("Running practical impact evaluation")
        
        # Run all evaluators in parallel
        statistical_eval = evaluate_statistical_rigor(paper_ir)
        methodological_eval = evaluate_methodological_soundness(paper_ir)
        clinical_eval = evaluate_clinical_relevance(paper_ir)
        practical_eval = evaluate_practical_impact(paper_ir)
        
        # Step 4: Aggregation
        logger.info("Aggregating evaluation results")
        evaluations = Evaluations(
            statistical_rigor=statistical_eval,
            methodological_soundness=methodological_eval,
            clinical_relevance=clinical_eval,
            practical_impact_priority=practical_eval,
        )
        
        aggregation = build_aggregation(evaluations)
        
        # Step 5: Build final response
        logger.info("Building final response")
        return AnalyzeResponse(
            meta=meta,
            extraction=extraction_info,
            preview=preview,
            classification=classification_result.model_dump(),
            paper_ir=paper_ir,
            evaluations=evaluations,
            aggregation=aggregation,
            warnings=warnings,
            error=None,
        )
        
    except Exception as exc:
        logger.error("Prefect flow failed: %s", exc)
        # Return error response matching existing pattern
        return AnalyzeResponse(
            meta=meta,
            extraction=extraction_info if 'extraction_info' in locals() else None,
            preview=preview,
            classification=None,
            paper_ir=None,
            evaluations=None,
            aggregation=None,
            warnings=warnings,
            error=str(exc),
        )


