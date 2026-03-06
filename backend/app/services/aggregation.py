"""
Aggregation service for building analysis results.

This module contains shared logic for aggregating evaluation results
into the final response format. It's intentionally separated from
routes to avoid circular imports with Prefect flows.
"""

import logging
from typing import List, Set

from ..models import (
    Aggregation,
    Evaluations,
    EvaluatorSummary,
    PracticalImpactSummary,
)

logger = logging.getLogger(__name__)


def build_aggregation(evaluations: Evaluations) -> Aggregation:
    """
    Build aggregation from evaluation results.
    
    Args:
        evaluations: All evaluator results
        
    Returns:
        Aggregated results with scores, quality band, and summaries
    """
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

    seen: Set[str] = set()
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
