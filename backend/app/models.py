from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, validator

from .paper_ir import PaperIR
from .evaluators.statistical_rigor import StatisticalRigorEvaluation
from .evaluators.methodological_soundness import MethodologicalSoundnessEvaluation
from .evaluators.clinical_relevance import ClinicalRelevanceEvaluation
from .evaluators.practical_impact_priority import PracticalImpactEvaluation


class DiscoveryItemIn(BaseModel):
    """Incoming discovery item payload from Yutori (or static list)."""

    paper_id: str
    title: str
    pdf_url: Optional[HttpUrl] = Field(default=None, alias="pdf_url")


class DiscoveryItemStore(BaseModel):
    """Representation of a discovered paper in the local store."""

    paper_id: str
    title: str
    pdf_url: Optional[str] = None
    added_at: datetime
    status: str = Field(pattern="^(new|ignored|analyzed)$")
    year: Optional[int] = None
    journal: Optional[str] = None

    @validator("paper_id", "title")
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class AnalyzedEntry(BaseModel):
    paper_id: str
    analyzed_at: datetime
    last_report: Optional[dict] = None


class ResearchStore(BaseModel):
    discovered: List[DiscoveryItemStore] = Field(default_factory=list)
    ignored_ids: List[str] = Field(default_factory=list)
    analyzed: List[AnalyzedEntry] = Field(default_factory=list)


class IgnoreRequest(BaseModel):
    paper_id: str


class RefreshSummary(BaseModel):
    added: int
    updated: int
    skipped_ignored: int
    total_new: int


class HealthResponse(BaseModel):
    status: str


class AnalyzeUrlRequest(BaseModel):
    paper_id: str
    mode: str


class AnalyzeMeta(BaseModel):
    mode: str
    filename: str
    content_type: str
    bytes: int
    saved_path: str
    timestamp: datetime
    request_id: UUID


class ExtractionInfo(BaseModel):
    pages: Optional[int] = None
    total_characters: int
    total_words: int
    average_chars_per_page: Optional[float] = None


class ClassificationResult(BaseModel):
    paper_type: str
    population: str
    domain_focus: str
    funding_detected: str


class Evaluations(BaseModel):
    statistical_rigor: Optional[StatisticalRigorEvaluation] = None
    methodological_soundness: Optional[MethodologicalSoundnessEvaluation] = None
    clinical_relevance: Optional[ClinicalRelevanceEvaluation] = None
    practical_impact_priority: Optional[PracticalImpactEvaluation] = None


class PracticalImpactSummary(BaseModel):
    score: int
    priority_label: str


class EvaluatorSummary(BaseModel):
    statistical_rigor: Optional[int] = None
    methodological_soundness: Optional[int] = None
    bias_promotional_risk: Optional[int] = None
    clinical_relevance: Optional[int] = None
    practical_impact_priority: Optional[PracticalImpactSummary] = None


class Aggregation(BaseModel):
    overall_score: int
    quality_band: str
    evaluator_summary: EvaluatorSummary
    top_strengths: List[str]
    top_risks: List[str]


class AnalyzeResponse(BaseModel):
    meta: Optional[AnalyzeMeta]
    extraction: Optional[ExtractionInfo] = None
    preview: Optional[str] = None
    classification: Optional[ClassificationResult] = None
    paper_ir: Optional[PaperIR] = None
    evaluations: Optional[Evaluations] = None
    aggregation: Optional[Aggregation] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None

