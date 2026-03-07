import json
import logging
from typing import List

from openai import OpenAI
from prefect import task
from pydantic import BaseModel, Field, ValidationError

from ..config import get_settings
from ..paper_ir import PaperIR

logger = logging.getLogger(__name__)


class RubricEntry(BaseModel):
    criterion_id: str
    description: str
    max_points: int
    assigned_points: int
    rationale: str
    evidence_fields: List[str] = Field(default_factory=list)


class PracticalImpactLLMOutput(BaseModel):
    rubric: List[RubricEntry]
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)


class PracticalImpactEvaluation(BaseModel):
    score: int
    priority_label: str
    rubric: List[RubricEntry]
    strengths: List[str]
    risks: List[str]
    questions_to_ask: List[str]


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot run practical impact evaluation."
        )
    return OpenAI(api_key=settings.openai_api_key)


@task(
    name="Evaluate Practical Impact",
    retries=2,
    retry_delay_seconds=5
)
def evaluate_practical_impact(paper_ir: PaperIR) -> PracticalImpactEvaluation:
    """
    Evaluate practical impact / priority based solely on paper_ir.
    """
    client = _get_client()

    logger.info("Practical Impact evaluation started")

    ir_json = paper_ir.model_dump()
    ir_text = json.dumps(ir_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are assessing the practical impact and reading priority of cochlear implant "
        "research papers for a practicing audiologist. You will receive a structured "
        "paper_ir object and must evaluate how important the paper is to read in full. "
        "Use ONLY information in paper_ir. Avoid hype and be realistic about impact."
    )

    rubric_description = (
        "You must evaluate exactly these criteria (each max_points=20):\n"
        "1) novelty: degree of new insights vs confirmatory/descriptive work.\n"
        "2) magnitude_of_findings: size and clinical importance of the effects reported.\n"
        "3) practice_changing_potential: likelihood the study would alter clinical protocols vs document existing practice.\n"
        "4) evidence_vs_claims_alignment: how well the strength of conclusions matches the underlying evidence.\n"
        "5) relevance_to_ci_subspecialists: breadth of relevance across CI clinicians vs narrow niche interest.\n"
    )

    user_prompt = (
        f"{rubric_description}\n\n"
        "Return a JSON object with EXACTLY these fields:\n"
        "{\n"
        '  \"rubric\": [\n'
        "    {\n"
        '      \"criterion_id\": \"novelty|magnitude_of_findings|practice_changing_potential|evidence_vs_claims_alignment|relevance_to_ci_subspecialists\",\n'
        '      \"description\": \"short phrase\",\n'
        '      \"max_points\": 20,\n'
        '      \"assigned_points\": int,\n'
        '      \"rationale\": \"short explanation\",\n'
        '      \"evidence_fields\": [\"paper_ir field names used\"]\n'
        "    },\n"
        "    ... one entry per criterion ...\n"
        "  ],\n"
        '  \"strengths\": [\"short bullet strings\"],\n'
        '  \"risks\": [\"short bullet strings\"],\n'
        '  \"questions_to_ask\": [\"short bullet strings\"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use ONLY the provided paper_ir JSON as evidence.\n"
        "- Descriptive retrospective complication studies are rarely high on practice_changing_potential.\n"
        "- Large sample size alone does not guarantee high impact; weigh effect sizes and novelty.\n"
        "- Do NOT assign impact beyond what results and claims justify.\n"
        "- assigned_points must be between 0 and max_points for each criterion.\n"
        "- Be concise in rationale and bullet strings.\n"
        "- evidence_fields should reference specific paper_ir fields, such as "
        "\"primary_outcomes\", \"main_claims\", \"key_numerical_results\", "
        "\"stated_limitations\", \"population_summary\".\n"
        "- Return ONLY the JSON object, with no extra commentary.\n\n"
        "paper_ir JSON:\n"
        "-----\n"
        f"{ir_text}\n"
        "-----\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty practical impact response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse practical impact JSON: %s", exc)
        raise RuntimeError("Practical impact schema validation failed.") from exc

    # Validate raw structure
    try:
        llm_output = PracticalImpactLLMOutput(**data)
    except ValidationError as exc:
        logger.warning("Practical impact failed Pydantic validation: %s", exc)
        raise RuntimeError("Practical impact schema validation failed.") from exc

    # Clamp assigned_points and compute score
    total_score = 0
    for entry in llm_output.rubric:
        original = entry.assigned_points
        if entry.assigned_points < 0:
            entry.assigned_points = 0
        if entry.assigned_points > entry.max_points:
            entry.assigned_points = entry.max_points
        logger.info(
            "Practical Impact criterion %s assigned_points=%d (original=%d, max=%d)",
            entry.criterion_id,
            entry.assigned_points,
            original,
            entry.max_points,
        )
        total_score += entry.assigned_points

    # Cap between 0 and 100
    total_score = max(0, min(100, total_score))

    # Priority label (Python only)
    if total_score >= 80:
        priority_label = "High"
    elif total_score >= 60:
        priority_label = "Selective"
    else:
        priority_label = "Low"

    logger.info(
        "Practical Impact final score=%d priority_label=%s",
        total_score,
        priority_label,
    )

    return PracticalImpactEvaluation(
        score=total_score,
        priority_label=priority_label,
        rubric=llm_output.rubric,
        strengths=llm_output.strengths,
        risks=llm_output.risks,
        questions_to_ask=llm_output.questions_to_ask,
    )

