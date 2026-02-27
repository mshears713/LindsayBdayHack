import json
import logging
from typing import List

from openai import OpenAI
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


class StatisticalRigorLLMOutput(BaseModel):
    rubric: List[RubricEntry]
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)


class StatisticalRigorEvaluation(BaseModel):
    score: int
    rubric: List[RubricEntry]
    strengths: List[str]
    risks: List[str]
    questions_to_ask: List[str]


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run statistical rigor evaluation.")
    return OpenAI(api_key=settings.openai_api_key)


def evaluate_statistical_rigor(paper_ir: PaperIR) -> StatisticalRigorEvaluation:
    """
    Evaluate statistical rigor based solely on paper_ir.
    """
    client = _get_client()

    logger.info("Statistical Rigor evaluation started")

    ir_json = paper_ir.model_dump()
    ir_text = json.dumps(ir_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a statistical rigor reviewer for cochlear implant research papers. "
        "You will receive a structured paper_ir object and must evaluate statistical rigor "
        "using a small rubric. Use ONLY the information provided in paper_ir. "
        "If information is missing or unclear, deduct points. Do not invent statistics."
    )

    rubric_description = (
        "You must evaluate exactly these criteria:\n"
        "1) sample_size_adequacy (max_points=20): Is the total sample and any groups reasonably large for the questions asked?\n"
        "2) statistical_methods_clarity (max_points=20): Are statistical methods clearly described (tests, models, handling of confounders)?\n"
        "3) numerical_reporting (max_points=20): Are key numerical results reported clearly (effect sizes, confidence intervals, p-values)?\n"
        "4) significance_testing_use (max_points=20): Is the use of significance testing appropriate and transparently reported?\n"
        "5) claims_vs_evidence_alignment (max_points=20): Are the main claims aligned with the numerical evidence, without overclaiming?\n"
    )

    user_prompt = (
        f"{rubric_description}\n\n"
        "Return a JSON object with EXACTLY these fields:\n"
        "{\n"
        '  \"rubric\": [\n'
        "    {\n"
        '      \"criterion_id\": \"sample_size_adequacy|statistical_methods_clarity|numerical_reporting|significance_testing_use|claims_vs_evidence_alignment\",\n'
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
        "- Do not invent missing statistics or results.\n"
        "- If information for a criterion is weak or missing, assign lower points.\n"
        "- assigned_points must be between 0 and max_points for each criterion.\n"
        "- Be concise in rationale and bullet strings.\n"
        "- evidence_fields should reference specific paper_ir fields, such as "
        "\"sample_sizes\", \"key_numerical_results\", \"statistical_mentions\", "
        "\"stated_limitations\", etc.\n"
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
        raise RuntimeError("Empty statistical rigor response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse statistical rigor JSON: %s", exc)
        raise RuntimeError("Statistical rigor schema validation failed.") from exc

    # Validate raw structure
    try:
        llm_output = StatisticalRigorLLMOutput(**data)
    except ValidationError as exc:
        logger.warning("Statistical rigor failed Pydantic validation: %s", exc)
        raise RuntimeError("Statistical rigor schema validation failed.") from exc

    # Clamp assigned_points and compute score
    total_score = 0
    for entry in llm_output.rubric:
        original = entry.assigned_points
        if entry.assigned_points < 0:
            entry.assigned_points = 0
        if entry.assigned_points > entry.max_points:
            entry.assigned_points = entry.max_points
        logger.info(
            "Statistical Rigor criterion %s assigned_points=%d (original=%d, max=%d)",
            entry.criterion_id,
            entry.assigned_points,
            original,
            entry.max_points,
        )
        total_score += entry.assigned_points

    # Cap between 0 and 100
    total_score = max(0, min(100, total_score))

    logger.info("Statistical Rigor final score=%d", total_score)

    return StatisticalRigorEvaluation(
        score=total_score,
        rubric=llm_output.rubric,
        strengths=llm_output.strengths,
        risks=llm_output.risks,
        questions_to_ask=llm_output.questions_to_ask,
    )

