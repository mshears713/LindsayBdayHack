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


class MethodologicalSoundnessLLMOutput(BaseModel):
    rubric: List[RubricEntry]
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)


class MethodologicalSoundnessEvaluation(BaseModel):
    score: int
    rubric: List[RubricEntry]
    strengths: List[str]
    risks: List[str]
    questions_to_ask: List[str]


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot run methodological soundness evaluation."
        )
    return OpenAI(api_key=settings.openai_api_key)


@task(name="Evaluate Methodological Soundness", retries=3, retry_delay_seconds=10)
def evaluate_methodological_soundness(paper_ir: PaperIR) -> MethodologicalSoundnessEvaluation:
    """
    Evaluate methodological soundness based solely on paper_ir.
    """
    client = _get_client()

    logger.info("Methodological Soundness evaluation started")

    ir_json = paper_ir.model_dump()
    ir_text = json.dumps(ir_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a methodological rigor reviewer for cochlear implant research papers. "
        "You will receive a structured paper_ir object and must evaluate methodological "
        "soundness using a small rubric. Use ONLY the information provided in paper_ir. "
        "If design details are missing or unclear, deduct points. Do not invent methods or controls."
    )

    rubric_description = (
        "You must evaluate exactly these criteria (each max_points=20):\n"
        "1) study_design_strength: prospective vs retrospective vs RCT, presence of control/comparator.\n"
        "2) inclusion_exclusion_clarity: clarity and completeness of inclusion/exclusion criteria.\n"
        "3) confounder_handling: explicit handling of confounders or use of multivariate analysis.\n"
        "4) followup_outcome_quality: description and adequacy of follow-up and outcome assessment.\n"
        "5) internal_validity_risks: potential bias, selection issues, and data completeness concerns.\n"
    )

    user_prompt = (
        f"{rubric_description}\n\n"
        "Return a JSON object with EXACTLY these fields:\n"
        "{\n"
        '  \"rubric\": [\n'
        "    {\n"
        '      \"criterion_id\": \"study_design_strength|inclusion_exclusion_clarity|confounder_handling|followup_outcome_quality|internal_validity_risks\",\n'
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
        "- Deduct points for retrospective design without controls.\n"
        "- If confounders are not addressed explicitly, deduct points.\n"
        "- Do NOT invent control groups or multivariate analyses.\n"
        "- If follow-up or outcome assessment is unclear, deduct points.\n"
        "- assigned_points must be between 0 and max_points for each criterion.\n"
        "- Be concise in rationale and bullet strings.\n"
        "- evidence_fields should reference specific paper_ir fields, such as "
        "\"study_design_summary\", \"inclusion_exclusion\", \"population_summary\", "
        "\"sample_sizes\", \"stated_limitations\", etc.\n"
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
        raise RuntimeError("Empty methodological soundness response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse methodological soundness JSON: %s", exc)
        raise RuntimeError("Methodological soundness schema validation failed.") from exc

    # Validate raw structure
    try:
        llm_output = MethodologicalSoundnessLLMOutput(**data)
    except ValidationError as exc:
        logger.warning("Methodological soundness failed Pydantic validation: %s", exc)
        raise RuntimeError("Methodological soundness schema validation failed.") from exc

    # Clamp assigned_points and compute score
    total_score = 0
    for entry in llm_output.rubric:
        original = entry.assigned_points
        if entry.assigned_points < 0:
            entry.assigned_points = 0
        if entry.assigned_points > entry.max_points:
            entry.assigned_points = entry.max_points
        logger.info(
            "Methodological Soundness criterion %s assigned_points=%d (original=%d, max=%d)",
            entry.criterion_id,
            entry.assigned_points,
            original,
            entry.max_points,
        )
        total_score += entry.assigned_points

    # Cap between 0 and 100
    total_score = max(0, min(100, total_score))

    logger.info("Methodological Soundness final score=%d", total_score)

    return MethodologicalSoundnessEvaluation(
        score=total_score,
        rubric=llm_output.rubric,
        strengths=llm_output.strengths,
        risks=llm_output.risks,
        questions_to_ask=llm_output.questions_to_ask,
    )

