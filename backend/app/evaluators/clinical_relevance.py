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


class ClinicalRelevanceLLMOutput(BaseModel):
    rubric: List[RubricEntry]
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)


class ClinicalRelevanceEvaluation(BaseModel):
    score: int
    rubric: List[RubricEntry]
    strengths: List[str]
    risks: List[str]
    questions_to_ask: List[str]


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot run clinical relevance evaluation."
        )
    return OpenAI(api_key=settings.openai_api_key)


def evaluate_clinical_relevance(paper_ir: PaperIR) -> ClinicalRelevanceEvaluation:
    """
    Evaluate clinical relevance based solely on paper_ir.
    """
    client = _get_client()

    logger.info("Clinical Relevance evaluation started")

    ir_json = paper_ir.model_dump()
    ir_text = json.dumps(ir_json, ensure_ascii=False, indent=2)

    system_prompt = (
        "You are a clinical relevance reviewer for cochlear implant research papers. "
        "You will receive a structured paper_ir object and must evaluate how clinically "
        "applicable and relevant the study is to real-world CI practice. "
        "Use ONLY the information provided in paper_ir. Do not assume applicability "
        "beyond what is clearly stated."
    )

    rubric_description = (
        "You must evaluate exactly these criteria (each max_points=20):\n"
        "1) population_applicability: how well the study population matches real-world CI patients; clarity of demographics.\n"
        "2) outcome_clinical_meaningfulness: whether outcomes are meaningful for patient care and treatment decisions.\n"
        "3) generalizability: extent to which findings generalize beyond the study site(s).\n"
        "4) practical_decision_utility: degree to which the study informs or changes clinical decisions vs being mainly descriptive.\n"
        "5) clarity_of_patient_impact: how clearly patient impact and risks are described and contextualized.\n"
    )

    user_prompt = (
        f"{rubric_description}\n\n"
        "Return a JSON object with EXACTLY these fields:\n"
        "{\n"
        '  \"rubric\": [\n'
        "    {\n"
        '      \"criterion_id\": \"population_applicability|outcome_clinical_meaningfulness|generalizability|practical_decision_utility|clarity_of_patient_impact\",\n'
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
        "- Retrospective descriptive studies should not receive full marks on practical_decision_utility.\n"
        "- If outcomes focus mainly on complication prevalence, score outcome_clinical_meaningfulness and practical_decision_utility moderately.\n"
        "- Do NOT assume broader generalizability than the sites and populations described.\n"
        "- assigned_points must be between 0 and max_points for each criterion.\n"
        "- Be concise in rationale and bullet strings.\n"
        "- evidence_fields should reference specific paper_ir fields, such as "
        "\"population_summary\", \"sample_sizes\", \"primary_outcomes\", "
        "\"main_claims\", \"key_numerical_results\", \"stated_limitations\".\n"
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
        raise RuntimeError("Empty clinical relevance response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse clinical relevance JSON: %s", exc)
        raise RuntimeError("Clinical relevance schema validation failed.") from exc

    # Validate raw structure
    try:
        llm_output = ClinicalRelevanceLLMOutput(**data)
    except ValidationError as exc:
        logger.warning("Clinical relevance failed Pydantic validation: %s", exc)
        raise RuntimeError("Clinical relevance schema validation failed.") from exc

    # Clamp assigned_points and compute score
    total_score = 0
    for entry in llm_output.rubric:
        original = entry.assigned_points
        if entry.assigned_points < 0:
            entry.assigned_points = 0
        if entry.assigned_points > entry.max_points:
            entry.assigned_points = entry.max_points
        logger.info(
            "Clinical Relevance criterion %s assigned_points=%d (original=%d, max=%d)",
            entry.criterion_id,
            entry.assigned_points,
            original,
            entry.max_points,
        )
        total_score += entry.assigned_points

    # Cap between 0 and 100
    total_score = max(0, min(100, total_score))

    logger.info("Clinical Relevance final score=%d", total_score)

    return ClinicalRelevanceEvaluation(
        score=total_score,
        rubric=llm_output.rubric,
        strengths=llm_output.strengths,
        risks=llm_output.risks,
        questions_to_ask=llm_output.questions_to_ask,
    )

