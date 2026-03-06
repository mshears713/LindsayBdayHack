import json
import logging
from typing import List, Optional

from openai import OpenAI
from prefect import task
from pydantic import BaseModel, Field, ValidationError

from .config import get_settings

logger = logging.getLogger(__name__)


class CitationBlock(BaseModel):
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None


class SampleGroup(BaseModel):
    name: str
    n: Optional[int] = None


class SampleSizes(BaseModel):
    total: Optional[int] = None
    groups: List[SampleGroup] = Field(default_factory=list)


class PaperIR(BaseModel):
    citation: CitationBlock
    study_design_summary: Optional[str] = None
    population_summary: Optional[str] = None
    inclusion_exclusion: Optional[str] = None
    sample_sizes: SampleSizes
    primary_outcomes: List[str] = Field(default_factory=list)
    measurement_instruments: List[str] = Field(default_factory=list)
    main_claims: List[str] = Field(default_factory=list)
    key_numerical_results: List[str] = Field(default_factory=list)
    stated_limitations: List[str] = Field(default_factory=list)
    funding_statement: Optional[str] = None
    conflict_of_interest_statement: Optional[str] = None
    statistical_mentions: List[str] = Field(default_factory=list)


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run IR extraction.")
    return OpenAI(api_key=settings.openai_api_key)


@task(name="Extract Paper IR", retries=3, retry_delay_seconds=10)
def extract_paper_ir(extracted_text: str, classification_context: Optional[dict]) -> PaperIR:
    """
    Run the main canonical IR extraction.
    """
    client = _get_client()

    # Truncate long texts to control token usage (keep early sections)
    truncated = extracted_text[:25000]

    logger.info("IR extraction started. Truncated_length=%d", len(truncated))

    classification_str = json.dumps(classification_context or {}, ensure_ascii=False)

    system_prompt = (
        "You are a careful, literal research extractor for cochlear implant papers.\n"
        "Your job is to fill a structured schema called paper_ir using ONLY information that is "
        "explicitly present in the text. When a field is missing or unclear, use null or an empty list.\n"
        "Never infer or guess details that are not clearly stated. Keep all strings concise, "
        "without long paragraphs or interpretation.\n"
    )

    user_prompt = (
        "You will receive:\n"
        "1) A small classification context (paper_type, population, domain_focus, funding_detected).\n"
        "2) Extracted text from a cochlear implant research paper.\n\n"
        "Use the classification ONLY as background context; do NOT introduce new facts from it.\n"
        "Fill the following JSON schema exactly (no extra fields):\n"
        "{\n"
        '  \"citation\": {\n'
        '    \"title\": string|null,\n'
        '    \"authors\": [string],\n'
        '    \"year\": int|null,\n'
        '    \"journal\": string|null\n'
        "  },\n"
        '  \"study_design_summary\": string|null,\n'
        '  \"population_summary\": string|null,\n'
        '  \"inclusion_exclusion\": string|null,\n'
        '  \"sample_sizes\": {\n'
        '    \"total\": int|null,\n'
        '    \"groups\": [ { \"name\": string, \"n\": int|null } ]\n'
        "  },\n"
        '  \"primary_outcomes\": [string],\n'
        '  \"measurement_instruments\": [string],\n'
        '  \"main_claims\": [string],\n'
        '  \"key_numerical_results\": [string],\n'
        '  \"stated_limitations\": [string],\n'
        '  \"funding_statement\": string|null,\n'
        '  \"conflict_of_interest_statement\": string|null,\n'
        '  \"statistical_mentions\": [string]\n'
        "}\n\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly present in the text.\n"
        "- If a value is not present, use null (for single fields) or an empty list.\n"
        "- Do NOT infer publication year unless a date is clearly stated (e.g., 'Published on: 09 January 2024').\n"
        "- Do NOT invent authors or journals.\n"
        "- key_numerical_results should be short, literal snippets of results (no interpretation).\n"
        "- stated_limitations must be the authors' explicit limitations only.\n"
        "- funding_statement and conflict_of_interest_statement must be null if not clearly mentioned.\n"
        "- statistical_mentions should list explicit references to p-values, tests, or 'significant' claims.\n"
        "- Avoid language like 'this suggests' or interpretation.\n"
        "- Return ONLY the JSON object, with no extra commentary.\n\n"
        f"Classification context (for background only):\n{classification_str}\n\n"
        "Extracted text:\n"
        "-----\n"
        f"{truncated}\n"
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
        raise RuntimeError("Empty IR extraction response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse IR extraction JSON: %s", exc)
        raise RuntimeError("IR extraction schema validation failed.") from exc

    # Strict schema validation
    try:
        paper_ir = PaperIR(**data)
    except ValidationError as exc:
        logger.warning("IR extraction failed Pydantic validation: %s", exc)
        raise RuntimeError("IR extraction schema validation failed.") from exc

    logger.info(
        "IR schema validation success. claims=%d results=%d limitations=%d",
        len(paper_ir.main_claims),
        len(paper_ir.key_numerical_results),
        len(paper_ir.stated_limitations),
    )
    logger.info("IR extraction completed.")

    return paper_ir

