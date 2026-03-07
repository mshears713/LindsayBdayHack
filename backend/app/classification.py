import json
import logging
from typing import Optional

from openai import OpenAI
from prefect import task
from pydantic import BaseModel, ValidationError

from .config import get_settings

logger = logging.getLogger(__name__)


class Classification(BaseModel):
    paper_type: str
    population: str
    domain_focus: str
    funding_detected: str


ALLOWED_PAPER_TYPES = {
    "trial",
    "cohort",
    "case",
    "meta-analysis",
    "engineering",
    "theoretical",
    "retrospective",
    "prospective",
    "unclear",
}

ALLOWED_POPULATIONS = {"pediatric", "adult", "mixed", "unclear"}
ALLOWED_FUNDING = {"yes", "no", "unclear"}


def _get_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run classification.")
    return OpenAI(api_key=settings.openai_api_key)


@task(
    name="Classify Paper",
    retries=2,
    retry_delay_seconds=5
)
def classify_paper(extracted_text: str, total_characters: int, total_words: int) -> Classification:
    """
    Run a lightweight classification call using truncated extracted text.
    """
    client = _get_client()

    # Truncate to ~10k characters for classification
    truncated = extracted_text[:10000]

    logger.info(
        "Starting mini classification: truncated_length=%d total_characters=%d total_words=%d",
        len(truncated),
        total_characters,
        total_words,
    )

    system_prompt = (
        "You are a careful research methods assistant. "
        "Read the provided cochlear implant research paper text and classify it. "
        "Only use information explicitly present in the text. "
        "If you are not sure, use 'unclear'. "
        "Do not hallucinate funding or details that are not clearly stated."
    )

    user_prompt = (
        "Based only on the text below, produce a JSON object with EXACTLY these fields:\n"
        '{\n'
        '  "paper_type": "trial|cohort|case|meta-analysis|engineering|theoretical|retrospective|prospective|unclear",\n'
        '  "population": "pediatric|adult|mixed|unclear",\n'
        '  "domain_focus": "string (short phrase, max 12 words)",\n'
        '  "funding_detected": "yes|no|unclear"\n'
        '}\n'
        "- If you are unsure for any field, use 'unclear'.\n"
        "- Do NOT invent funding; only answer 'yes' if funding or sponsorship is explicitly stated.\n"
        "- Keep domain_focus concise (<= 12 words).\n"
        "- Return ONLY the JSON object, with no extra text.\n\n"
        "Text:\n"
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

    content: Optional[str] = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty classification response from model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logger.warning("Failed to parse classification JSON: %s", exc)
        raise RuntimeError("Classification schema validation failed.") from exc

    # Strict schema validation via Pydantic
    try:
        classification = Classification(**data)
    except ValidationError as exc:
        logger.warning("Classification failed Pydantic validation: %s", exc)
        raise RuntimeError("Classification schema validation failed.") from exc

    # Additional value checks for enumerated fields
    if classification.paper_type not in ALLOWED_PAPER_TYPES:
        logger.warning("Invalid paper_type value: %s", classification.paper_type)
        raise RuntimeError("Classification schema validation failed.")
    if classification.population not in ALLOWED_POPULATIONS:
        logger.warning("Invalid population value: %s", classification.population)
        raise RuntimeError("Classification schema validation failed.")
    if classification.funding_detected not in ALLOWED_FUNDING:
        logger.warning("Invalid funding_detected value: %s", classification.funding_detected)
        raise RuntimeError("Classification schema validation failed.")
    if len(classification.domain_focus.split()) > 12:
        logger.warning("domain_focus too long: %s", classification.domain_focus)
        raise RuntimeError("Classification schema validation failed.")

    logger.info(
        "Mini classification completed: paper_type=%s population=%s funding_detected=%s",
        classification.paper_type,
        classification.population,
        classification.funding_detected,
    )

    return classification

