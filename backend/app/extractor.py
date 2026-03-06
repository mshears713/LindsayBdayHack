import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prefect import task
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    pages: Optional[int]
    text: str
    total_characters: int
    total_words: int
    average_chars_per_page: Optional[float]


@task(name="Extract PDF Text", retries=2, retry_delay_seconds=5)
def extract_pdf_text(pdf_path: str) -> ExtractionResult:
    """
    Extract text and simple metrics from a PDF file.

    This is a lightweight, non-OCR extractor. If no text can be extracted,
    the caller should treat the PDF as likely scanned or image-based.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    logger.info("Starting PDF text extraction from %s", pdf_path)

    reader = PdfReader(str(path))
    num_pages = len(reader.pages) if reader.pages is not None else None

    texts = []
    for page_index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from page %d: %s", page_index, exc)
            page_text = ""
        texts.append(page_text)

    full_text = "\n".join(texts)
    total_characters = len(full_text)
    total_words = len(full_text.split())

    if num_pages and num_pages > 0:
        average_chars_per_page: Optional[float] = total_characters / float(num_pages)
    else:
        average_chars_per_page = None

    logger.info(
        "Extraction complete: pages=%s total_characters=%d total_words=%d avg_chars_per_page=%s",
        num_pages,
        total_characters,
        total_words,
        average_chars_per_page,
    )

    return ExtractionResult(
        pages=num_pages,
        text=full_text,
        total_characters=total_characters,
        total_words=total_words,
        average_chars_per_page=average_chars_per_page,
    )

