import logging
import os
import re
import time
from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

YUTORI_BASE_URL = "https://api.yutori.com"
POLL_INTERVAL = 2  # seconds
TIMEOUT_SECONDS = 90


def run_yutori_research(query: str) -> dict:
    """
    Call Yutori Research API to run a research task.
    
    1. Read YUTORI_API_KEY from environment
    2. POST to create research task
    3. Poll every 2 seconds until completion or timeout
    4. Extract results with titles, URLs, and detect PDFs
    5. Return structured response
    """
    api_key = os.getenv("YUTORI_API_KEY")
    if not api_key:
        logger.error("YUTORI_API_KEY environment variable not set")
        return {
            "task_id": None,
            "status": "failed",
            "error": "YUTORI_API_KEY not configured",
            "raw_output": None,
            "extracted_items": [],
        }
    
    # Create research task
    try:
        logger.info("Creating Yutori research task for query: %s", query)
        create_response = requests.post(
            f"{YUTORI_BASE_URL}/v1/research/tasks",
            headers={"X-API-Key": api_key},
            json={
                "query": query,
                "user_timezone": "America/Los_Angeles",
            },
            timeout=10,
        )
        create_response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to create Yutori research task: %s", str(e))
        return {
            "task_id": None,
            "status": "failed",
            "error": f"Failed to create research task: {str(e)}",
            "raw_output": None,
            "extracted_items": [],
        }
    
    try:
        task_data = create_response.json()
        task_id = task_data.get("task_id") or task_data.get("id")
        if not task_id:
            logger.error("No task_id in Yutori response: %s", task_data)
            return {
                "task_id": None,
                "status": "failed",
                "error": "No task_id returned from API",
                "raw_output": None,
                "extracted_items": [],
            }
    except Exception as e:
        logger.error("Failed to parse Yutori create response: %s", str(e))
        return {
            "task_id": None,
            "status": "failed",
            "error": f"Failed to parse API response: {str(e)}",
            "raw_output": None,
            "extracted_items": [],
        }
    
    logger.info("Created Yutori research task: %s", task_id)
    
    # Poll for completion
    start_time = time.time()
    poll_count = 0
    
    while time.time() - start_time < TIMEOUT_SECONDS:
        poll_count += 1
        time.sleep(POLL_INTERVAL)
        
        try:
            logger.debug("Polling Yutori task %s (attempt %d)", task_id, poll_count)
            poll_response = requests.get(
                f"{YUTORI_BASE_URL}/v1/research/tasks/{task_id}",
                headers={"X-API-Key": api_key},
                timeout=10,
            )
            poll_response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to poll Yutori task %s: %s", task_id, str(e))
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"Failed to poll task: {str(e)}",
                "raw_output": None,
                "extracted_items": [],
            }
        
        try:
            task_status = poll_response.json()
        except Exception as e:
            logger.error("Failed to parse Yutori poll response: %s", str(e))
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"Failed to parse poll response: {str(e)}",
                "raw_output": None,
                "extracted_items": [],
            }
        
        status_value = task_status.get("status", "unknown")
        logger.info("Yutori task %s status: %s", task_id, status_value)
        
        if status_value == "succeeded":
            # Extract results
            raw_output = task_status.get("result", {})
            extracted_items = _extract_items(raw_output)
            logger.info("Yutori task %s succeeded, extracted %d items", task_id, len(extracted_items))
            return {
                "task_id": task_id,
                "status": "succeeded",
                "raw_output": raw_output,
                "extracted_items": extracted_items,
            }
        elif status_value == "failed":
            error_msg = task_status.get("error", "Unknown error")
            logger.error("Yutori task %s failed: %s", task_id, error_msg)
            return {
                "task_id": task_id,
                "status": "failed",
                "error": error_msg,
                "raw_output": None,
                "extracted_items": [],
            }
        elif status_value not in ("queued", "running"):
            logger.warning("Yutori task %s has unexpected status: %s", task_id, status_value)
    
    # Timeout
    logger.warning("Yutori task %s timed out after %d seconds", task_id, TIMEOUT_SECONDS)
    return {
        "task_id": task_id,
        "status": "timeout",
        "error": f"Task did not complete within {TIMEOUT_SECONDS} seconds",
        "raw_output": None,
        "extracted_items": [],
    }


def _extract_items(raw_output: dict) -> List[dict]:
    """
    Extract simplified results from Yutori API response.
    
    Attempts to parse:
    - titles and URLs from response
    - PDF links using basic regex
    """
    extracted = []
    
    if not raw_output:
        return extracted
    
    # Handle various possible response structures
    # Try to find items in common Yutori response formats
    items = raw_output.get("items", [])
    if not items and "results" in raw_output:
        items = raw_output.get("results", [])
    if not items and isinstance(raw_output, dict):
        # Flatten single-level dict if it looks like results
        for key, value in raw_output.items():
            if isinstance(value, list):
                items = value
                break
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        title = item.get("title") or item.get("name") or ""
        source_url = item.get("url") or item.get("link") or item.get("source_url") or ""
        
        # Basic PDF detection with regex
        pdf_url = None
        for field in [item.get("pdf_url"), item.get("pdf_link"), source_url]:
            if field and _is_pdf_url(field):
                pdf_url = field
                break
        
        if title or source_url:
            extracted.append({
                "title": title.strip() if title else "",
                "source_url": source_url.strip() if source_url else "",
                "pdf_url": pdf_url,
            })
    
    return extracted


def _is_pdf_url(url: str) -> bool:
    """Check if URL appears to be a PDF link."""
    if not url:
        return False
    url_lower = url.lower()
    return url_lower.endswith(".pdf") or "/pdf" in url_lower


@router.get("", response_model=dict)
def search(q: str) -> dict:
    """
    Search research papers using Yutori Research API.
    
    Query parameter: q (search query)
    
    Returns:
    {
        "task_id": "...",
        "status": "succeeded|failed|timeout",
        "raw_output": {...},
        "extracted_items": [
            {
                "title": "...",
                "source_url": "...",
                "pdf_url": "..."
            }
        ]
    }
    """
    if not q or not q.strip():
        logger.warning("Search endpoint called with empty query")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' is required and cannot be empty",
        )
    
    logger.info("GET /search called with query: %s", q)
    result = run_yutori_research(q.strip())
    logger.info("Search result status: %s", result.get("status"))
    
    return result
