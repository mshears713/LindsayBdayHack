import logging
from typing import List

from fastapi import APIRouter, Body, HTTPException, status

from ..models import DiscoveryItemIn, IgnoreRequest, RefreshSummary
from ..storage import load_store, mark_ignored, upsert_discovered

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discover", tags=["discover"])


@router.get("", response_model=List[dict])
def get_discover_new() -> List[dict]:
    """
    Return only discovered items with status='new'.
    """
    logger.info("GET /discover called")
    store = load_store()
    discovered = store.get("discovered", [])
    new_items = [item for item in discovered if item.get("status") == "new"]
    logger.info("GET /discover returning %d new items", len(new_items))
    return new_items


@router.post("/refresh", response_model=RefreshSummary)
def refresh_discover(items: List[DiscoveryItemIn] = Body(...)) -> RefreshSummary:
    """
    Upsert discovery items.
    """
    logger.info("POST /discover/refresh called with %d items", len(items))
    if not isinstance(items, list):
        logger.warning("Invalid payload for /discover/refresh: not a list")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Body must be a JSON list of discovery items.",
        )

    _, summary = upsert_discovered(items)
    logger.info(
        "Refresh summary: added=%d updated=%d skipped_ignored=%d total_new=%d",
        summary["added"],
        summary["updated"],
        summary["skipped_ignored"],
        summary["total_new"],
    )
    return RefreshSummary(**summary)


@router.post("/ignore")
def ignore_discovery(req: IgnoreRequest) -> dict:
    """
    Mark a paper as ignored.
    """
    logger.info("POST /discover/ignore called for paper_id=%s", req.paper_id)
    if not req.paper_id:
        logger.warning("Missing paper_id in /discover/ignore")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'paper_id' is required.",
        )

    updated_store = mark_ignored(req.paper_id)
    logger.info("Marked paper_id=%s as ignored", req.paper_id)
    return {"ok": True, "paper_id": req.paper_id, "status": "ignored"}

