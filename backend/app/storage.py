import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .models import DiscoveryItemIn, DiscoveryItemStore, ResearchStore

logger = logging.getLogger(__name__)


def _store_path() -> Path:
    # research_store.json lives at backend/research_store.json
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "research_store.json"


def _default_store() -> ResearchStore:
    return ResearchStore()


def load_store() -> Dict:
    """Load research_store.json, creating a default one if missing or invalid."""
    path = _store_path()
    if not path.exists():
        logger.info("research_store.json not found; creating new default store.")
        store = _default_store()
        save_store(store.dict())
        return store.dict()

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
        store = ResearchStore(**data)
        logger.info(
            "Loaded research_store.json: %d discovered, %d ignored, %d analyzed",
            len(store.discovered),
            len(store.ignored_ids),
            len(store.analyzed),
        )
        return store.dict()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load research_store.json, recreating default. Error: %s", exc)
        store = _default_store()
        save_store(store.dict())
        return store.dict()


def save_store(store: Dict) -> None:
    """Atomically write the store dict to research_store.json."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="research_store_", suffix=".json", dir=str(path.parent))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, default=str)
        os.replace(tmp_path, path)
        logger.info("Saved research_store.json successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save research_store.json: %s", exc)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _index_discovered(store: Dict) -> Dict[str, int]:
    return {item["paper_id"]: idx for idx, item in enumerate(store.get("discovered", []))}


def upsert_discovered(items: Iterable[DiscoveryItemIn]) -> Tuple[Dict, Dict]:
    """
    Upsert discovered items by paper_id.

    Returns (updated_store, summary_dict).
    """
    store = load_store()
    rs = ResearchStore(**store)
    index = _index_discovered(store)

    added = 0
    updated = 0
    skipped_ignored = 0

    now = datetime.now(timezone.utc)

    for incoming in items:
        paper_id = incoming.paper_id
        if paper_id in rs.ignored_ids:
            skipped_ignored += 1
            logger.info("Skipping ignored paper_id=%s during refresh.", paper_id)
            continue

        data = DiscoveryItemStore(
            paper_id=paper_id,
            title=incoming.title,
            pdf_url=str(incoming.pdf_url) if incoming.pdf_url else None,
            added_at=now,
            status="new",
        )

        if paper_id in index:
            # Update existing record
            idx = index[paper_id]
            rs.discovered[idx].title = data.title
            rs.discovered[idx].pdf_url = data.pdf_url
            # Keep existing status (might already be analyzed)
            updated += 1
            logger.info("Updated discovered item paper_id=%s.", paper_id)
        else:
            rs.discovered.append(data)
            added += 1
            logger.info("Added new discovered item paper_id=%s.", paper_id)

    # Recompute total_new
    total_new = sum(1 for item in rs.discovered if item.status == "new")

    updated_store = rs.dict()
    save_store(updated_store)
    summary = {
        "added": added,
        "updated": updated,
        "skipped_ignored": skipped_ignored,
        "total_new": total_new,
    }
    return updated_store, summary


def mark_ignored(paper_id: str) -> Dict:
    """
    Mark a paper_id as ignored.

    Ensures paper_id is present in ignored_ids and sets status="ignored"
    on any matching discovered item.
    """
    if not paper_id:
        raise ValueError("paper_id must be non-empty")

    store = load_store()
    rs = ResearchStore(**store)

    changed = False

    if paper_id not in rs.ignored_ids:
        rs.ignored_ids.append(paper_id)
        changed = True
        logger.info("Added paper_id=%s to ignored_ids.", paper_id)

    for item in rs.discovered:
        if item.paper_id == paper_id and item.status != "ignored":
            item.status = "ignored"
            changed = True
            logger.info("Marked discovered item paper_id=%s as ignored.", paper_id)

    if changed:
        updated_store = rs.dict()
        save_store(updated_store)
        return updated_store

    # No change but still return current store
    return rs.dict()

