import logging
from fastapi import APIRouter

from ..models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    logger.info("GET /health called")
    return HealthResponse(status="ok")

