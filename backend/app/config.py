import os
from functools import lru_cache
from typing import List


class Settings:
    """Simple settings container for local development."""

    dev_mode: bool
    allowed_origins: List[str]
    openai_api_key: str | None

    def __init__(self) -> None:
        dev_env = os.getenv("DEV_MODE", "true").lower()
        self.dev_mode = dev_env in ("1", "true", "yes", "y")

        frontend_origin = os.getenv("FRONTEND_ORIGIN")

        # Always allow common local dev ports
        origins = {
            "http://localhost:3000",
            "http://localhost:5173",
        }
        if frontend_origin:
            origins.add(frontend_origin)

        self.allowed_origins = sorted(origins)

        self.openai_api_key = os.getenv("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

