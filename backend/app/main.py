import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routes.health import router as health_router
from .routes.discover import router as discover_router
from .routes.analyze import router as analyze_router
from .routes.search import router as search_router


def load_environment():
    """Load environment variables from .env file."""
    # Try to load .env from the backend directory
    backend_dir = Path(__file__).resolve().parents[1]
    env_file = backend_dir / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
    else:
        print("No .env file found, using system environment variables")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def create_app() -> FastAPI:
    load_environment()
    configure_logging()
    settings = get_settings()
    logger = logging.getLogger(__name__)

    app = FastAPI(title="CI Research Copilot Backend", debug=settings.dev_mode)

    logger.info("Starting FastAPI app with dev_mode=%s", settings.dev_mode)

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS allowed origins: %s", settings.allowed_origins)

    # Routers
    app.include_router(health_router)
    app.include_router(discover_router)
    app.include_router(analyze_router)
    app.include_router(search_router)

    return app


app = create_app()

