import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routes.health import router as health_router
from .routes.discover import router as discover_router
from .routes.analyze import router as analyze_router
from .routes.search import router as search_router


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    logger = logging.getLogger(__name__)

    app = FastAPI(title="CI Research Copilot Backend", debug=settings.dev_mode)

    logger.info("Starting FastAPI app with dev_mode=%s", settings.dev_mode)

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
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

