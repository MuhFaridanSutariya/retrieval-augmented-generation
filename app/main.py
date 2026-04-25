from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.application.health_routes import router as health_router
from app.api.exception_handlers import register_exception_handlers
from app.api.v1.routes.ask_routes import router as ask_router
from app.api.v1.routes.demo_routes import router as demo_router
from app.api.v1.routes.document_routes import router as document_router
from app.api.web.views import router as web_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.dependencies import build_container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger(__name__)

    container = build_container()
    app.state.container = container

    try:
        await container.faiss_store.ensure_index()
    except Exception as exc:
        logger.warning("faiss_ensure_index_failed", error=str(exc))

    try:
        await container.intent_classifier.warm()
    except Exception as exc:
        logger.warning("intent_classifier_warm_failed", error=str(exc))

    logger.info("app_started", environment=settings.environment)
    try:
        yield
    finally:
        await container.shutdown()
        logger.info("app_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Knowledge Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(ask_router, prefix=settings.api_v1_prefix)
    app.include_router(document_router, prefix=settings.api_v1_prefix)
    app.include_router(demo_router, prefix=settings.api_v1_prefix)
    app.include_router(web_router)

    return app


app = create_app()
