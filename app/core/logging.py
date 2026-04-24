import hashlib
import logging
import sys

import structlog

from app.core.config import Settings


def mask_query(query: str) -> str:
    if not query:
        return ""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.upper(),
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
