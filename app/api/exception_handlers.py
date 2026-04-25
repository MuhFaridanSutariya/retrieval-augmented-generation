from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    CacheError,
    DatabaseError,
    DocumentAlreadyExists,
    DocumentFileMissing,
    DocumentNotFound,
    EmbeddingError,
    EmptyQuery,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    MalformedLLMResponse,
    NoRelevantContext,
    QueryTooLong,
    TokenBudgetExceeded,
    ToolLoopExceeded,
    UnsupportedFileType,
    UploadTooLarge,
    VectorStoreError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


_STATUS_MAP: dict[type[AppError], int] = {
    EmptyQuery: 400,
    QueryTooLong: 400,
    UnsupportedFileType: 415,
    UploadTooLarge: 413,
    DocumentNotFound: 404,
    DocumentFileMissing: 404,
    DocumentAlreadyExists: 409,
    NoRelevantContext: 404,
    TokenBudgetExceeded: 413,
    ToolLoopExceeded: 502,
    LLMContentFilterError: 422,
    LLMRateLimitError: 429,
    LLMTimeoutError: 504,
    MalformedLLMResponse: 502,
    LLMError: 502,
    EmbeddingError: 502,
    VectorStoreError: 502,
    CacheError: 502,
    DatabaseError: 500,
}


def _status_for(exc: AppError) -> int:
    for cls in type(exc).__mro__:
        if cls in _STATUS_MAP:
            return _STATUS_MAP[cls]
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status_code = _status_for(exc)
        if status_code >= 500:
            logger.error(
                "app_error",
                error_type=type(exc).__name__,
                message=exc.message,
                details=exc.details,
                path=request.url.path,
            )
        else:
            logger.info(
                "app_error_client",
                error_type=type(exc).__name__,
                message=exc.message,
                path=request.url.path,
            )

        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )
