class AppError(Exception):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentNotFound(AppError):
    pass


class DocumentFileMissing(AppError):
    pass


class DocumentAlreadyExists(AppError):
    pass


class UnsupportedFileType(AppError):
    pass


class UploadTooLarge(AppError):
    pass


class EmptyQuery(AppError):
    pass


class QueryTooLong(AppError):
    pass


class NoRelevantContext(AppError):
    pass


class InfrastructureError(AppError):
    pass


class DatabaseError(InfrastructureError):
    pass


class CacheError(InfrastructureError):
    pass


class VectorStoreError(InfrastructureError):
    pass


class EmbeddingError(InfrastructureError):
    pass


class LLMError(InfrastructureError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMContentFilterError(LLMError):
    pass


class MalformedLLMResponse(LLMError):
    pass


class TokenBudgetExceeded(AppError):
    pass


class ToolError(AppError):
    pass


class ToolNotFound(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolLoopExceeded(ToolError):
    pass
