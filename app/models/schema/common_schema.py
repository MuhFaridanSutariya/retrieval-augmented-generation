from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict = Field(default_factory=dict)


class PageParams(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
