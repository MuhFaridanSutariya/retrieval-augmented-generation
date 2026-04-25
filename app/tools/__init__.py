from app.core.config import Settings
from app.services.document_service import DocumentService
from app.tools.base import Tool, ToolCallResult, ToolRegistry
from app.tools.calculator_tool import CALCULATOR_TOOL
from app.tools.list_documents_tool import build_list_documents_tool


def build_default_registry(
    *,
    settings: Settings,
    document_service: DocumentService,
) -> ToolRegistry:
    registry = ToolRegistry(max_output_chars=settings.max_tool_output_chars)
    registry.register(CALCULATOR_TOOL)
    registry.register(build_list_documents_tool(document_service))
    return registry


__all__ = [
    "Tool",
    "ToolCallResult",
    "ToolRegistry",
    "build_default_registry",
]
