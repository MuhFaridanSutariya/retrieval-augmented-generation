from pydantic import BaseModel, Field

from app.services.document_service import DocumentService
from app.tools.base import Tool


class ListDocumentsArgs(BaseModel):
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of documents to return. Defaults to 20.",
    )


def build_list_documents_tool(document_service: DocumentService) -> Tool:
    async def _handler(*, limit: int = 20) -> str:
        documents, total = await document_service.list(limit=limit, offset=0)
        if not documents:
            return "No documents are indexed yet."
        lines = [f"Total indexed: {total} document(s)."]
        for doc in documents:
            lines.append(
                f"- {doc.filename} "
                f"({doc.file_type.value}, {doc.size_bytes:,} bytes, "
                f"{doc.chunk_count} chunks, status={doc.status.value})"
            )
        return "\n".join(lines)

    return Tool(
        name="list_documents",
        description=(
            "Return the list of documents currently indexed in the knowledge base, including "
            "filename, type, size, chunk count, and status. Use this when the user asks "
            "meta-questions like 'what documents do I have?' or 'how many PDFs are indexed?'."
        ),
        args_model=ListDocumentsArgs,
        handler=_handler,
    )
