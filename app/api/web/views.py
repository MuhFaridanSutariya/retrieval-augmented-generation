from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.dependencies import get_ask_service, get_document_service
from app.models.domain.answer import Answer
from app.models.domain.document import Document
from app.services.ask_service import AskService
from app.services.document_service import DocumentService
from app.utils.sample_pdf import SAMPLE_FILENAME, generate_sample_complex_pdf

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
) -> HTMLResponse:
    documents, _ = await document_service.list(limit=50, offset=0)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"documents": documents},
    )


@router.get("/web/documents", response_class=HTMLResponse)
async def list_documents(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
) -> HTMLResponse:
    documents, _ = await document_service.list(limit=50, offset=0)
    return templates.TemplateResponse(
        request,
        "_document_list.html",
        {"documents": documents},
    )


@router.post(
    "/web/upload",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> HTMLResponse:
    content = await file.read()
    filename = file.filename or "upload"
    document = await document_service.create(filename=filename, content=content)
    return await _render_document_list(request, document_service, latest=document)


@router.post(
    "/web/demo/run-complex",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
)
async def run_complex_demo(
    request: Request,
    document_service: DocumentService = Depends(get_document_service),
) -> HTMLResponse:
    pdf_bytes = generate_sample_complex_pdf()
    document = await document_service.create(filename=SAMPLE_FILENAME, content=pdf_bytes)
    return await _render_document_list(request, document_service, latest=document)


@router.post("/web/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(""),
    document_id: str = Form(""),
    enable_cot: str = Form(""),
    enable_rerank: str = Form(""),
    ask_service: AskService = Depends(get_ask_service),
) -> HTMLResponse:
    document_ids: list[UUID] | None = None
    if document_id.strip():
        try:
            document_ids = [UUID(document_id.strip())]
        except ValueError:
            document_ids = None

    # HTML checkboxes submit "on"/"true" when checked, nothing when unchecked.
    truthy = {"on", "true", "1", "yes"}
    use_cot = enable_cot.strip().lower() in truthy
    use_rerank = enable_rerank.strip().lower() in truthy

    try:
        result = await ask_service.ask(
            question=question,
            document_ids=document_ids,
            use_cot=use_cot,
            use_rerank=use_rerank,
        )
        return templates.TemplateResponse(
            request,
            "_chat_exchange.html",
            {
                "question": question,
                "answer": result.answer,
                "request_id": result.request_id,
                "error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "_chat_exchange.html",
            {
                "question": question,
                "answer": _empty_answer(),
                "request_id": None,
                "error": _classify_error(exc),
            },
        )


def _empty_answer() -> Answer:
    return Answer(
        text="",
        is_grounded=False,
        prompt_version="",
        model="",
    )


def _classify_error(exc: Exception) -> dict:
    name = type(exc).__name__
    return {
        "type": name,
        "message": getattr(exc, "message", str(exc)),
    }


async def _render_document_list(
    request: Request,
    document_service: DocumentService,
    *,
    latest: Document | None = None,
) -> HTMLResponse:
    documents, _ = await document_service.list(limit=50, offset=0)
    return templates.TemplateResponse(
        request,
        "_document_list.html",
        {"documents": documents, "latest_id": str(latest.id) if latest else None},
    )
