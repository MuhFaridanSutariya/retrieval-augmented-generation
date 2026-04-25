from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import Settings
from app.dependencies import (
    get_ask_service,
    get_conversation_store,
    get_document_service,
    get_settings_dep,
)
from app.models.domain.answer import Answer
from app.models.domain.document import Document
from app.services.ask_service import AskService
from app.services.document_service import DocumentService
from app.storages.conversation_store import ConversationStore
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
    enable_tools: str = Form(""),
    ask_service: AskService = Depends(get_ask_service),
    settings: Settings = Depends(get_settings_dep),
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
    use_tools = enable_tools.strip().lower() in truthy

    cookie_session_id = request.cookies.get(settings.chat_session_cookie_name)

    try:
        result = await ask_service.ask(
            question=question,
            document_ids=document_ids,
            use_cot=use_cot,
            use_rerank=use_rerank,
            use_tools=use_tools,
            session_id=cookie_session_id,
        )
        response = templates.TemplateResponse(
            request,
            "_chat_exchange.html",
            {
                "question": question,
                "answer": result.answer,
                "request_id": result.request_id,
                "error": None,
            },
        )
        response.set_cookie(
            key=settings.chat_session_cookie_name,
            value=result.session_id,
            max_age=settings.chat_session_cookie_max_age_seconds,
            httponly=True,
            samesite="lax",
        )
        return response
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


@router.post("/web/chat/clear", response_class=HTMLResponse)
async def clear_chat(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> HTMLResponse:
    cookie_session_id = request.cookies.get(settings.chat_session_cookie_name)
    if cookie_session_id:
        await conversation_store.clear(cookie_session_id)
    response = HTMLResponse(content="", status_code=status.HTTP_200_OK)
    response.delete_cookie(settings.chat_session_cookie_name)
    return response


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
