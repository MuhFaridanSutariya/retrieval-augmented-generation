from fastapi import APIRouter, Depends, Request, Response

from app.core.config import Settings
from app.dependencies import get_ask_service, get_settings_dep
from app.models.mappers import answer_to_response
from app.models.schema.ask_schema import AskRequest, AskResponse
from app.services.ask_service import AskService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    ask_service: AskService = Depends(get_ask_service),
    settings: Settings = Depends(get_settings_dep),
) -> AskResponse:
    cookie_session_id = request.cookies.get(settings.chat_session_cookie_name)
    effective_session_id = payload.session_id or cookie_session_id

    result = await ask_service.ask(
        question=payload.question,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        use_cot=payload.enable_cot,
        use_rerank=payload.enable_rerank,
        use_tools=payload.enable_tools,
        session_id=effective_session_id,
    )

    response.set_cookie(
        key=settings.chat_session_cookie_name,
        value=result.session_id,
        max_age=settings.chat_session_cookie_max_age_seconds,
        httponly=True,
        samesite="lax",
    )

    return answer_to_response(
        result.answer,
        result.request_id,
        session_id=result.session_id,
    )
