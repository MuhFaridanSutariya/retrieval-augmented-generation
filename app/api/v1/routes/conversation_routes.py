from fastapi import APIRouter, Depends, Request, Response, status

from app.core.config import Settings
from app.dependencies import get_conversation_store, get_settings_dep
from app.storages.conversation_store import ConversationStore

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(
    session_id: str,
    response: Response,
    settings: Settings = Depends(get_settings_dep),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> Response:
    await conversation_store.clear(session_id)
    response.delete_cookie(settings.chat_session_cookie_name)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_current_session(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings_dep),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> Response:
    cookie_session_id = request.cookies.get(settings.chat_session_cookie_name)
    if cookie_session_id:
        await conversation_store.clear(cookie_session_id)
    response.delete_cookie(settings.chat_session_cookie_name)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
