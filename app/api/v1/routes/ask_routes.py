from fastapi import APIRouter, Depends

from app.dependencies import get_ask_service
from app.models.mappers import answer_to_response
from app.models.schema.ask_schema import AskRequest, AskResponse
from app.services.ask_service import AskService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    ask_service: AskService = Depends(get_ask_service),
) -> AskResponse:
    result = await ask_service.ask(
        question=payload.question,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
    )
    return answer_to_response(result.answer, result.request_id)
