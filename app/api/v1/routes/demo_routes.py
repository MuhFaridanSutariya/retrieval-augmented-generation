from fastapi import APIRouter, Depends, status

from app.dependencies import get_document_service
from app.models.mappers import document_domain_to_response
from app.models.schema.document_schema import DocumentResponse
from app.services.document_service import DocumentService
from app.utils.sample_pdf import SAMPLE_FILENAME, generate_sample_complex_pdf

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post(
    "/run-complex",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def run_complex_demo(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    pdf_bytes = generate_sample_complex_pdf()
    document = await document_service.create(filename=SAMPLE_FILENAME, content=pdf_bytes)
    return document_domain_to_response(document)
