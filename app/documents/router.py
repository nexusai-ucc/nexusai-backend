"""
Document upload + status endpoints.

Flujo:
  1. El docente desde Moodle sube un PDF (vía External Function PHP).
  2. PHP baja el archivo del file API de Moodle, lo encodea en base64, y POSTea
     un JSON al backend (más simple que multipart con HMAC, ver decisión abajo).
  3. Este endpoint crea el row `documents` con status='pending', dispara la
     indexación con BackgroundTasks (no bloquea la respuesta), y devuelve el
     document_id al cliente para que pueda hacer polling de estado.
  4. El docente ve el progreso (pending → indexing → indexed | error) en la UI.

Decisión: JSON+base64 vs multipart/form-data:
  - Multipart con HMAC requiere generar manualmente el body con boundary fija
    en PHP, calcular HMAC sobre el body completo, y enviar con cURL. Frágil:
    cualquier diferencia de line ending (LF vs CRLF) o whitespace rompe la
    firma.
  - JSON con base64 tiene 33% overhead, pero el body es predictible (mismo
    string que se firma y que se envía). Para MVP con PDFs <10MB es aceptable.
  - Cuando lleguemos a archivos grandes (>50MB) en producción, evaluar
    pre-signed URLs a S3/storage + metadata en JSON.

Por qué BackgroundTasks (FastAPI nativo) y no Celery:
  - Indexar 1 PDF de 50 pág tarda 30-60 s. BackgroundTasks corre en el mismo
    event loop pero después de devolver la response, así que el cliente no
    espera.
  - Para volúmenes serios (cientos de PDFs simultáneos), conviene Celery con
    worker pool. Eso es post-MVP — ver ADR-001 (monolito modular).
"""

from __future__ import annotations

import base64
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import Document
from app.db.session import get_db, get_session_factory
from app.documents.pipeline import index_document
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

# Límite de tamaño para el upload. PDFs típicos universitarios pesan 1-5 MB.
# Permitimos hasta 20 MB → en base64 son ~27 MB en el JSON, manejable.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class DocumentUploadRequest(BaseModel):
    """Payload de upload firmado con HMAC."""

    course_id: int = Field(gt=0, description="ID del curso de Moodle")
    uploader_id: int = Field(gt=0, description="$USER->id real del docente (no del cliente)")
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(default="application/pdf")
    content_b64: str = Field(
        min_length=1,
        description="Archivo encodeado en base64. Tamaño máximo decodeado: 20 MB",
    )


class DocumentOut(BaseModel):
    """Estado de un documento — devuelto en POST y GET."""

    id: UUID
    course_id: int
    uploader_id: int
    filename: str
    mime_type: str
    status: str  # pending | indexing | indexed | error
    error_message: Optional[str] = None

    @classmethod
    def from_orm(cls, doc: Document) -> "DocumentOut":
        return cls(
            id=doc.id,
            course_id=doc.course_id,
            uploader_id=doc.uploader_id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            status=doc.status,
            error_message=doc.error_message,
        )


# ============================================================
# Background task wrapper
# ============================================================

async def _index_document_task(
    document_id: UUID,
    file_bytes: bytes,
    embeddings: EmbeddingProvider,
) -> None:
    """
    Wrapper para correr index_document() en background.

    BackgroundTasks ejecuta la corutina DESPUÉS de devolver la response al
    cliente, pero comparte el event loop. Como `index_document()` necesita una
    AsyncSession y el get_db() de FastAPI ya cerró su sesión cuando termina la
    request, abrimos una sesión nueva acá.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            await index_document(
                document_id=document_id,
                file_bytes=file_bytes,
                db=session,
                embeddings=embeddings,
            )
        except Exception as exc:
            # El pipeline ya marca el document como 'error' en la DB y commitea
            # antes de re-raise. Solo logueamos acá.
            import traceback
            print(
                f"[NexusAI] Indexing failed for document {document_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()


# ============================================================
# Endpoints
# ============================================================

@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    payload: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentOut:
    """
    Sube un documento y dispara la indexación en background.

    Devuelve 202 Accepted con el document_id para que el cliente haga polling
    al endpoint GET /documents/{id} y vea cuando termine.

    Aceptamos solo PDF en MVP. DOCX y TXT en Sprint 3.
    """
    # Validación de mime_type. Soft check: confiamos en el header del cliente
    # pero rechazamos si claramente no es PDF.
    if payload.mime_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF is supported in MVP. Got: {payload.mime_type}",
        )

    # Decodear base64. Si está corrupto, rechazamos con 400.
    try:
        file_bytes = base64.b64decode(payload.content_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 content: {exc}",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {len(file_bytes)} bytes (max {MAX_UPLOAD_BYTES})",
        )

    # Sanity check del header magic bytes — los PDFs empiezan con "%PDF-".
    # Defensa adicional contra archivos mal labelados.
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not look like a valid PDF (magic bytes mismatch)",
        )

    # Crear el row en la DB con status='pending'.
    document = Document(
        course_id=payload.course_id,
        uploader_id=payload.uploader_id,
        filename=payload.filename,
        mime_type=payload.mime_type,
        status="pending",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Disparar la indexación async — la response sale antes de que termine.
    background_tasks.add_task(
        _index_document_task,
        document_id=document.id,
        file_bytes=file_bytes,
        embeddings=embeddings,
    )

    return DocumentOut.from_orm(document)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document_status(
    document_id: UUID,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    """
    Estado actual de un documento. Útil para polling desde la UI docente.

    Estados posibles: pending, indexing, indexed, error.
    Cuando es 'error', `error_message` tiene el detalle.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentOut.from_orm(document)


@router.get("", response_model=list[DocumentOut])
async def list_documents_by_course(
    course_id: int,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> list[DocumentOut]:
    """
    Lista todos los documentos de un curso. Para mostrar la tabla en la vista docente.
    """
    if course_id <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="course_id must be positive")

    result = await db.execute(
        select(Document).where(Document.course_id == course_id).order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    return [DocumentOut.from_orm(d) for d in documents]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,  # FastAPI quirk: 204 + tipo None requiere response_class explícita
)
async def delete_document(
    document_id: UUID,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Borra un documento + cascada en chunks (gracias al ON DELETE CASCADE en el FK).

    Devuelve 204 No Content sin body — convención REST para deletes exitosos.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await db.delete(document)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
