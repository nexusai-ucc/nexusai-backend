"""
Document upload + status endpoints — CONT-03: soporte PDF, DOCX y TXT.

Flujo:
  1. El docente desde Moodle sube un archivo (vía External Function PHP).
  2. PHP baja el archivo del file API de Moodle, lo encodea en base64, y POSTea
     un JSON al backend (más simple que multipart con HMAC).
  3. Este endpoint crea el row `documents` con status='pending', dispara la
     indexación con BackgroundTasks (no bloquea la respuesta), y devuelve el
     document_id al cliente para que pueda hacer polling de estado.
  4. El docente ve el progreso (pending → indexing → indexed | error) en la UI.

Tipos soportados (CONT-03):
  - application/pdf              → magic bytes %PDF-
  - application/vnd...docx       → magic bytes PK (ZIP)
  - text/plain                   → sin magic bytes (confiar en mime + extensión)

Por qué JSON+base64 y no multipart:
  Multipart con HMAC requiere generar manualmente el boundary en PHP y calcular
  HMAC sobre el body completo. Cualquier diferencia de line ending rompe la firma.
  JSON con base64 tiene 33% overhead pero el body es predecible. Para archivos
  <20MB es aceptable. Cuando escale a >50MB, evaluar pre-signed URLs a S3.
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
from app.documents.extractor import SUPPORTED_MIME_TYPES
from app.documents.pipeline import index_document
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

router = APIRouter()


# ============================================================
# Constantes
# ============================================================

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Magic bytes por mime type. None = sin verificación (ej. text/plain).
_MAGIC_BYTES: dict[str, bytes | None] = {
    "application/pdf": b"%PDF-",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": b"PK",
    "text/plain": None,
}


# ============================================================
# Schemas
# ============================================================

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

    Tipos aceptados: PDF, DOCX, TXT (máx 20 MB).
    Devuelve 202 Accepted con el document_id para polling.
    """
    if payload.mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Tipo de archivo no soportado: {payload.mime_type!r}. "
                f"Tipos aceptados: {sorted(SUPPORTED_MIME_TYPES)}"
            ),
        )

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

    # Verificación de magic bytes por tipo de archivo.
    # TXT no tiene magic bytes → None = skip check.
    expected_magic = _MAGIC_BYTES.get(payload.mime_type)
    if expected_magic is not None and not file_bytes.startswith(expected_magic):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"El archivo no parece ser del tipo declarado ({payload.mime_type!r}): "
                "los primeros bytes no coinciden con el formato esperado."
            ),
        )

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
    """Estado actual de un documento. Útil para polling desde la UI docente.

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
    """Lista todos los documentos de un curso. Para la tabla en la vista docente."""
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
    response_class=Response,
)
async def delete_document(
    document_id: UUID,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Borra un documento + cascada en chunks (ON DELETE CASCADE en FK).

    Devuelve 204 No Content sin body.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await db.delete(document)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
