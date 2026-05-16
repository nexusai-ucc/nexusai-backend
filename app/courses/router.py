"""
Endpoints de estadísticas de curso — BACK-13.

GET /api/v1/courses/{course_id}/stats
  Devuelve:
    - document_count: documentos con status='indexed'
    - chunk_count: total de chunks indexados del curso
    - last_indexed_at: timestamp del último documento indexado exitosamente
    - has_indexed_content: bool conveniente para que la UI sepa si hay material

El aislamiento multi-curso es garantizado por el filtro course_id en todos
los queries del retrieval RAG (retriever.py). Este endpoint permite que la
UI docente y alumno verifiquen el estado antes de intentar usar el chat.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import Chunk, Document
from app.db.session import get_db

router = APIRouter()


class CourseStats(BaseModel):
    course_id: int
    document_count: int
    chunk_count: int
    last_indexed_at: Optional[datetime]
    has_indexed_content: bool


@router.get("/{course_id}/stats", response_model=CourseStats)
async def get_course_stats(
    course_id: int,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> CourseStats:
    """Estadísticas de material indexado para un curso.

    Si el curso no existe o no tiene documentos indexados, devuelve ceros
    con has_indexed_content=False en lugar de 404 — el caller (UI del alumno
    o del docente) decide cómo mostrar eso.
    """
    if course_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="course_id debe ser positivo",
        )

    stmt = (
        select(
            func.count(distinct(Document.id)).label("document_count"),
            func.count(Chunk.id).label("chunk_count"),
            func.max(Document.updated_at).label("last_indexed_at"),
        )
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .where(Document.course_id == course_id)
        .where(Document.status == "indexed")
    )
    result = await db.execute(stmt)
    row = result.one()

    doc_count = int(row.document_count or 0)
    chunk_count = int(row.chunk_count or 0)

    return CourseStats(
        course_id=course_id,
        document_count=doc_count,
        chunk_count=chunk_count,
        last_indexed_at=row.last_indexed_at,
        has_indexed_content=doc_count > 0,
    )
