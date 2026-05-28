"""
Buscador semántico — Feature A.

POST /api/v1/search
  Recibe una consulta en lenguaje natural y devuelve los fragmentos
  del material del curso más similares semánticamente, sin generar
  texto con el LLM. Es retrieval puro sobre pgvector.

  Útil para:
  - Encontrar en qué archivo/sección está un tema
  - Ver el fragmento exacto del material sin pasar por el chat
  - Verificar qué material tiene el curso indexado
"""

from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.documents.retriever import retrieve_context
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    document_filename: str
    chunk_index: int
    content: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> SearchResponse:
    """Búsqueda semántica en el material del curso.

    Devuelve los fragmentos más relevantes sin pasar por el LLM. Si el curso
    no tiene material indexado, devuelve lista vacía (no 404).
    """
    try:
        chunks = await retrieve_context(
            question=payload.query,
            course_id=payload.course_id,
            db=db,
            embeddings=embeddings,
            top_k=payload.top_k,
            min_similarity=0.25,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de búsqueda no está disponible temporalmente",
        ) from exc

    results = [
        SearchResult(
            document_filename=chunk.document_filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content[:400].strip(),
            similarity=round(chunk.similarity, 3),
        )
        for chunk in chunks
    ]

    return SearchResponse(
        query=payload.query,
        results=results,
        total=len(results),
    )
