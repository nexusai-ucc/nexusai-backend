"""
Buscador híbrido (semántico + full-text) — Feature A.

POST /api/v1/search
  Combina similitud coseno (pgvector) con ts_rank (PostgreSQL full-text)
  para devolver los fragmentos más relevantes del material del curso.
  Score combinado: 0.6 * semantic + 0.4 * lexical.
  No genera texto con el LLM — es retrieval puro.

IMPORTANTE: esta función NO modifica retrieve_context() ni ninguna función
compartida con chat/quiz. La query híbrida vive exclusivamente aquí.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

router = APIRouter()

_MIN_COMBINED_SCORE = 0.35

_HYBRID_SQL = text("""
    SELECT
        c.document_id,
        c.content,
        c.chunk_index,
        d.filename        AS document_filename,
        d.course_id,
        1 - (c.embedding <=> CAST(:query_embedding AS vector))                          AS semantic_score,
        COALESCE(ts_rank(c.content_tsv, plainto_tsquery('simple', :query_text)), 0)    AS lexical_score,
        (
            0.6 * (1 - (c.embedding <=> CAST(:query_embedding AS vector))) +
            0.4 * COALESCE(ts_rank(c.content_tsv, plainto_tsquery('simple', :query_text)), 0)
        )                                                                                AS combined_score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE
        d.course_id IN :course_ids
        AND d.status = 'indexed'
        AND c.embedding IS NOT NULL
        AND (
            1 - (c.embedding <=> CAST(:query_embedding AS vector)) >= 0.32
            OR c.content_tsv @@ plainto_tsquery('simple', :query_text)
            OR d.filename ILIKE :filename_pattern
        )
    ORDER BY combined_score DESC
    LIMIT :top_k
""").bindparams(bindparam("course_ids", expanding=True))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    top_k: int = Field(default=5, ge=1, le=10)
    course_ids: Optional[List[int]] = Field(default=None)


class SearchResult(BaseModel):
    document_id: Optional[str] = None
    document_filename: str
    course_id: int = 0
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
    """Búsqueda híbrida (semántica + full-text) en el material del curso."""
    ids_to_query = payload.course_ids if payload.course_ids else [payload.course_id]
    ids_to_query = [i for i in ids_to_query if i > 0]
    if not ids_to_query:
        return SearchResponse(query=payload.query, results=[], total=0)

    try:
        question_vector = await embeddings.embed(payload.query)
        embedding_str = "[" + ",".join(str(x) for x in question_vector) + "]"

        result = await db.execute(
            _HYBRID_SQL,
            {
                "query_embedding": embedding_str,
                "query_text": payload.query,
                "course_ids": ids_to_query,
                "top_k": payload.top_k,
                "filename_pattern": f"%{payload.query.strip()}%",
            },
        )
        rows = result.all()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servicio de búsqueda no está disponible temporalmente",
        ) from exc

    seen: dict = {}
    for row in rows:
        score = float(row.combined_score)
        if score < _MIN_COMBINED_SCORE:
            continue
        doc_id = str(row.document_id)
        if doc_id not in seen or score > float(seen[doc_id].combined_score):
            seen[doc_id] = row
    deduped = sorted(seen.values(), key=lambda r: float(r.combined_score), reverse=True)

    results = [
        SearchResult(
            document_id=str(row.document_id) if row.document_id else None,
            document_filename=row.document_filename,
            course_id=row.course_id,
            chunk_index=row.chunk_index,
            content=row.content[:400].strip(),
            similarity=round(float(row.combined_score), 3),
        )
        for row in deduped
    ]

    return SearchResponse(
        query=payload.query,
        results=results,
        total=len(results),
    )
