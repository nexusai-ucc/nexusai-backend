"""
Foros con IA — Épica 06.

Endpoints:

  POST /api/v1/forums/index-post
    Indexa (o re-indexa) un post de foro. Llamado por el observer PHP
    cada vez que un post se crea o edita en Moodle.
    Solo re-embeddea si el content_hash cambió (evita trabajo innecesario).

  DELETE /api/v1/forums/index-post/{post_id}
    Elimina el embedding de un post. Llamado por el observer PHP cuando
    el post se borra en Moodle.

  POST /api/v1/forums/similar-posts
    Recibe el texto de un post en redacción y devuelve los posts existentes
    en el mismo curso que sean semánticamente similares.
    Usado por el frontend para avisar al alumno antes de publicar.

Todos los endpoints llevan verify_hmac — contrato de seguridad invariante.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import ForumPostEmbedding
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.shared.config import get_settings

router = APIRouter()

_DEFAULT_SIMILARITY_THRESHOLD = 0.75
_DEFAULT_TOP_K = 5


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class IndexPostRequest(BaseModel):
    post_id: int = Field(gt=0, description="ID de mdl_forum_posts")
    discussion_id: int = Field(gt=0, description="ID de mdl_forum_discussions")
    course_id: int = Field(gt=0)
    content: str = Field(min_length=1, max_length=10_000)


class IndexPostResponse(BaseModel):
    post_id: int
    status: str  # "indexed" | "skipped" (sin cambios en contenido)


class SimilarPostsRequest(BaseModel):
    course_id: int = Field(gt=0)
    text: str = Field(min_length=10, max_length=5_000, description="Texto del post en redacción")
    exclude_post_id: Optional[int] = Field(
        default=None,
        description="Post a excluir de los resultados (útil al editar un post existente)",
    )
    threshold: float = Field(
        default=_DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similitud mínima para considerar duplicado",
    )
    top_k: int = Field(default=_DEFAULT_TOP_K, ge=1, le=10)


class SimilarPost(BaseModel):
    forum_post_id: int
    discussion_id: int
    similarity: float
    preview: str  # primeros 200 chars del contenido


class SimilarPostsResponse(BaseModel):
    similar_posts: List[SimilarPost]
    threshold_used: float


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_to_pg_literal(vector: List[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/index-post", response_model=IndexPostResponse, status_code=status.HTTP_200_OK)
async def index_post(
    payload: IndexPostRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> IndexPostResponse:
    """Indexa o actualiza el embedding de un post de foro.

    Si el post ya existe con el mismo contenido (igual content_hash),
    devuelve status='skipped' sin hacer nada. Si el contenido cambió,
    re-embeddea y actualiza.
    """
    content_hash = _sha256(payload.content)

    # Buscar si ya existe
    result = await db.execute(
        select(ForumPostEmbedding).where(
            ForumPostEmbedding.forum_post_id == payload.post_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None and existing.content_hash == content_hash:
        return IndexPostResponse(post_id=payload.post_id, status="skipped")

    try:
        vector = await embeddings.embed(payload.content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar el embedding del post",
        ) from exc

    if existing is None:
        record = ForumPostEmbedding(
            id=uuid.uuid4(),
            forum_post_id=payload.post_id,
            discussion_id=payload.discussion_id,
            course_id=payload.course_id,
            content_hash=content_hash,
            content=payload.content,
            embedding=vector,
        )
        db.add(record)
    else:
        existing.content_hash = content_hash
        existing.content = payload.content
        existing.embedding = vector

    await db.commit()
    return IndexPostResponse(post_id=payload.post_id, status="indexed")


@router.delete("/index-post/{post_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_post_embedding(
    _body: Annotated[bytes, Depends(verify_hmac)],
    post_id: int = Path(gt=0),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Elimina el embedding de un post cuando se borra en Moodle."""
    await db.execute(
        delete(ForumPostEmbedding).where(
            ForumPostEmbedding.forum_post_id == post_id
        )
    )
    await db.commit()


@router.post("/similar-posts", response_model=SimilarPostsResponse)
async def similar_posts(
    payload: SimilarPostsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> SimilarPostsResponse:
    """Detecta posts existentes similares al texto que el alumno está escribiendo.

    Usa similitud coseno sobre pgvector. Solo busca dentro del mismo curso.
    Devuelve lista vacía si no hay nada por encima del threshold.
    """
    try:
        vector = await embeddings.embed(payload.text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo vectorizar la consulta",
        ) from exc

    embedding_literal = _embedding_to_pg_literal(vector)

    sql = text("""
        SELECT
            fpe.forum_post_id,
            fpe.discussion_id,
            fpe.content,
            1 - (fpe.embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM forum_post_embeddings fpe
        WHERE
            fpe.course_id = :course_id
            AND fpe.embedding IS NOT NULL
            AND (:exclude_post_id IS NULL OR fpe.forum_post_id != :exclude_post_id)
            AND 1 - (fpe.embedding <=> CAST(:query_embedding AS vector)) >= :threshold
        ORDER BY fpe.embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_k
    """)

    try:
        result = await db.execute(
            sql,
            {
                "query_embedding": embedding_literal,
                "course_id": payload.course_id,
                "exclude_post_id": payload.exclude_post_id,
                "threshold": payload.threshold,
                "top_k": payload.top_k,
            },
        )
        rows = result.all()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error al buscar posts similares",
        ) from exc

    similar = [
        SimilarPost(
            forum_post_id=row.forum_post_id,
            discussion_id=row.discussion_id,
            similarity=round(float(row.similarity), 3),
            preview=row.content[:200].strip(),
        )
        for row in rows
    ]

    return SimilarPostsResponse(
        similar_posts=similar,
        threshold_used=payload.threshold,
    )
