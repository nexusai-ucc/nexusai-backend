"""Retrieval semántico contra pgvector.

Esta es la pieza que cierra el RAG: dado una pregunta del alumno y un curso,
devuelve los N chunks de material que mejor matchean semánticamente. Esos
chunks se inyectan al system prompt del LLM antes de generar la respuesta.

El query usa el operador `<=>` de pgvector que es la distancia coseno
(equivalente a 1 - similitud). El índice HNSW (migración 002) lo hace
rápido O(log N).

Filtro por curso (`documents.course_id = X`): es CRÍTICO para aislamiento
multi-curso. Sin esto, una pregunta de Cálculo podría traer chunks de
Programación. La query siempre debe joinear documents y filtrar por
course_id.

Filtro por status='indexed': previene devolver chunks parcialmente
indexados o de documents en estado 'error'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class RetrievedChunk:
    """Resultado del retrieval — un chunk + metadata para citas."""

    content: str
    document_filename: str
    chunk_index: int
    distance: float  # 0.0 = idéntico semánticamente, 2.0 = opuesto

    @property
    def similarity(self) -> float:
        """Convierte distancia coseno a similitud en [0, 1]."""
        # pgvector <=> devuelve distancia en [0, 2]. La similitud coseno está en
        # [-1, 1]. Para tener un score en [0, 1] usamos: similarity = 1 - distance/2.
        return max(0.0, min(1.0, 1.0 - self.distance / 2.0))


async def retrieve_context(
    question: str,
    course_id: int,
    db: AsyncSession,
    embeddings: EmbeddingProvider,
    top_k: int = 5,
    min_similarity: float = 0.3,
) -> List[RetrievedChunk]:
    """Devuelve los top_k chunks más similares a la pregunta, filtrados por curso.

    Args:
        question: la pregunta del alumno (string no vacío).
        course_id: ID del curso de Moodle. Solo se buscan chunks de documents
                   de ese curso.
        db: AsyncSession ya abierta (la inyecta FastAPI Dependency get_db).
        embeddings: instancia de EmbeddingProvider para vectorizar la pregunta.
        top_k: cantidad máxima de chunks a devolver. 5 es el default usual
               para balancear contexto vs ruido.
        min_similarity: filtra resultados con similitud baja. 0.3 deja pasar
                        chunks "razonablemente" relacionados; subir a 0.5 si
                        querés solo hits muy buenos.

    Returns:
        Lista de RetrievedChunk ordenada por similitud descendente. Vacía si
        el curso no tiene documentos indexados, o si ninguno supera
        min_similarity.

    Raises:
        ValueError: si question está vacío o top_k <= 0.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # 1. Vectorizar la pregunta.
    question_vector = await embeddings.embed(question)

    # 2. Query con join a documents para filtrar por course_id + status.
    # El operador `<=>` de pgvector es la distancia coseno.
    # SQLAlchemy 2.0 no tiene un operador nativo para esto, lo expresamos con
    # `Chunk.embedding.cosine_distance(...)` que viene de pgvector-python.
    stmt = (
        select(
            Chunk.content,
            Chunk.chunk_index,
            Document.filename,
            Chunk.embedding.cosine_distance(question_vector).label("distance"),
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.course_id == course_id)
        .where(Document.status == "indexed")
        .where(Chunk.embedding.is_not(None))
        .order_by("distance")
        .limit(top_k)
    )

    result = await db.execute(stmt)
    rows = result.all()

    # 3. Convertir a dataclasses + filtrar por similitud mínima.
    chunks = [
        RetrievedChunk(
            content=row.content,
            document_filename=row.filename,
            chunk_index=row.chunk_index,
            distance=float(row.distance),
        )
        for row in rows
    ]
    return [c for c in chunks if c.similarity >= min_similarity]


def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Formatea los chunks recuperados para inyectarlos al system prompt.

    Output ejemplo:

        FRAGMENTO 1 (de "apunte-derivadas.pdf"):
        Una derivada mide la tasa instantánea de cambio...

        FRAGMENTO 2 (de "teorema-fundamental.pdf"):
        El teorema fundamental del cálculo...

    Args:
        chunks: lista de RetrievedChunk (puede estar vacía).

    Returns:
        String formateado, listo para concatenar al system prompt. Si la lista
        está vacía, devuelve string vacío (el endpoint detecta este caso y
        agrega una nota al system prompt: "no hay material disponible").
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        # Truncar contenido si es muy largo (defensa contra prompt injection
        # via chunks ruidosos). 800 chars ~= 200 tokens, suficiente para
        # contexto sin saturar.
        content = chunk.content.strip()
        if len(content) > 800:
            content = content[:800] + "..."
        parts.append(
            f'FRAGMENTO {i} (de "{chunk.document_filename}", chunk #{chunk.chunk_index}):\n{content}'
        )

    return "\n\n".join(parts)
