"""
Lógica de resumen automático de documentos indexados (BUS-03).

Dado un document_id, recupera todos los chunks del documento (ordenados por
chunk_index), los concatena hasta un máximo de MAX_CHARS caracteres y pide
al LLM un resumen estructurado en el idioma del propio documento.

No hace embedding ni búsqueda semántica: lee los chunks en orden secuencial
para preservar la estructura original del documento.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.llm import LLMProvider

MAX_CHARS = 15_000
_SUMMARY_PROMPT_TEMPLATE = """\
Sos un asistente académico. Resumí el siguiente documento de forma clara y estructurada.

El resumen debe:
- Estar en el mismo idioma que el documento
- Tener entre 150 y 300 palabras
- Destacar los conceptos principales y la estructura del documento
- Ser útil para un estudiante que quiere saber de qué trata antes de leerlo
- No inventar información que no esté en el texto

DOCUMENTO: {filename}

CONTENIDO:
{content}
"""


async def summarize_document(
    document_id: UUID,
    course_id: int,
    db: AsyncSession,
    llm: LLMProvider,
) -> dict:
    """
    Genera un resumen del documento usando el LLM.

    Args:
        document_id: UUID del documento a resumir.
        course_id: ID del curso — se usa para validar que el documento pertenece
                   al curso del usuario (aislamiento multi-curso).
        db: sesión async de SQLAlchemy.
        llm: instancia de LLMProvider.

    Returns:
        dict con: document_id, document_filename, summary, chunks_used, total_chunks.

    Raises:
        LookupError: si el documento no existe o no pertenece al course_id.
        RuntimeError: si el LLM falla al generar el resumen.
    """
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = doc_result.scalar_one_or_none()

    if document is None:
        raise LookupError(f"Document {document_id} not found")
    if document.course_id != course_id:
        raise LookupError(f"Document {document_id} does not belong to course {course_id}")

    chunks_result = await db.execute(
        select(Chunk.content, Chunk.chunk_index)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    chunks = chunks_result.all()
    total_chunks = len(chunks)

    if total_chunks == 0:
        raise LookupError(f"Document {document_id} has no indexed chunks")

    concatenated = ""
    chunks_used = 0
    truncated = False
    for chunk in chunks:
        piece = chunk.content.strip()
        if len(concatenated) + len(piece) + 2 > MAX_CHARS:
            truncated = True
            break
        concatenated += piece + "\n\n"
        chunks_used += 1

    if truncated:
        concatenated += "[documento truncado para el resumen]"

    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        filename=document.filename,
        content=concatenated.strip(),
    )

    try:
        result = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        summary_text = result.text.strip()
    except Exception as exc:
        raise RuntimeError("LLM summary generation failed") from exc

    return {
        "document_id": str(document_id),
        "document_filename": document.filename,
        "summary": summary_text,
        "chunks_used": chunks_used,
        "total_chunks": total_chunks,
    }
