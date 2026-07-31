"""
Lógica de resumen automático de documentos indexados (BUS-03) y de resumen
pre-parcial multi-documento (BUS-04).

Dado un document_id, recupera todos los chunks del documento (ordenados por
chunk_index), los concatena hasta un máximo de MAX_CHARS caracteres y pide
al LLM un resumen estructurado en el idioma del propio documento.

No hace embedding ni búsqueda semántica: lee los chunks en orden secuencial
para preservar la estructura original del documento.

BUS-04 reusa `summarize_document` documento por documento (sin caché — cada
resumen es un LLM call real) y hace un único LLM call final de síntesis para
combinarlos en un solo resumen de repaso, mismo criterio de "una sola llamada
de síntesis" ya usado en Study Plan y en el FAQ dashboard.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.llm import LLMProvider

logger = logging.getLogger("nexusai.documents.summarizer")

MAX_CHARS = 20_000
_SUMMARY_PROMPT_TEMPLATE = """\
Sos un asistente académico. Leé el siguiente documento y generá un resumen completo y útil.

Instrucciones:
- Escribí en el mismo idioma que el documento (español o inglés).
- El resumen debe tener entre 200 y 400 palabras.
- Organizá el resumen en párrafos claros: primero una introducción general, luego los temas principales, y finalmente una conclusión breve.
- No uses asteriscos, guiones ni ningún tipo de formato especial. Solo texto plano con saltos de línea entre párrafos.
- No inventes información que no esté en el documento.
- Si el documento es un cronograma o índice, mencioná los temas principales que cubre y su estructura general.

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
            max_tokens=1200,
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


_PRE_EXAM_SYNTHESIS_PROMPT_TEMPLATE = """\
Sos un asistente académico que ayuda a un alumno a repasar antes de un examen.
Te paso los resúmenes individuales de los documentos indexados de su curso (o
de la unidad que eligió repasar). Combinalos en un único resumen de repaso.

Instrucciones:
- Escribí en español.
- Organizá el resumen por tema, no por documento — si dos documentos cubren
  lo mismo, combiná esa información en un solo bloque en vez de repetirla.
- Cuando menciones un tema puntual, indicá entre paréntesis de qué archivo
  sale, usando el nombre exacto que te paso.
- No uses asteriscos, guiones ni formato especial. Solo texto plano con
  saltos de línea entre párrafos.
- No inventes información que no esté en los resúmenes que te paso.

RESÚMENES POR DOCUMENTO:
{summaries_block}
"""


async def summarize_pre_exam(
    course_id: int,
    db: AsyncSession,
    llm: LLMProvider,
    section: Optional[int] = None,
) -> dict:
    """
    Genera un resumen de repaso combinando todo el material indexado y
    relevante para un próximo examen (BUS-04).

    Args:
        course_id: ID del curso.
        db: sesión async de SQLAlchemy.
        llm: instancia de LLMProvider.
        section: unidad/sección opcional para acotar el material (BUS-05).
                 Si es None, se usa todo el material indexado del curso.

    Returns:
        dict con: summary, documents_used (lista de {document_id, filename}),
        total_documents. Si no hay documentos indexados, summary es "" y
        total_documents es 0 — no se llama al LLM sin señal real.
    """
    stmt = select(Document.id, Document.filename).where(
        Document.course_id == course_id,
        Document.status == "indexed",
    )
    if section is not None:
        stmt = stmt.where(Document.section == section)

    docs_result = await db.execute(stmt.order_by(Document.filename))
    docs = docs_result.all()

    if not docs:
        return {"summary": "", "documents_used": [], "total_documents": 0}

    # 1) Resumen individual por documento — reusa summarize_document tal
    # cual. Si un documento puntual falla (LLM caído, sin chunks, etc.), se
    # lo salta en vez de tirar abajo todo el resumen de repaso.
    per_doc_summaries: list[dict] = []
    for doc_id, filename in docs:
        try:
            result = await summarize_document(document_id=doc_id, course_id=course_id, db=db, llm=llm)
            per_doc_summaries.append(result)
        except (LookupError, RuntimeError) as exc:
            logger.warning("Pre-exam summary: skipping document %s (%s): %s", doc_id, filename, exc)
            continue

    if not per_doc_summaries:
        return {"summary": "", "documents_used": [], "total_documents": 0}

    # 2) Un único LLM call de síntesis — combina los resúmenes ya generados
    # (cada uno acotado a 200-400 palabras) en un solo resumen de repaso.
    summaries_block = "\n\n".join(
        f'Archivo: "{r["document_filename"]}"\n{r["summary"]}'
        for r in per_doc_summaries
    )
    prompt = _PRE_EXAM_SYNTHESIS_PROMPT_TEMPLATE.format(summaries_block=summaries_block)

    try:
        synthesis = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        summary_text = synthesis.text.strip()
    except Exception as exc:
        raise RuntimeError("LLM pre-exam synthesis failed") from exc

    return {
        "summary": summary_text,
        "documents_used": [
            {"document_id": r["document_id"], "filename": r["document_filename"]}
            for r in per_doc_summaries
        ],
        "total_documents": len(per_doc_summaries),
    }
