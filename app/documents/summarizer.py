"""
Lógica de resumen automático de documentos indexados (BUS-03) y de resumen
pre-parcial multi-documento (BUS-04).

Dado un document_id, recupera todos los chunks del documento (ordenados por
chunk_index), los concatena hasta un máximo de MAX_CHARS caracteres y pide
al LLM un resumen estructurado en el idioma del propio documento.

No hace embedding ni búsqueda semántica: lee los chunks en orden secuencial
para preservar la estructura original del documento.

BUS-04 resume documento por documento y hace un único LLM call final de
síntesis para combinarlos en un solo resumen de repaso, mismo criterio de "una
sola llamada de síntesis" ya usado en Study Plan y en el FAQ dashboard.

PERF-02: esos resúmenes por documento se piden al LLM en paralelo (con tope de
concurrencia) y se cachean en Redis por versión del archivo. Antes eran N
llamadas en serie sin cache, que es lo que hacía que el resumen pre-parcial
tardara minutos y se pasara del timeout del cliente PHP.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.llm import LLMProvider
from app.shared.config import get_settings

logger = logging.getLogger("nexusai.documents.summarizer")

MAX_CHARS = 20_000

# Prefijo + versión de las cache keys de Redis (PERF-02). Bumpear la versión
# invalida todos los resúmenes cacheados de una — hacelo si cambia el prompt
# de resumen o la forma del dict que se guarda.
_CACHE_PREFIX = "nexusai:summary:v1"
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


def _cache_key(document: Document, model: str) -> str:
    """Key de Redis para el resumen de un documento (PERF-02).

    Incluye una huella del archivo (`file_hash`, o `updated_at` si el hash no
    está) para que reemplazar un documento manteniendo su id (CONT-07 / #356)
    genere una key distinta — o sea, la entrada vieja queda huérfana y expira
    sola por TTL, sin necesidad de invalidar nada a mano.

    Incluye también el modelo: si el equipo cambia `LLM_MODEL`, los resúmenes
    se regeneran con el modelo nuevo en vez de servir los del anterior.
    """
    fingerprint = document.file_hash or document.updated_at.isoformat()
    return f"{_CACHE_PREFIX}:{document.id}:{model}:{fingerprint}"


async def _cache_get(cache: Any, key: str) -> Optional[dict]:
    """Lee un resumen cacheado. Nunca propaga: si Redis está caído o devolvió
    basura, se comporta como un miss y el resumen se regenera."""
    try:
        raw = await cache.get(key)
    except Exception as exc:
        logger.warning(
            "Cache de resúmenes no disponible en lectura: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None

    if not raw:
        return None

    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Entrada de cache corrupta en %s — se regenera el resumen", key)
        return None

    return value if isinstance(value, dict) else None


async def _cache_set(cache: Any, key: str, value: dict, ttl_sec: int) -> None:
    """Guarda un resumen en cache. Nunca propaga — no poder cachear no es
    motivo para fallar un resumen que ya se generó bien."""
    if ttl_sec <= 0:
        return
    try:
        await cache.setex(key, ttl_sec, json.dumps(value, ensure_ascii=False))
    except Exception as exc:
        logger.warning(
            "Cache de resúmenes no disponible en escritura: %s: %s",
            type(exc).__name__,
            exc,
        )


async def _load_document_for_summary(
    document_id: UUID,
    course_id: int,
    db: AsyncSession,
) -> tuple[Document, str, int, int]:
    """Trae de la DB todo lo que hace falta para resumir un documento.

    Separado del LLM a propósito (PERF-02): `AsyncSession` de SQLAlchemy NO es
    seguro de usar concurrentemente, así que el resumen pre-parcial hace TODAS
    las lecturas de DB en serie con esta función y recién después paraleliza
    las llamadas al LLM, que son las que realmente tardan.

    Returns:
        (document, prompt, chunks_used, total_chunks)

    Raises:
        LookupError: si el documento no existe, no pertenece al course_id, o
                     no tiene chunks indexados.
    """
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()

    if document is None:
        raise LookupError(f"Document {document_id} not found")
    if document.course_id != course_id:
        raise LookupError(
            f"Document {document_id} does not belong to course {course_id}"
        )

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
    return document, prompt, chunks_used, total_chunks


async def _summarize_loaded_document(
    document: Document,
    prompt: str,
    chunks_used: int,
    total_chunks: int,
    llm: LLMProvider,
    cache: Any = None,
) -> dict:
    """Resuelve el resumen de un documento ya leído de la DB: cache primero,
    LLM si hay miss. No toca la DB — se puede correr en paralelo con otros.

    Raises:
        RuntimeError: si el LLM falla al generar el resumen.
    """
    settings = get_settings()
    key = _cache_key(document, llm.model) if cache is not None else None

    if key is not None:
        cached = await _cache_get(cache, key)
        if cached is not None:
            logger.info("Resumen servido desde cache para documento %s", document.id)
            return cached

    try:
        result = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        summary_text = result.text.strip()
    except Exception as exc:
        raise RuntimeError("LLM summary generation failed") from exc

    payload = {
        "document_id": str(document.id),
        "document_filename": document.filename,
        "summary": summary_text,
        "chunks_used": chunks_used,
        "total_chunks": total_chunks,
    }

    if key is not None:
        await _cache_set(cache, key, payload, settings.summary_cache_ttl_sec)

    return payload


async def summarize_document(
    document_id: UUID,
    course_id: int,
    db: AsyncSession,
    llm: LLMProvider,
    cache: Any = None,
) -> dict:
    """
    Genera un resumen del documento usando el LLM.

    Args:
        document_id: UUID del documento a resumir.
        course_id: ID del curso — se usa para validar que el documento pertenece
                   al curso del usuario (aislamiento multi-curso).
        db: sesión async de SQLAlchemy.
        llm: instancia de LLMProvider.
        cache: cliente Redis opcional (PERF-02). Si se pasa, el resumen se
               sirve de cache cuando ya existe uno para esa versión del
               archivo. Si es None, se llama al LLM siempre — comportamiento
               idéntico al previo a PERF-02.

    Returns:
        dict con: document_id, document_filename, summary, chunks_used, total_chunks.

    Raises:
        LookupError: si el documento no existe o no pertenece al course_id.
        RuntimeError: si el LLM falla al generar el resumen.
    """
    document, prompt, chunks_used, total_chunks = await _load_document_for_summary(
        document_id, course_id, db
    )
    return await _summarize_loaded_document(
        document, prompt, chunks_used, total_chunks, llm, cache
    )


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
    cache: Any = None,
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
        cache: cliente Redis opcional — se propaga a cada resumen individual.

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

    # 1) Resumen individual por documento, en dos fases (PERF-02).
    #
    # Antes esto era un for secuencial de N llamadas al LLM: con ~8 documentos
    # indexados el endpoint tardaba más que el CURLOPT_TIMEOUT de 120s del
    # cliente PHP del plugin, o sea que además de lento se moría por timeout
    # antes de responder.
    #
    # Fase A — lecturas de DB, EN SERIE. `AsyncSession` no es seguro de usar
    # concurrentemente: paralelizar acá corrompe la sesión. Es barato igual,
    # son queries indexadas.
    loaded: list[tuple[Document, str, int, int]] = []
    for doc_id, filename in docs:
        try:
            loaded.append(await _load_document_for_summary(doc_id, course_id, db))
        except LookupError as exc:
            logger.warning(
                "Pre-exam summary: skipping document %s (%s): %s", doc_id, filename, exc
            )
            continue

    if not loaded:
        return {"summary": "", "documents_used": [], "total_documents": 0}

    # Fase B — llamadas al LLM, EN PARALELO con un tope de concurrencia. El
    # tope existe por la cuota gratuita de Gemini: disparar 8 requests juntas
    # se come el rate limit y devuelve 429/503 en vez de ir más rápido.
    settings = get_settings()
    semaphore = asyncio.Semaphore(max(1, settings.summary_max_concurrency))

    async def _one(loaded_doc: tuple[Document, str, int, int]) -> Optional[dict]:
        document, prompt, chunks_used, total_chunks = loaded_doc
        async with semaphore:
            try:
                return await _summarize_loaded_document(
                    document, prompt, chunks_used, total_chunks, llm, cache
                )
            except RuntimeError as exc:
                # Un documento que falla no tira abajo el repaso entero —
                # mismo criterio que antes de PERF-02.
                logger.warning(
                    "Pre-exam summary: skipping document %s (%s): %s",
                    document.id,
                    document.filename,
                    exc,
                )
                return None

    results = await asyncio.gather(*(_one(item) for item in loaded))
    per_doc_summaries = [r for r in results if r is not None]

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
