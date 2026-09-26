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
concurrencia). Antes eran N llamadas en serie, que es lo que hacía que el
resumen pre-parcial tardara minutos y se pasara del timeout del cliente PHP.

COST-02: los resúmenes (el de cada documento y el de repaso pre-examen) se
guardan en la base hasta que cambie el archivo, el modelo o el prompt, en vez
de vencer a las 24 h en Redis. Todos los alumnos leen el mismo sin volver a
pagarlo. Ver summary_store.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.documents import summary_store
from app.providers.llm import LLMProvider
from app.shared.config import get_settings
from app.shared.language import detect_language, language_directive
from app.shared.usage_ledger import UsageRecord, record_usage

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


# Bump it whenever the summary prompt changes in a way that should invalidate
# the stored summaries.
_PROMPT_VERSION = "p2"
# Same for the pre-exam synthesis prompt (_PRE_EXAM_SYNTHESIS_PROMPT_TEMPLATE).
_SYNTHESIS_PROMPT_VERSION = "s1"


def _summary_key(document: Document, model: str) -> str:
    """Clave del resumen guardado de un documento (COST-02).

    Incluye una huella del archivo (`file_hash`, o `updated_at` si el hash no
    está) para que reemplazar un documento manteniendo su id (CONT-07 / #356)
    genere una clave distinta: el resumen viejo deja de servirse solo.

    Incluye también el modelo: si el equipo cambia `LLM_MODEL`, los resúmenes
    se regeneran con el modelo nuevo en vez de servir los del anterior. Y la
    versión del prompt (`_PROMPT_VERSION`): al cambiar cómo se pide el
    resumen, los viejos dejan de servirse.
    """
    fingerprint = document.file_hash or document.updated_at.isoformat()
    return summary_store.make_key(
        "document", str(document.id), model, _PROMPT_VERSION, fingerprint
    )


def _pre_exam_key(course_id: int, document_keys: list[str], model: str) -> str:
    """Clave del resumen de repaso: depende de la lista de documentos con sus
    huellas (ya están dentro de cada clave de documento), del modelo y de las
    dos versiones de prompt. Si cambia el material o se agrega un documento,
    es otra clave."""
    return summary_store.make_key(
        "pre_exam",
        str(course_id),
        model,
        _PROMPT_VERSION,
        _SYNTHESIS_PROMPT_VERSION,
        *sorted(document_keys),
    )


@dataclass
class _Generated:
    """Un resumen recién generado y lo que costó."""

    payload: dict
    model: Optional[str]
    provider: Optional[str]
    prompt_tokens: int
    completion_tokens: int


async def _record_hit(llm: LLMProvider, stored: summary_store.StoredSummary) -> None:
    """Deja en el registro de consumo que este pedido salió de lo guardado:
    0 tokens gastados y los que se ahorró."""
    await record_usage(
        UsageRecord(
            kind="llm",
            provider=stored.provider or llm.provider_name,
            model=stored.model or llm.model,
            cache_hit=True,
            saved_tokens=stored.tokens,
        )
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
    lang = detect_language(concatenated)
    if lang:
        prompt = f"{prompt}\n\n{language_directive(lang)}"
    return document, prompt, chunks_used, total_chunks


async def _generate_summary(
    document: Document,
    prompt: str,
    chunks_used: int,
    total_chunks: int,
    llm: LLMProvider,
) -> _Generated:
    """Genera el resumen de un documento ya leído de la DB. No toca la DB — se
    puede correr en paralelo con otros.

    Raises:
        RuntimeError: si el LLM falla al generar el resumen.
    """
    try:
        result = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        summary_text = result.text.strip()
    except Exception as exc:
        raise RuntimeError("LLM summary generation failed") from exc

    return _Generated(
        payload={
            "document_id": str(document.id),
            "document_filename": document.filename,
            "summary": summary_text,
            "chunks_used": chunks_used,
            "total_chunks": total_chunks,
        },
        model=result.model or None,
        provider=result.provider or None,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


async def _save_document_summary(
    db: AsyncSession, document: Document, key: str, generated: _Generated
) -> None:
    await summary_store.save(
        db,
        cache_key=key,
        kind=summary_store.KIND_DOCUMENT,
        course_id=document.course_id,
        document_id=document.id,
        prompt_version=_PROMPT_VERSION,
        payload=generated.payload,
        model=generated.model,
        provider=generated.provider,
        prompt_tokens=generated.prompt_tokens,
        completion_tokens=generated.completion_tokens,
    )


async def summarize_document(
    document_id: UUID,
    course_id: int,
    db: AsyncSession,
    llm: LLMProvider,
) -> dict:
    """
    Devuelve el resumen del documento: el guardado si el archivo, el modelo y
    el prompt no cambiaron (0 tokens), o uno nuevo generado con el LLM, que
    queda guardado para los que lo pidan después (COST-02).

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
    document, prompt, chunks_used, total_chunks = await _load_document_for_summary(
        document_id, course_id, db
    )
    key = _summary_key(document, llm.model)

    stored = await summary_store.lookup(db, key)
    if stored is not None:
        logger.info("Resumen servido desde lo guardado para documento %s", document.id)
        await _record_hit(llm, stored)
        return stored.payload

    generated = await _generate_summary(
        document, prompt, chunks_used, total_chunks, llm
    )
    await _save_document_summary(db, document, key, generated)
    return generated.payload


_PRE_EXAM_SYNTHESIS_PROMPT_TEMPLATE = """\
Sos un asistente académico que ayuda a un alumno a repasar antes de un examen.
Te paso los resúmenes individuales de los documentos indexados de su curso (o
de la unidad que eligió repasar). Combinalos en un único resumen de repaso.

Instrucciones:
- Escribí en el mismo idioma que los resúmenes individuales (si mezclan
  idiomas, usá el predominante).
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

    Si el material, el modelo y los prompts no cambiaron desde la última vez,
    devuelve el resumen guardado sin llamar al LLM (COST-02). Si no, reusa los
    resúmenes guardados de cada documento, genera solo los que faltan y
    guarda todo.

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

    # Fase A — lecturas de DB, EN SERIE (PERF-02). `AsyncSession` no es seguro
    # de usar concurrentemente: paralelizar acá corrompe la sesión. Es barato
    # igual, son queries indexadas.
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

    keys = [_summary_key(item[0], llm.model) for item in loaded]

    # El resumen de repaso completo, si ya existe para este mismo material.
    pre_exam_key = _pre_exam_key(course_id, keys, llm.model)
    stored_pre_exam = await summary_store.lookup(db, pre_exam_key)
    if stored_pre_exam is not None:
        logger.info(
            "Resumen pre-examen servido desde lo guardado (curso %s)", course_id
        )
        await _record_hit(llm, stored_pre_exam)
        return stored_pre_exam.payload

    # Los resúmenes por documento que ya están guardados no se piden de nuevo.
    resolved: dict[int, dict] = {}
    doc_tokens = 0
    missing: list[int] = []
    for index, key in enumerate(keys):
        stored = await summary_store.lookup(db, key)
        if stored is None:
            missing.append(index)
            continue
        await _record_hit(llm, stored)
        resolved[index] = stored.payload
        doc_tokens += stored.tokens

    # Fase B — llamadas al LLM de los que faltan, EN PARALELO con un tope de
    # concurrencia. El tope existe por la cuota gratuita de Gemini: disparar 8
    # requests juntas se come el rate limit y devuelve 429/503 en vez de ir
    # más rápido.
    settings = get_settings()
    semaphore = asyncio.Semaphore(max(1, settings.summary_max_concurrency))

    async def _one(index: int) -> tuple[int, Optional[_Generated]]:
        document, prompt, chunks_used, total_chunks = loaded[index]
        async with semaphore:
            try:
                return index, await _generate_summary(
                    document, prompt, chunks_used, total_chunks, llm
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
                return index, None

    all_generated = True
    for index, generated in await asyncio.gather(*(_one(i) for i in missing)):
        if generated is None:
            all_generated = False
            continue
        resolved[index] = generated.payload
        doc_tokens += generated.prompt_tokens + generated.completion_tokens
        # Fase C — guardado, EN SERIE (misma sesión).
        await _save_document_summary(db, loaded[index][0], keys[index], generated)

    per_doc_summaries = [resolved[i] for i in sorted(resolved)]

    if not per_doc_summaries:
        return {"summary": "", "documents_used": [], "total_documents": 0}

    # Un único LLM call de síntesis — combina los resúmenes ya generados
    # (cada uno acotado a 200-400 palabras) en un solo resumen de repaso.
    summaries_block = "\n\n".join(
        f'Archivo: "{r["document_filename"]}"\n{r["summary"]}'
        for r in per_doc_summaries
    )
    prompt = _PRE_EXAM_SYNTHESIS_PROMPT_TEMPLATE.format(summaries_block=summaries_block)
    lang = detect_language(*(r["summary"] for r in per_doc_summaries))
    if lang:
        prompt = f"{prompt}\n\n{language_directive(lang)}"

    try:
        synthesis = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        summary_text = synthesis.text.strip()
    except Exception as exc:
        raise RuntimeError("LLM pre-exam synthesis failed") from exc

    result = {
        "summary": summary_text,
        "documents_used": [
            {"document_id": r["document_id"], "filename": r["document_filename"]}
            for r in per_doc_summaries
        ],
        "total_documents": len(per_doc_summaries),
    }

    # Solo se guarda si entraron todos los documentos: un repaso armado con
    # menos porque uno falló no tiene que quedar como "el resumen del curso".
    if all_generated:
        await summary_store.save(
            db,
            cache_key=pre_exam_key,
            kind=summary_store.KIND_PRE_EXAM,
            course_id=course_id,
            document_id=None,
            prompt_version=f"{_PROMPT_VERSION}+{_SYNTHESIS_PROMPT_VERSION}",
            payload=result,
            model=synthesis.model or None,
            provider=synthesis.provider or None,
            # Lo que se ahorra en un acierto es la síntesis más los resúmenes
            # por documento que hubo que generar o que ya estaban guardados.
            prompt_tokens=synthesis.prompt_tokens + doc_tokens,
            completion_tokens=synthesis.completion_tokens,
        )

    return result
