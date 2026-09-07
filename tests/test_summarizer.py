"""
Tests del summarizer de documentos (BUS-03 / BUS-04) con foco en PERF-02:
cache en Redis y paralelización del resumen pre-parcial.

La DB y el LLM van mockeados — lo que se verifica acá es la coreografía
(cuántas veces se llama al LLM, qué se cachea, qué pasa si Redis se cae), no
la calidad del resumen.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.documents import summarizer
from app.documents.summarizer import summarize_document, summarize_pre_exam


def _fake_document(filename: str = "apunte.pdf", file_hash: str = "hash-abc") -> MagicMock:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.course_id = 7
    doc.filename = filename
    doc.file_hash = file_hash
    doc.updated_at = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    return doc


def _fake_chunk(content: str, index: int) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.chunk_index = index
    return chunk


def _fake_db(document: MagicMock, chunks: list) -> MagicMock:
    """Sesión mockeada: la 1ra query devuelve el documento, la 2da sus chunks."""
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document

    chunks_result = MagicMock()
    chunks_result.all.return_value = chunks

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[doc_result, chunks_result])
    return db


def _fake_llm(text: str = "resumen generado", model: str = "gemini-3.5-flash") -> MagicMock:
    llm = MagicMock()
    llm.model = model
    result = MagicMock()
    result.text = text
    llm.chat_completion = AsyncMock(return_value=result)
    return llm


def _fake_cache() -> MagicMock:
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock(return_value=True)
    return cache


# ============================================================
# Cache de resúmenes por documento (PERF-02)
# ============================================================

@pytest.mark.asyncio
async def test_summarize_document_caches_result_after_generating():
    """Un miss de cache genera el resumen y lo guarda con TTL."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido del apunte", 0)])
    llm = _fake_llm()
    cache = _fake_cache()

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm, cache=cache
    )

    assert result["summary"] == "resumen generado"
    llm.chat_completion.assert_awaited_once()
    cache.setex.assert_awaited_once()

    key, ttl, raw = cache.setex.await_args.args
    assert str(document.id) in key
    assert "hash-abc" in key      # huella del archivo
    assert "gemini-3.5-flash" in key  # modelo, para no servir resúmenes de otro
    assert ttl > 0
    assert json.loads(raw)["summary"] == "resumen generado"


@pytest.mark.asyncio
async def test_summarize_document_serves_from_cache_without_calling_llm():
    """Un hit de cache devuelve el resumen guardado y NO llama al LLM."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido", 0)])
    llm = _fake_llm()

    cached = {
        "document_id": str(document.id),
        "document_filename": "apunte.pdf",
        "summary": "resumen cacheado",
        "chunks_used": 1,
        "total_chunks": 1,
    }
    cache = _fake_cache()
    cache.get = AsyncMock(return_value=json.dumps(cached))

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm, cache=cache
    )

    assert result["summary"] == "resumen cacheado"
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_document_survives_redis_being_down():
    """Si Redis está caído, el resumen se genera igual — la cache es una
    optimización, no una dependencia dura."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido", 0)])
    llm = _fake_llm()

    cache = _fake_cache()
    cache.get = AsyncMock(side_effect=ConnectionError("redis caído"))
    cache.setex = AsyncMock(side_effect=ConnectionError("redis caído"))

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm, cache=cache
    )

    assert result["summary"] == "resumen generado"
    llm.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_document_regenerates_on_corrupt_cache_entry():
    """Una entrada de cache que no es JSON válido se trata como miss."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido", 0)])
    llm = _fake_llm()

    cache = _fake_cache()
    cache.get = AsyncMock(return_value="{no es json")

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm, cache=cache
    )

    assert result["summary"] == "resumen generado"
    llm.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_document_without_cache_always_calls_llm():
    """Sin cache pasada, el comportamiento es el previo a PERF-02."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido", 0)])
    llm = _fake_llm()

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm
    )

    assert result["summary"] == "resumen generado"
    llm.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_key_changes_when_document_is_replaced():
    """Reemplazar un documento manteniendo su id (CONT-07) cambia la key, así
    que no se sirve el resumen del archivo viejo."""
    document = _fake_document(file_hash="hash-original")
    key_before = summarizer._cache_key(document, "gemini-3.5-flash")

    document.file_hash = "hash-nuevo"
    key_after = summarizer._cache_key(document, "gemini-3.5-flash")

    assert key_before != key_after


# ============================================================
# Resumen pre-parcial en paralelo (PERF-02)
# ============================================================

def _pre_exam_db(documents: list[MagicMock], chunks_by_doc: dict) -> MagicMock:
    """Sesión mockeada para summarize_pre_exam: primero la lista de documentos
    del curso, después (documento, chunks) por cada uno."""
    docs_result = MagicMock()
    docs_result.all.return_value = [(d.id, d.filename) for d in documents]

    side_effects = [docs_result]
    for doc in documents:
        doc_result = MagicMock()
        doc_result.scalar_one_or_none.return_value = doc
        chunks_result = MagicMock()
        chunks_result.all.return_value = chunks_by_doc[doc.id]
        side_effects.extend([doc_result, chunks_result])

    db = MagicMock()
    db.execute = AsyncMock(side_effect=side_effects)
    return db


@pytest.mark.asyncio
async def test_pre_exam_summary_runs_document_summaries_concurrently():
    """Los resúmenes por documento se piden en paralelo, no en serie.

    Es el arreglo central de PERF-02: con N documentos, el endpoint tardaba
    N veces la latencia del LLM y se pasaba del timeout de 120s del cliente
    PHP. Se verifica midiendo cuántas llamadas están en vuelo a la vez.
    """
    documents = [_fake_document(f"doc{i}.pdf", f"hash-{i}") for i in range(4)]
    chunks_by_doc = {d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)}
    db = _pre_exam_db(documents, chunks_by_doc)

    in_flight = 0
    max_in_flight = 0

    async def slow_completion(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        result = MagicMock()
        result.text = "resumen"
        return result

    llm = _fake_llm()
    llm.chat_completion = AsyncMock(side_effect=slow_completion)

    result = await summarize_pre_exam(course_id=7, db=db, llm=llm)

    assert result["total_documents"] == 4
    # 4 resúmenes + 1 síntesis final.
    assert llm.chat_completion.await_count == 5
    # Si corriera en serie, max_in_flight sería 1.
    assert max_in_flight > 1


@pytest.mark.asyncio
async def test_pre_exam_summary_respects_concurrency_limit(monkeypatch):
    """El paralelismo tiene tope: contra la cuota gratuita de Gemini, disparar
    todo junto devuelve 429/503 en vez de ir más rápido."""
    from app.shared.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "summary_max_concurrency", 2, raising=False)

    documents = [_fake_document(f"doc{i}.pdf", f"hash-{i}") for i in range(6)]
    chunks_by_doc = {d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)}
    db = _pre_exam_db(documents, chunks_by_doc)

    in_flight = 0
    max_in_flight = 0

    async def slow_completion(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        result = MagicMock()
        result.text = "resumen"
        return result

    llm = _fake_llm()
    llm.chat_completion = AsyncMock(side_effect=slow_completion)

    await summarize_pre_exam(course_id=7, db=db, llm=llm)

    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_pre_exam_summary_skips_document_whose_llm_call_fails():
    """Un documento que falla se saltea, el repaso se arma con el resto —
    mismo criterio que antes de PERF-02."""
    documents = [_fake_document(f"doc{i}.pdf", f"hash-{i}") for i in range(3)]
    chunks_by_doc = {d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)}
    db = _pre_exam_db(documents, chunks_by_doc)

    calls = {"n": 0}

    async def flaky_completion(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("LLM caído para este documento")
        result = MagicMock()
        result.text = "resumen"
        return result

    llm = _fake_llm()
    llm.chat_completion = AsyncMock(side_effect=flaky_completion)

    result = await summarize_pre_exam(course_id=7, db=db, llm=llm)

    assert result["total_documents"] == 2
    assert len(result["documents_used"]) == 2


@pytest.mark.asyncio
async def test_pre_exam_summary_returns_empty_without_indexed_documents():
    """Sin material indexado no se llama al LLM."""
    docs_result = MagicMock()
    docs_result.all.return_value = []
    db = MagicMock()
    db.execute = AsyncMock(return_value=docs_result)
    llm = _fake_llm()

    result = await summarize_pre_exam(course_id=7, db=db, llm=llm)

    assert result == {"summary": "", "documents_used": [], "total_documents": 0}
    llm.chat_completion.assert_not_awaited()
