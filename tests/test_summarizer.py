"""
Tests del summarizer de documentos (BUS-03 / BUS-04): resúmenes permanentes
(COST-02) y paralelización del resumen pre-parcial (PERF-02).

La DB y el LLM van mockeados y el almacén de resúmenes es uno en memoria — lo
que se verifica acá es la coreografía (cuántas veces se llama al LLM, qué se
guarda, cuándo se invalida), no la calidad del resumen.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.documents import summarizer, summary_store
from app.documents.summarizer import summarize_document, summarize_pre_exam

# Las funciones reales del almacén, antes de que el fixture las reemplace.
_real_lookup = summary_store.lookup
_real_save = summary_store.save
_real_make_key = summary_store.make_key


def _fake_document(
    filename: str = "apunte.pdf", file_hash: str = "hash-abc"
) -> MagicMock:
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


def _fake_llm(
    text: str = "resumen generado", model: str = "gemini-3.5-flash"
) -> MagicMock:
    llm = MagicMock()
    llm.model = model
    result = MagicMock()
    result.text = text
    llm.chat_completion = AsyncMock(return_value=result)
    return llm


class _FakeStore:
    """Reemplaza a summary_store: guarda en un dict y cuenta los accesos, sin base."""

    def __init__(self) -> None:
        self.rows: dict[str, summary_store.StoredSummary] = {}
        self.saved: list[dict] = []
        self.lookups: list[str] = []

    async def lookup(self, db, cache_key):
        self.lookups.append(cache_key)
        return self.rows.get(cache_key)

    async def save(self, db, **kwargs):
        self.saved.append(kwargs)
        self.rows[kwargs["cache_key"]] = summary_store.StoredSummary(
            payload=kwargs["payload"],
            model=kwargs["model"],
            provider=kwargs["provider"],
            prompt_tokens=kwargs["prompt_tokens"],
            completion_tokens=kwargs["completion_tokens"],
        )


@pytest.fixture(autouse=True)
def store(monkeypatch) -> _FakeStore:
    """Todos los tests del módulo corren con un almacén en memoria (sin base)
    y capturan el registro de aciertos de caché."""
    fake = _FakeStore()
    monkeypatch.setattr(summary_store, "lookup", fake.lookup)
    monkeypatch.setattr(summary_store, "save", fake.save)
    hits: list = []

    async def fake_record(record):
        hits.append(record)

    monkeypatch.setattr(summarizer, "record_usage", fake_record)
    fake.usage_records = hits
    return fake


# ============================================================
# Resúmenes permanentes por documento (COST-02)
# ============================================================


@pytest.mark.asyncio
async def test_summarize_document_stores_result_after_generating(store):
    """Un miss genera el resumen y lo guarda, con lo que costó generarlo."""
    document = _fake_document()
    db = _fake_db(document, [_fake_chunk("contenido del apunte", 0)])
    llm = _fake_llm()
    llm.chat_completion.return_value.prompt_tokens = 300
    llm.chat_completion.return_value.completion_tokens = 120
    llm.chat_completion.return_value.model = "gemini-3.5-flash"
    llm.chat_completion.return_value.provider = "google"

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm
    )

    assert result["summary"] == "resumen generado"
    llm.chat_completion.assert_awaited_once()
    [saved] = store.saved
    assert saved["kind"] == "document"
    assert saved["document_id"] == document.id
    assert saved["course_id"] == 7
    assert saved["payload"]["summary"] == "resumen generado"
    assert (saved["prompt_tokens"], saved["completion_tokens"]) == (300, 120)
    assert (saved["model"], saved["provider"]) == ("gemini-3.5-flash", "google")
    assert store.usage_records == []  # un miss no es un acierto de caché


@pytest.mark.asyncio
async def test_summarize_document_serves_stored_summary_without_calling_llm(store):
    """Un acierto devuelve lo guardado, NO llama al LLM y deja registrado que
    salió de la caché con los tokens que se ahorró."""
    document = _fake_document()
    llm = _fake_llm()
    stored_payload = {
        "document_id": str(document.id),
        "document_filename": "apunte.pdf",
        "summary": "resumen guardado",
        "chunks_used": 1,
        "total_chunks": 1,
    }
    store.rows[summarizer._summary_key(document, llm.model)] = (
        summary_store.StoredSummary(
            payload=stored_payload,
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=300,
            completion_tokens=120,
        )
    )
    db = _fake_db(document, [_fake_chunk("contenido", 0)])

    result = await summarize_document(
        document_id=document.id, course_id=7, db=db, llm=llm
    )

    assert result["summary"] == "resumen guardado"
    llm.chat_completion.assert_not_awaited()
    assert store.saved == []
    [hit] = store.usage_records
    assert hit.cache_hit is True
    assert hit.saved_tokens == 420
    assert (hit.provider, hit.model) == ("openai", "gpt-4o-mini")
    assert hit.prompt_tokens == 0 and hit.completion_tokens == 0


@pytest.mark.asyncio
async def test_second_request_for_same_document_costs_nothing(store):
    """Dos alumnos piden el mismo resumen: el LLM se llama una sola vez."""
    document = _fake_document()
    llm = _fake_llm()

    first = await summarize_document(
        document_id=document.id,
        course_id=7,
        db=_fake_db(document, [_fake_chunk("contenido", 0)]),
        llm=llm,
    )
    second = await summarize_document(
        document_id=document.id,
        course_id=7,
        db=_fake_db(document, [_fake_chunk("contenido", 0)]),
        llm=llm,
    )

    assert first["summary"] == second["summary"]
    llm.chat_completion.assert_awaited_once()
    assert len(store.usage_records) == 1


@pytest.mark.asyncio
async def test_summary_key_changes_when_document_is_replaced():
    """Reemplazar un documento manteniendo su id (CONT-07) cambia la clave, así
    que no se sirve el resumen del archivo viejo."""
    document = _fake_document(file_hash="hash-original")
    key_before = summarizer._summary_key(document, "gemini-3.5-flash")

    document.file_hash = "hash-nuevo"
    key_after = summarizer._summary_key(document, "gemini-3.5-flash")

    assert key_before != key_after


@pytest.mark.asyncio
async def test_summary_key_changes_with_model_and_prompt_version(monkeypatch):
    document = _fake_document()
    base = summarizer._summary_key(document, "gemini-3.5-flash")

    assert summarizer._summary_key(document, "gpt-4o-mini") != base
    monkeypatch.setattr(summarizer, "_PROMPT_VERSION", "p99")
    assert summarizer._summary_key(document, "gemini-3.5-flash") != base


@pytest.mark.asyncio
async def test_summary_key_is_stable_for_the_same_document():
    document = _fake_document()
    assert summarizer._summary_key(document, "m") == summarizer._summary_key(
        document, "m"
    )


# ============================================================
# Almacén (summary_store) — tolerancia a fallas de la base
# ============================================================


@pytest.mark.asyncio
async def test_store_lookup_returns_none_when_the_database_fails():
    db = MagicMock()
    db.begin_nested = MagicMock(side_effect=RuntimeError("base caída"))

    assert await _real_lookup(db, "clave") is None


@pytest.mark.asyncio
async def test_store_save_never_raises_when_the_database_fails():
    db = MagicMock()
    db.begin_nested = MagicMock(side_effect=RuntimeError("base caída"))

    await _real_save(
        db,
        cache_key="k",
        kind="document",
        course_id=1,
        document_id=None,
        prompt_version="p2",
        payload={},
        model=None,
        provider=None,
        prompt_tokens=0,
        completion_tokens=0,
    )


def test_store_make_key_is_stable_and_depends_on_every_part():
    assert _real_make_key("a", "b") == _real_make_key("a", "b")
    assert _real_make_key("a", "b") != _real_make_key("a", "c")
    assert _real_make_key("ab", "c") != _real_make_key("a", "bc")


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
    chunks_by_doc = {
        d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)
    }
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
    chunks_by_doc = {
        d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)
    }
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
    chunks_by_doc = {
        d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)
    }
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


# ============================================================
# Resumen pre-examen permanente (COST-02)
# ============================================================


def _pre_exam_setup(n: int = 3):
    documents = [_fake_document(f"doc{i}.pdf", f"hash-{i}") for i in range(n)]
    chunks_by_doc = {
        d.id: [_fake_chunk(f"contenido {i}", 0)] for i, d in enumerate(documents)
    }
    return documents, chunks_by_doc


@pytest.mark.asyncio
async def test_pre_exam_summary_is_stored_and_served_without_any_llm_call(store):
    """El primer pedido genera todo y lo guarda; el segundo, con el mismo
    material, no llama al LLM ni una vez (ni por documento ni la síntesis)."""
    documents, chunks_by_doc = _pre_exam_setup(3)
    llm = _fake_llm()

    first = await summarize_pre_exam(
        course_id=7, db=_pre_exam_db(documents, chunks_by_doc), llm=llm
    )
    calls_after_first = llm.chat_completion.await_count
    second = await summarize_pre_exam(
        course_id=7, db=_pre_exam_db(documents, chunks_by_doc), llm=llm
    )

    assert calls_after_first == 4  # 3 resúmenes + 1 síntesis
    assert llm.chat_completion.await_count == calls_after_first
    assert second == first
    kinds = sorted(saved["kind"] for saved in store.saved)
    assert kinds == ["document", "document", "document", "pre_exam"]
    assert len(store.usage_records) == 1
    assert store.usage_records[0].cache_hit is True


@pytest.mark.asyncio
async def test_pre_exam_summary_only_regenerates_the_changed_document(store):
    """Si cambió un documento, solo ese se vuelve a resumir: los demás salen
    de lo guardado. La síntesis se rehace porque cambió el material."""
    documents, chunks_by_doc = _pre_exam_setup(3)
    llm = _fake_llm()
    await summarize_pre_exam(
        course_id=7, db=_pre_exam_db(documents, chunks_by_doc), llm=llm
    )
    llm.chat_completion.reset_mock()
    store.usage_records.clear()

    documents[1].file_hash = "hash-reemplazado"  # CONT-07: mismo id, archivo nuevo
    await summarize_pre_exam(
        course_id=7, db=_pre_exam_db(documents, chunks_by_doc), llm=llm
    )

    assert llm.chat_completion.await_count == 2  # el documento nuevo + la síntesis
    assert len(store.usage_records) == 2  # los otros dos documentos: aciertos


@pytest.mark.asyncio
async def test_pre_exam_summary_is_not_stored_when_a_document_failed(store):
    """Un repaso armado con menos documentos porque uno falló no queda como
    "el resumen del curso"."""
    documents, chunks_by_doc = _pre_exam_setup(3)
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("LLM caído")
        result = MagicMock()
        result.text = "resumen"
        result.prompt_tokens = 10
        result.completion_tokens = 5
        result.model = "m"
        result.provider = "p"
        return result

    llm = _fake_llm()
    llm.chat_completion = AsyncMock(side_effect=flaky)

    result = await summarize_pre_exam(
        course_id=7, db=_pre_exam_db(documents, chunks_by_doc), llm=llm
    )

    assert result["total_documents"] == 2
    kinds = sorted(saved["kind"] for saved in store.saved)
    assert kinds == ["document", "document"]  # los dos que salieron, ningún pre_exam


@pytest.mark.asyncio
async def test_pre_exam_key_depends_on_the_document_set_and_versions(monkeypatch):
    keys = ["k1", "k2"]
    base = summarizer._pre_exam_key(7, keys, "m")

    assert summarizer._pre_exam_key(7, ["k2", "k1"], "m") == base  # el orden no importa
    assert summarizer._pre_exam_key(7, keys + ["k3"], "m") != base  # otro documento
    assert summarizer._pre_exam_key(8, keys, "m") != base  # otro curso
    monkeypatch.setattr(summarizer, "_SYNTHESIS_PROMPT_VERSION", "s99")
    assert summarizer._pre_exam_key(7, keys, "m") != base
