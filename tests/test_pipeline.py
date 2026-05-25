"""
Tests del pipeline de indexación (app.documents.pipeline).

Todos los tests usan AsyncMock para DB y EmbeddingProvider.
La función `extract_text` se parchea para evitar dependencias de formato de archivo.
No hay conexión real a Postgres ni a ninguna API externa.

Casos cubiertos:
  1. Happy path — extrae, chunkea, embeddea y persiste con status='indexed'.
  2. CONT-04 skip — si el documento ya tiene chunks, salta la indexación.
  3. Documento no encontrado en DB → ValueError.
  4. Fallo total de embeddings → status='error' con error_message.
  5. Error en extracción (PDF vacío / corrupto) → status='error'.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from app.documents.pipeline import index_document
from app.providers.embeddings import EmbeddingProvider


# ============================================================
# Helpers
# ============================================================

def _make_doc(**kwargs) -> SimpleNamespace:
    """Simula un ORM Document con atributos seteables."""
    now = datetime.now(timezone.utc)
    doc = SimpleNamespace(
        id=uuid4(),
        course_id=1,
        uploader_id=42,
        filename="apuntes.pdf",
        mime_type="application/pdf",
        status="pending",
        error_message=None,
        file_hash="abc123def456",
        created_at=now,
        updated_at=now,
    )
    for k, v in kwargs.items():
        setattr(doc, k, v)
    return doc


def _db_exec(scalar_one_or_none=None, scalar_one=0) -> MagicMock:
    """Retorna el mock apropiado para un resultado de db.execute()."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = scalar_one_or_none
    r.scalar_one.return_value = scalar_one
    return r


# ============================================================
# Test 1: happy path — indexación completa
# ============================================================

async def test_index_document_indexes_successfully():
    """
    Flujo normal: extrae texto → 2 chunks → 2 embeddings → persiste como 'indexed'.
    """
    doc = _make_doc()
    db = AsyncMock()
    # add_all es sync en SQLAlchemy → usar MagicMock para evitar coroutine warning.
    db.add_all = MagicMock()

    # execute() call 1: cargar Document desde DB
    # execute() call 2: contar chunks existentes → 0 (CONT-04 check)
    db.execute.side_effect = [
        _db_exec(scalar_one_or_none=doc),
        _db_exec(scalar_one=0),
    ]

    embeddings = AsyncMock(spec=EmbeddingProvider)
    embeddings.embed.return_value = [0.1] * 768

    with patch("app.documents.pipeline.extract_text", return_value="Hola mundo. Texto de prueba."):
        result = await index_document(doc.id, b"%PDF-fake", db, embeddings)

    assert result.status == "indexed"
    assert result.error_message is None
    db.add_all.assert_called_once()
    # commit: 1 vez para 'indexing' + 1 vez para 'indexed'
    assert db.commit.call_count == 2
    db.refresh.assert_called_once_with(doc)


# ============================================================
# Test 2: CONT-04 — skip si ya hay chunks
# ============================================================

async def test_index_document_skips_if_chunks_already_exist():
    """
    Si el documento ya tiene chunks en DB (re-encola de intento previo),
    se marca 'indexed' directamente sin extraer ni embeddear nada.
    """
    doc = _make_doc()
    db = AsyncMock()

    db.execute.side_effect = [
        _db_exec(scalar_one_or_none=doc),
        _db_exec(scalar_one=5),  # 5 chunks ya existen
    ]

    embeddings = AsyncMock(spec=EmbeddingProvider)

    with patch("app.documents.pipeline.extract_text") as mock_extract:
        result = await index_document(doc.id, b"%PDF-fake", db, embeddings)
        mock_extract.assert_not_called()

    assert result.status == "indexed"
    embeddings.embed.assert_not_called()
    db.add_all.assert_not_called()
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(doc)


# ============================================================
# Test 3: documento no encontrado
# ============================================================

async def test_index_document_raises_if_document_not_found():
    """
    Si el document_id no existe en DB, el pipeline levanta ValueError.
    """
    db = AsyncMock()
    db.execute.return_value = _db_exec(scalar_one_or_none=None)
    embeddings = AsyncMock(spec=EmbeddingProvider)

    with pytest.raises(ValueError, match="not found"):
        await index_document(uuid4(), b"bytes", db, embeddings)

    embeddings.embed.assert_not_called()


# ============================================================
# Test 4: fallo total de embeddings → status='error'
# ============================================================

async def test_index_document_marks_error_when_all_embeddings_fail():
    """
    Si todos los chunks fallan al embeddear, el pipeline marca status='error'
    con error_message y relanza la excepción.
    """
    doc = _make_doc()
    db = AsyncMock()

    # El error handler re-fetches el documento con un tercer execute()
    db.execute.side_effect = [
        _db_exec(scalar_one_or_none=doc),   # fetch inicial
        _db_exec(scalar_one=0),              # count chunks → 0
        _db_exec(scalar_one_or_none=doc),   # re-fetch en bloque except
    ]

    embeddings = AsyncMock(spec=EmbeddingProvider)
    embeddings.embed.side_effect = RuntimeError("cuota de API agotada")

    with patch("app.documents.pipeline.extract_text", return_value="Hola mundo"):
        with pytest.raises(RuntimeError):
            await index_document(doc.id, b"%PDF-fake", db, embeddings)

    assert doc.status == "error"
    assert doc.error_message is not None
    assert "cuota" in doc.error_message
    db.rollback.assert_called_once()


# ============================================================
# Test 5: error en extracción (PDF vacío / corrupto)
# ============================================================

async def test_index_document_marks_error_on_extraction_failure():
    """
    Si extract_text() falla (PDF ilegible, archivo corrupto, etc.),
    el pipeline marca status='error' y relanza la excepción.
    """
    doc = _make_doc()
    db = AsyncMock()

    # Flujo: fetch doc → count chunks → commit 'indexing' → extract falla
    # → rollback → re-fetch doc → commit 'error' → raise
    db.execute.side_effect = [
        _db_exec(scalar_one_or_none=doc),   # fetch inicial
        _db_exec(scalar_one=0),              # count chunks → 0
        _db_exec(scalar_one_or_none=doc),   # re-fetch en bloque except
    ]

    embeddings = AsyncMock(spec=EmbeddingProvider)

    with patch(
        "app.documents.pipeline.extract_text",
        side_effect=ValueError("No se pudo extraer texto del PDF"),
    ):
        with pytest.raises(ValueError, match="No se pudo extraer"):
            await index_document(doc.id, b"%PDF-fake", db, embeddings)

    assert doc.status == "error"
    assert "No se pudo extraer" in doc.error_message
    embeddings.embed.assert_not_called()
    db.rollback.assert_called_once()
