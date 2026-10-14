"""
Tests de la visibilidad del material (VIS-01, issue #536).

Lo que importa: un documento solo entra en una respuesta si su `cmid` está en
la lista que mandó el plugin; sin lista el pedido se rechaza; lista vacía no
devuelve nada y sin gastar embeddings ni consultas. La DB va mockeada, así que
el filtro se verifica mirando el SQL compilado.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.documents.retriever import retrieve_context
from app.documents.summarizer import (
    _load_document_for_summary,
    summarize_document,
    summarize_pre_exam,
)
from app.quiz.router import _sample_chunks_for_quiz
from app.shared.visibility import enforce_visible_cmids


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


# ---------- enforce_visible_cmids ----------


def test_enforce_rejects_missing_list_by_default():
    with patch(
        "app.shared.visibility.get_settings",
        return_value=SimpleNamespace(require_visible_cmids=True),
    ):
        with pytest.raises(HTTPException) as exc:
            enforce_visible_cmids(None)
    assert exc.value.status_code == 422


def test_enforce_keeps_an_empty_list_as_nothing_visible():
    with patch(
        "app.shared.visibility.get_settings",
        return_value=SimpleNamespace(require_visible_cmids=True),
    ):
        assert enforce_visible_cmids([]) == []


def test_enforce_returns_the_list_untouched():
    assert enforce_visible_cmids([3, 9]) == [3, 9]


def test_enforce_lets_a_missing_list_through_when_the_requirement_is_off():
    with patch(
        "app.shared.visibility.get_settings",
        return_value=SimpleNamespace(require_visible_cmids=False),
    ):
        assert enforce_visible_cmids(None) is None


def test_requirement_is_on_by_default():
    from app.shared.config import Settings

    assert Settings.model_fields["require_visible_cmids"].default is True


# ---------- retrieve_context ----------


def _embeddings():
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[0.1] * 768)
    return emb


def _db_returning(rows):
    result = MagicMock()
    result.all.return_value = rows
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_retrieve_with_empty_visible_list_returns_nothing_without_embedding():
    emb, db = _embeddings(), _db_returning([])

    chunks = await retrieve_context("hola", 1, db, emb, visible_cmids=[])

    assert chunks == []
    emb.embed.assert_not_called()
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_filters_by_cmid_when_a_list_is_given():
    db = _db_returning([])

    await retrieve_context("hola", 1, db, _embeddings(), visible_cmids=[10, 11])

    sql = _sql(db.execute.await_args.args[0])
    assert "documents.cmid IN" in sql


@pytest.mark.asyncio
async def test_retrieve_does_not_filter_by_cmid_without_a_list():
    db = _db_returning([])

    await retrieve_context("hola", 1, db, _embeddings())

    assert "cmid" not in _sql(db.execute.await_args.args[0])


# ---------- quiz ----------


@pytest.mark.asyncio
async def test_quiz_sampling_with_empty_list_returns_nothing():
    db = _db_returning([])

    assert await _sample_chunks_for_quiz(db, 1, visible_cmids=[]) == []
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_quiz_sampling_filters_by_cmid():
    db = _db_returning([])

    await _sample_chunks_for_quiz(db, 1, visible_cmids=[7])

    assert "documents.cmid IN" in _sql(db.execute.await_args.args[0])


# ---------- summarizer ----------


def _doc(cmid):
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.course_id = 7
    doc.cmid = cmid
    return doc


def _db_with_document(doc):
    result = MagicMock()
    result.scalar_one_or_none.return_value = doc
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("cmid", [None, 999])
async def test_summary_of_a_document_that_is_not_visible_looks_like_not_found(cmid):
    doc = _doc(cmid)
    db = _db_with_document(doc)

    with pytest.raises(LookupError, match="not found"):
        await _load_document_for_summary(doc.id, 7, db, visible_cmids=[10])

    # Solo se leyó el documento: ni chunks, ni resúmenes guardados.
    assert db.execute.await_count == 1


@pytest.mark.asyncio
async def test_summarize_document_never_serves_a_stored_summary_for_hidden_material():
    doc = _doc(55)
    db = _db_with_document(doc)
    llm = MagicMock()
    llm.model = "m"

    with patch("app.documents.summarizer.summary_store.lookup") as lookup:
        with pytest.raises(LookupError):
            await summarize_document(doc.id, 7, db, llm, visible_cmids=[10])

    lookup.assert_not_called()


@pytest.mark.asyncio
async def test_pre_exam_summary_with_empty_list_returns_nothing_without_querying():
    db = _db_returning([])

    result = await summarize_pre_exam(7, db, MagicMock(), visible_cmids=[])

    assert result == {"summary": "", "documents_used": [], "total_documents": 0}
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pre_exam_summary_only_lists_visible_documents():
    result = MagicMock()
    result.all.return_value = []
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)

    await summarize_pre_exam(7, db, MagicMock(), visible_cmids=[10, 11])

    assert "documents.cmid IN" in _sql(db.execute.await_args.args[0])
