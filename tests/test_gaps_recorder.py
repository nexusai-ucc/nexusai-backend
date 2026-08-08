"""
Tests de app/gaps/recorder.py — específicamente el embedding opcional
agregado en DOC-D06 (issue #313). La lógica de detección de gaps
(no_chunks/weak_match/llm_said_no) ya funcionaba antes de esta issue y no
se tocó — estos tests cubren el comportamiento nuevo.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.gaps.recorder import record_gap_if_needed


@pytest.mark.asyncio
async def test_records_embedding_when_embeddings_provider_given():
    db = MagicMock()
    embeddings = AsyncMock()
    embeddings.embed.return_value = [0.1, 0.2, 0.3]

    recorded = await record_gap_if_needed(
        db,
        course_id=1,
        user_id=1,
        question="¿qué es overfitting?",
        chunks_count=0,
        max_similarity=None,
        embeddings=embeddings,
    )

    assert recorded is True
    gap = db.add.call_args[0][0]
    assert gap.embedding == [0.1, 0.2, 0.3]
    embeddings.embed.assert_awaited_once_with("¿qué es overfitting?")


@pytest.mark.asyncio
async def test_records_gap_without_embedding_when_no_provider_given():
    """Backward compat: embeddings=None (default) sigue registrando el gap,
    solo que sin vector — mismo comportamiento que antes de DOC-D06."""
    db = MagicMock()

    recorded = await record_gap_if_needed(
        db,
        course_id=1,
        user_id=1,
        question="¿qué es overfitting?",
        chunks_count=0,
        max_similarity=None,
    )

    assert recorded is True
    gap = db.add.call_args[0][0]
    assert gap.embedding is None


@pytest.mark.asyncio
async def test_records_gap_even_if_embedding_fails():
    """Un embed fallido nunca debe impedir registrar el gap en sí."""
    db = MagicMock()
    embeddings = AsyncMock()
    embeddings.embed.side_effect = RuntimeError("Gemini caído")

    recorded = await record_gap_if_needed(
        db,
        course_id=1,
        user_id=1,
        question="¿qué es overfitting?",
        chunks_count=0,
        max_similarity=None,
        embeddings=embeddings,
    )

    assert recorded is True
    gap = db.add.call_args[0][0]
    assert gap.embedding is None


@pytest.mark.asyncio
async def test_does_not_embed_when_gap_not_needed():
    """Si no hace falta registrar el gap, ni siquiera se llama a embed()
    (evita gastar una llamada al provider al pedo)."""
    db = MagicMock()
    embeddings = AsyncMock()

    recorded = await record_gap_if_needed(
        db,
        course_id=1,
        user_id=1,
        question="pregunta bien respondida",
        chunks_count=5,
        max_similarity=0.9,
        embeddings=embeddings,
    )

    assert recorded is False
    embeddings.embed.assert_not_called()
    db.add.assert_not_called()
