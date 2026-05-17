"""
Tests del retriever — la pieza que cierra el loop RAG: dado una pregunta y
un curso, devuelve los top-K chunks del material indexado más similares.

Estructura:
  1. Tests de `RetrievedChunk.similarity` — conversión distancia coseno → similitud
  2. Tests de `format_context_for_prompt` — función pura
  3. Tests de `retrieve_context` — validaciones de input + happy path con mocks

Para `retrieve_context` mockeamos AsyncSession y EmbeddingProvider porque no
queremos levantar Postgres para tests unitarios — eso es trabajo de los tests
de integración (que se corren contra la DB en docker-compose).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.documents.retriever import (
    RetrievedChunk,
    format_context_for_prompt,
    retrieve_context,
)


# ============================================================
# RetrievedChunk.similarity
# ============================================================

def test_similarity_distance_zero_means_perfect_match():
    """Distancia 0 (vectores idénticos) → similitud 1.0."""
    c = RetrievedChunk(content="x", document_filename="a.pdf", chunk_index=0, distance=0.0)
    assert c.similarity == pytest.approx(1.0)


def test_similarity_distance_two_means_opposite():
    """Distancia 2 (vectores opuestos) → similitud 0.0."""
    c = RetrievedChunk(content="x", document_filename="a.pdf", chunk_index=0, distance=2.0)
    assert c.similarity == pytest.approx(0.0)


def test_similarity_midpoint():
    """Distancia 1 (vectores ortogonales) → similitud 0.5."""
    c = RetrievedChunk(content="x", document_filename="a.pdf", chunk_index=0, distance=1.0)
    assert c.similarity == pytest.approx(0.5)


def test_similarity_clamps_to_zero_one_range():
    """Si por error la distancia es > 2 (no debería), no rompemos."""
    c_neg = RetrievedChunk(content="x", document_filename="a.pdf", chunk_index=0, distance=-0.5)
    c_huge = RetrievedChunk(content="x", document_filename="a.pdf", chunk_index=0, distance=3.0)
    assert 0.0 <= c_neg.similarity <= 1.0
    assert 0.0 <= c_huge.similarity <= 1.0


# ============================================================
# format_context_for_prompt
# ============================================================

def test_format_empty_list_returns_empty_string():
    assert format_context_for_prompt([]) == ""


def test_format_single_chunk_includes_filename_and_index():
    chunks = [
        RetrievedChunk(
            content="Una derivada mide la tasa instantánea de cambio.",
            document_filename="apunte-derivadas.pdf",
            chunk_index=3,
            distance=0.1,
        )
    ]
    output = format_context_for_prompt(chunks)
    assert "FRAGMENTO 1" in output
    assert "apunte-derivadas.pdf" in output
    assert "chunk #3" in output
    assert "derivada mide" in output


def test_format_multiple_chunks_are_numbered_sequentially():
    chunks = [
        RetrievedChunk(content=f"contenido {i}", document_filename=f"doc{i}.pdf", chunk_index=i, distance=0.1)
        for i in range(3)
    ]
    output = format_context_for_prompt(chunks)
    assert "FRAGMENTO 1" in output
    assert "FRAGMENTO 2" in output
    assert "FRAGMENTO 3" in output
    # Aparecen en el mismo orden que se pasaron.
    assert output.index("FRAGMENTO 1") < output.index("FRAGMENTO 2") < output.index("FRAGMENTO 3")


def test_format_truncates_very_long_content():
    """Defensa contra prompt-injection: chunks > 800 chars se truncan."""
    long_content = "A" * 2000
    chunks = [
        RetrievedChunk(content=long_content, document_filename="big.pdf", chunk_index=0, distance=0.1)
    ]
    output = format_context_for_prompt(chunks)
    # No debe contener los 2000 chars, debe tener "..." al final del fragmento.
    assert "AAAA..." in output
    assert "A" * 1500 not in output


# ============================================================
# retrieve_context — validaciones de input
# ============================================================

@pytest.mark.asyncio
async def test_retrieve_context_rejects_empty_question():
    db = MagicMock()
    embeddings = MagicMock()
    with pytest.raises(ValueError, match="non-empty"):
        await retrieve_context(question="", course_id=1, db=db, embeddings=embeddings)


@pytest.mark.asyncio
async def test_retrieve_context_rejects_whitespace_question():
    db = MagicMock()
    embeddings = MagicMock()
    with pytest.raises(ValueError, match="non-empty"):
        await retrieve_context(question="   \n\t  ", course_id=1, db=db, embeddings=embeddings)


@pytest.mark.asyncio
async def test_retrieve_context_rejects_zero_top_k():
    db = MagicMock()
    embeddings = MagicMock()
    with pytest.raises(ValueError, match="positive"):
        await retrieve_context(
            question="hola", course_id=1, db=db, embeddings=embeddings, top_k=0
        )


@pytest.mark.asyncio
async def test_retrieve_context_rejects_negative_top_k():
    db = MagicMock()
    embeddings = MagicMock()
    with pytest.raises(ValueError, match="positive"):
        await retrieve_context(
            question="hola", course_id=1, db=db, embeddings=embeddings, top_k=-3
        )


# ============================================================
# retrieve_context — happy path
# ============================================================

def _fake_row(content: str, chunk_index: int, filename: str, distance: float) -> MagicMock:
    """Helper para construir filas mockeadas con los atributos que el código espera."""
    row = MagicMock()
    row.content = content
    row.chunk_index = chunk_index
    row.filename = filename
    row.distance = distance
    return row


@pytest.mark.asyncio
async def test_retrieve_context_returns_chunks_above_min_similarity():
    """retrieve_context: filtra chunks por similitud y devuelve los que pasan."""
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 768)

    db = MagicMock()
    fake_result = MagicMock()
    # Tres filas con distancias distintas → similitudes 0.95, 0.5, 0.1
    fake_result.all.return_value = [
        _fake_row("contenido bueno", 0, "doc-a.pdf", distance=0.1),
        _fake_row("contenido medio", 1, "doc-b.pdf", distance=1.0),
        _fake_row("contenido lejano", 2, "doc-c.pdf", distance=1.8),
    ]
    db.execute = AsyncMock(return_value=fake_result)

    chunks = await retrieve_context(
        question="¿qué es una derivada?",
        course_id=1,
        db=db,
        embeddings=embeddings,
        top_k=5,
        min_similarity=0.3,
    )

    # 1.8 distance → 0.1 similarity → filtrado.
    assert len(chunks) == 2
    assert chunks[0].document_filename == "doc-a.pdf"
    assert chunks[1].document_filename == "doc-b.pdf"
    # Embeddings.embed se llamó UNA SOLA vez con la pregunta.
    embeddings.embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_context_returns_empty_when_no_chunks_indexed():
    """Si el curso no tiene material indexado, devuelve [] sin reventar."""
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 768)

    db = MagicMock()
    fake_result = MagicMock()
    fake_result.all.return_value = []
    db.execute = AsyncMock(return_value=fake_result)

    chunks = await retrieve_context(
        question="hola",
        course_id=999,
        db=db,
        embeddings=embeddings,
    )
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_context_returns_empty_when_all_below_threshold():
    """Si todos los hits tienen similitud baja, devuelve [] (no inventa contexto)."""
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 768)

    db = MagicMock()
    fake_result = MagicMock()
    fake_result.all.return_value = [
        _fake_row("ruido", 0, "doc.pdf", distance=1.9),
    ]
    db.execute = AsyncMock(return_value=fake_result)

    chunks = await retrieve_context(
        question="pregunta off-topic",
        course_id=1,
        db=db,
        embeddings=embeddings,
        min_similarity=0.3,
    )
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_context_preserves_order_from_db():
    """El retriever no re-ordena: confía en el ORDER BY de la query."""
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=[0.1] * 768)

    db = MagicMock()
    fake_result = MagicMock()
    fake_result.all.return_value = [
        _fake_row("a", 0, "x.pdf", distance=0.1),
        _fake_row("b", 1, "y.pdf", distance=0.2),
        _fake_row("c", 2, "z.pdf", distance=0.3),
    ]
    db.execute = AsyncMock(return_value=fake_result)

    chunks = await retrieve_context(
        question="q", course_id=1, db=db, embeddings=embeddings
    )
    assert [c.content for c in chunks] == ["a", "b", "c"]
    assert [c.document_filename for c in chunks] == ["x.pdf", "y.pdf", "z.pdf"]
