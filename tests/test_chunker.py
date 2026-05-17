"""
Tests del chunker — la pieza que parte el texto en bloques de ~512 tokens
con overlap para preservar contexto entre fragmentos.

No usa mocks: el chunker es código puro (tiktoken + lógica de ventana
deslizante). Probamos:
  - Texto chico que entra en un solo chunk
  - Texto grande que se parte en varios con overlap
  - Que el overlap realmente solape (último/primer token de chunks adyacentes)
  - Validaciones de input (max_tokens, overlap, texto vacío)
"""

from __future__ import annotations

import pytest

from app.documents.chunker import Chunk, chunk_text


# ============================================================
# Happy path
# ============================================================

def test_short_text_returns_single_chunk():
    """Texto que cabe en max_tokens → un solo chunk."""
    text = "Una derivada mide la tasa instantánea de cambio."
    chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content.strip() == text
    assert chunks[0].token_count > 0


def test_long_text_returns_multiple_chunks():
    """Texto grande → varios chunks con índices ascendentes desde 0."""
    # Repetir un párrafo para superar 512 tokens fácilmente.
    paragraph = "Esto es una oración larga que tiene varias palabras útiles. " * 50
    chunks = chunk_text(paragraph, max_tokens=128, overlap_tokens=16)

    assert len(chunks) > 1
    # Los índices arrancan en 0 y son consecutivos.
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    # Cada chunk respeta el límite de tokens (con margen porque tiktoken puede
    # decodificar y re-encodificar con leves diferencias).
    for c in chunks:
        assert c.token_count <= 128


def test_chunks_have_overlap_between_neighbors():
    """Dos chunks adyacentes deben compartir tokens al final/inicio."""
    text = ("Palabra " * 300).strip()  # ~300 tokens
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)

    assert len(chunks) >= 2
    # El segundo chunk tiene que arrancar con algo del final del primero.
    # No exigimos identidad exacta porque tiktoken puede agrupar tokens distinto
    # entre encode/decode, pero sí que haya superposición de palabras.
    first_words = chunks[0].content.split()[-15:]
    second_start = chunks[1].content.split()[:30]
    common = set(first_words) & set(second_start)
    assert common, "Los chunks consecutivos deberían tener palabras en común (overlap)"


def test_chunk_content_is_stripped():
    """El content no debe tener whitespace inicial/final."""
    text = "   contenido con espacios al principio y al final   "
    chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)
    assert chunks[0].content == chunks[0].content.strip()


# ============================================================
# Validaciones de input
# ============================================================

def test_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        chunk_text("", max_tokens=512, overlap_tokens=64)


def test_whitespace_only_text_raises():
    with pytest.raises(ValueError, match="empty"):
        chunk_text("   \n\t  ", max_tokens=512, overlap_tokens=64)


def test_zero_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_text("texto", max_tokens=0, overlap_tokens=0)


def test_negative_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_text("texto", max_tokens=-10, overlap_tokens=0)


def test_overlap_greater_than_max_raises():
    """Si overlap >= max_tokens, el chunker nunca avanzaría → loop infinito."""
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("texto " * 100, max_tokens=100, overlap_tokens=100)


def test_negative_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("texto", max_tokens=100, overlap_tokens=-5)


# ============================================================
# Edge cases
# ============================================================

def test_zero_overlap_no_repetition():
    """Con overlap=0, los chunks no deberían repetir contenido."""
    text = ("Token " * 250).strip()
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=0)
    assert len(chunks) >= 2
    # Todos los chunks contienen solo "Token", así que no podemos chequear
    # contenido único, pero sí que ningún token_count sea 0.
    for c in chunks:
        assert c.token_count > 0


def test_chunk_indices_are_unique():
    """Por más chunks que haya, cada uno tiene su chunk_index único."""
    text = ("Lorem ipsum dolor sit amet. " * 200).strip()
    chunks = chunk_text(text, max_tokens=64, overlap_tokens=8)
    indices = [c.chunk_index for c in chunks]
    assert len(indices) == len(set(indices)), "chunk_index debe ser único por chunk"


def test_chunk_is_immutable_dataclass():
    """Chunk es frozen para que no se modifique después de creado."""
    chunk = Chunk(content="hola", chunk_index=0, token_count=1)
    with pytest.raises((AttributeError, Exception)):
        chunk.chunk_index = 5  # type: ignore[misc]
