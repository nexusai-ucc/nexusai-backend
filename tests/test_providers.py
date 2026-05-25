"""
Tests de los providers (LLM y Embeddings).

Mockeamos el cliente AsyncOpenAI para no hacer llamadas reales en CI.
Si querés correr contra Gemini real, remové los mocks y exportá las env vars.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider


# ============================================================
# LLMProvider — chat_completion
# ============================================================

@pytest.mark.asyncio
async def test_chat_completion_returns_message_content(fake_openai_chat_response):
    """chat_completion devuelve CompletionResult con .text y token counts."""
    provider = LLMProvider()
    fake_openai_chat_response.usage.prompt_tokens = 10
    fake_openai_chat_response.usage.completion_tokens = 5
    fake_openai_chat_response.usage.total_tokens = 15
    provider.client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    result = await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}]
    )
    assert result.text == "respuesta mockeada"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_completion_empty_content_returns_empty_string():
    """Si el LLM devuelve content=None, .text es '' (no None)."""
    provider = LLMProvider()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    response.choices = [choice]
    response.usage = None  # sin usage → tokens en 0
    provider.client.chat.completions.create = AsyncMock(return_value=response)

    result = await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}]
    )
    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


# ============================================================
# LLMProvider — chat_stream
# ============================================================

@pytest.mark.asyncio
async def test_chat_stream_yields_deltas():
    """chat_stream debería ceder solo los deltas de texto, no chunks crudos."""
    provider = LLMProvider()

    async def fake_stream():
        for word in ["Hola", " ", "mundo"]:
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta.content = word
            chunk.choices = [choice]
            yield chunk

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    deltas = []
    async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
        deltas.append(d)

    assert deltas == ["Hola", " ", "mundo"]


@pytest.mark.asyncio
async def test_chat_stream_skips_empty_chunks():
    """Chunks sin choices o sin content no deberían yieldearse."""
    provider = LLMProvider()

    async def fake_stream():
        # Chunk válido
        chunk1 = MagicMock()
        choice1 = MagicMock()
        choice1.delta.content = "ok"
        chunk1.choices = [choice1]
        yield chunk1

        # Chunk sin choices (algunos providers lo mandan al final)
        chunk2 = MagicMock()
        chunk2.choices = []
        yield chunk2

        # Chunk con delta vacío
        chunk3 = MagicMock()
        choice3 = MagicMock()
        choice3.delta.content = None
        chunk3.choices = [choice3]
        yield chunk3

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    deltas = []
    async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
        deltas.append(d)

    assert deltas == ["ok"]


# ============================================================
# EmbeddingProvider
# ============================================================

@pytest.mark.asyncio
async def test_embed_returns_vector_with_correct_dimensions(
    fake_openai_embedding_response,
):
    provider = EmbeddingProvider()
    provider.client.embeddings.create = AsyncMock(
        return_value=fake_openai_embedding_response
    )

    vector = await provider.embed("texto cualquiera")
    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_embed_rejects_empty_text():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="empty"):
        await provider.embed("")
    with pytest.raises(ValueError, match="empty"):
        await provider.embed("   ")  # solo whitespace también es "vacío"


@pytest.mark.asyncio
async def test_embed_validates_dimensions_mismatch():
    """Si el modelo devuelve N dimensiones distintas a las configuradas, error."""
    provider = EmbeddingProvider()
    response = MagicMock()
    data = MagicMock()
    data.index = 0
    data.embedding = [0.1] * 1024  # esperamos 768, devuelve 1024
    response.data = [data]
    provider.client.embeddings.create = AsyncMock(return_value=response)

    with pytest.raises(RuntimeError, match="no coincide"):
        await provider.embed("texto")


@pytest.mark.asyncio
async def test_embed_many_preserves_order():
    """Si el SDK devuelve los embeddings desordenados, los re-ordenamos por índice."""
    provider = EmbeddingProvider()
    response = MagicMock()
    # Devolvemos en orden 2, 0, 1 — el provider tiene que re-ordenarlos.
    d2 = MagicMock(); d2.index = 2; d2.embedding = [0.3] * 768
    d0 = MagicMock(); d0.index = 0; d0.embedding = [0.1] * 768
    d1 = MagicMock(); d1.index = 1; d1.embedding = [0.2] * 768
    response.data = [d2, d0, d1]
    provider.client.embeddings.create = AsyncMock(return_value=response)

    vectors = await provider.embed_many(["a", "b", "c"])
    assert vectors[0][0] == pytest.approx(0.1)
    assert vectors[1][0] == pytest.approx(0.2)
    assert vectors[2][0] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_embed_many_rejects_empty_list():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="empty list"):
        await provider.embed_many([])


@pytest.mark.asyncio
async def test_embed_many_rejects_empty_strings():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="non-empty"):
        await provider.embed_many(["ok", ""])
