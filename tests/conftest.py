"""
Configuración global de pytest.

Setea env vars de test ANTES de que se importe `app.shared.config` —
sino Pydantic Settings explota porque las vars marcadas requeridas
(NEXUSAI_API_KEY, etc.) no están en el entorno del CI/dev.

También provee fixtures comunes: cliente HTTP de test, mock de Redis,
mock del SDK de OpenAI.
"""

from __future__ import annotations

import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

# ============================================================
# Env vars de test — DEBEN setearse ANTES de cualquier import de app.*
# ============================================================
os.environ.setdefault("ENV", "test")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("LLM_BASE_URL", "https://test.example/v1")
os.environ.setdefault("LLM_MODEL", "gemini-2.0-flash")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embed-key")
os.environ.setdefault("EMBEDDING_BASE_URL", "https://test.example/v1")
os.environ.setdefault("EMBEDDING_MODEL", "models/text-embedding-004")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")  # DB 15 para tests
os.environ.setdefault("NEXUSAI_SHARED_SECRET", "test-shared-secret-32-chars-long")
os.environ.setdefault("NEXUSAI_API_KEY", "test-api-key-32-chars-long-okok")
os.environ.setdefault("HMAC_REPLAY_WINDOW_SEC", "300")
os.environ.setdefault("RATE_LIMIT_PER_USER_DAILY", "50")


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def fake_redis() -> MagicMock:
    """
    Mock de un cliente Redis async. Comportamiento default:
      - SET con NX devuelve True (nonce nuevo, no es replay).
      - GET devuelve None.
    Los tests pueden sobreescribir el comportamiento per-test.
    """
    redis_mock = MagicMock()
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.aclose = AsyncMock(return_value=None)
    return redis_mock


@pytest.fixture
def fake_openai_chat_response() -> MagicMock:
    """Mock de una response no-streaming del SDK de OpenAI."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "respuesta mockeada"
    response.choices = [choice]
    return response


@pytest.fixture
def fake_openai_embedding_response() -> MagicMock:
    """Mock de una response del endpoint embeddings."""
    response = MagicMock()
    # 768 dimensiones para que matchee con EMBEDDING_DIMENSIONS de test.
    data = MagicMock()
    data.index = 0
    data.embedding = [0.1] * 768
    response.data = [data]
    return response


@pytest.fixture
async def fake_chat_stream() -> AsyncIterator[MagicMock]:
    """
    Mock de un stream de chunks del SDK de OpenAI.
    Cada chunk simula un delta con 1 palabra.
    """

    async def gen():
        for word in ["Hola", " ", "mundo", "."]:
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta.content = word
            chunk.choices = [choice]
            yield chunk

    return gen()
