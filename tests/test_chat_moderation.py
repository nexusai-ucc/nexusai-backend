"""
Tests de moderación de contenido integrada en POST /chat/messages.

Misma estrategia de aislamiento que test_chat_message_feedback.py: mini
FastAPI solo con el chat router, dependencias mockeadas, sin llamadas
reales a Postgres, Redis ni APIs externas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.infrastructure.redis_client import get_redis
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

_PAYLOAD = {
    "question": "¿Qué es el teorema de Bayes?",
    "course_id": 1,
    "user_id": 7,
}


def _fake_message(**overrides):
    base = dict(
        id=uuid4(),
        role="assistant",
        content="Respuesta de prueba.",
        created_at=datetime.now(timezone.utc),
        token_count_prompt=10,
        token_count_completion=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _assign_id_on_add(obj):
    """Simula el default `id=uuid.uuid4` de los modelos (ChatSession/Message)
    que normalmente aplica SQLAlchemy al hacer flush contra un engine real."""
    if getattr(obj, "id", None) is None:
        obj.id = uuid4()


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock(side_effect=_assign_id_on_add)

    query_result = MagicMock()
    query_result.all.return_value = []  # retrieve_context
    query_result.scalars.return_value.all.return_value = [
        _fake_message(role="user", content=_PAYLOAD["question"]),
        _fake_message(role="assistant", content="Respuesta de prueba."),
    ]
    db.execute = AsyncMock(return_value=query_result)
    return db


@pytest.fixture
def mock_embeddings():
    emb = AsyncMock(spec=EmbeddingProvider)
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMProvider)


@pytest.fixture
def mock_redis():
    pipe = MagicMock()
    pipe.execute = AsyncMock(
        return_value=[1, True]
    )  # 1 request en la ventana, bajo el límite
    redis_mock = MagicMock()
    redis_mock.pipeline.return_value = pipe
    return redis_mock


@pytest.fixture
async def client(mock_db, mock_embeddings, mock_llm, mock_redis):
    from app.chat.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/chat")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_redis] = lambda: mock_redis

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _fake_settings(*, moderation_api_key=None, moderation_fail_open=True):
    return SimpleNamespace(
        moderation_enabled=True,
        moderation_api_key=moderation_api_key,
        moderation_fail_open=moderation_fail_open,
    )


async def test_messages_blocks_inappropriate_question_before_llm_answer(
    client, mock_db, mock_embeddings, mock_llm
):
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.return_value = MagicMock(
            text='{"flagged": true, "categories": ["hate"]}'
        )

        response = await client.post("/api/v1/chat/messages", json=_PAYLOAD)

    assert response.status_code == 400
    assert "no cumple" in response.json()["detail"].lower()

    # Se bloqueó ANTES de crear la sesión/persistir el mensaje del alumno y
    # antes del RAG — el único gasto fue el clasificador de moderación.
    mock_db.add.assert_not_called()
    mock_embeddings.embed.assert_not_called()
    assert mock_llm.chat_completion.await_count == 1


async def test_messages_allows_acceptable_question(
    client, mock_db, mock_embeddings, mock_llm
):
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.side_effect = [
            MagicMock(text='{"flagged": false, "categories": []}'),  # clasificador
            MagicMock(  # respuesta real del chat
                text="La respuesta es...",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        ]

        response = await client.post("/api/v1/chat/messages", json=_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["answer"] == "La respuesta es..."
    assert mock_llm.chat_completion.await_count == 2


async def test_messages_moderation_failure_is_fail_safe_not_500(
    client, mock_db, mock_embeddings, mock_llm
):
    """Si la moderación falla por completo, el chat sigue funcionando
    (fail-open, el default) en vez de devolver un 500 al alumno."""
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.side_effect = [
            Exception("proveedor caído durante la clasificación"),
            MagicMock(
                text="La respuesta es...",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        ]

        response = await client.post("/api/v1/chat/messages", json=_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["answer"] == "La respuesta es..."
