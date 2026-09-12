"""
Tests de moderación de contenido integrada en POST /forums/suggest-reply.

Misma estrategia de aislamiento que test_forums_router.py: mini FastAPI solo
con el forums router, dependencias mockeadas, sin llamadas reales a Postgres
ni a APIs externas.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

_SUGGEST_PAYLOAD = {
    "discussion_id": 1,
    "course_id": 1,
    "posts": [{"post_id": 1, "author": "Alumno 1", "content": "¿Cómo resuelvo esto?"}],
    "question": "¿Cómo resuelvo esto?",
}


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.all.return_value = []
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
async def client(mock_db, mock_embeddings, mock_llm):
    from app.forums.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/forums")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _fake_settings(moderation_api_key: str | None):
    from types import SimpleNamespace
    return SimpleNamespace(
        moderation_enabled=True,
        moderation_api_key=moderation_api_key,
        moderation_fail_open=True,
    )


async def test_suggest_reply_blocks_flagged_question_before_rag(client, mock_db, mock_embeddings, mock_llm):
    """El post a responder es inapropiado: se bloquea con 400 y NUNCA se
    llega a gastar en RAG (embeddings) ni en la llamada principal al LLM."""
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings(None)):
        # El LLM inyectado en el endpoint es el mismo usado por el fallback
        # de moderación (agnóstico de proveedor) — acá simula "flagged".
        mock_llm.chat_completion.return_value = MagicMock(
            text='{"flagged": true, "categories": ["harassment"]}'
        )

        response = await client.post("/api/v1/forums/suggest-reply", json=_SUGGEST_PAYLOAD)

    assert response.status_code == 400
    assert "no cumple" in response.json()["detail"].lower()

    # Costo evitado: ni el embedding del RAG ni una segunda llamada al LLM
    # (la de generar la respuesta sugerida) deberían haberse disparado.
    mock_embeddings.embed.assert_not_called()
    assert mock_llm.chat_completion.await_count == 1  # solo la del clasificador


async def test_suggest_reply_allows_acceptable_question(client, mock_db, mock_embeddings, mock_llm):
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings(None)):
        mock_llm.chat_completion.side_effect = [
            MagicMock(text='{"flagged": false, "categories": []}'),  # clasificador
            MagicMock(text="Respuesta sugerida para el alumno."),     # generación real
        ]

        response = await client.post("/api/v1/forums/suggest-reply", json=_SUGGEST_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["suggested_reply"] == "Respuesta sugerida para el alumno."
    assert mock_llm.chat_completion.await_count == 2


async def test_suggest_reply_blocks_flagged_content_in_thread_posts(client, mock_db, mock_embeddings, mock_llm):
    """La pregunta (`payload.question`) es aceptable, pero un post anterior
    del hilo (`payload.posts`) es inapropiado. La respuesta sugerida se
    sintetiza combinando ambos, así que el post del hilo también tiene que
    pasar por moderación — no alcanza con revisar solo la pregunta."""
    payload = {
        "discussion_id": 1,
        "course_id": 1,
        "posts": [
            {"post_id": 1, "author": "Alumno 1", "content": "contenido de odio en un post anterior"},
        ],
        "question": "¿Cómo resuelvo esto?",
    }

    with patch("app.shared.moderation.get_settings", return_value=_fake_settings(None)):
        mock_llm.chat_completion.return_value = MagicMock(
            text='{"flagged": true, "categories": ["hate"]}'
        )

        response = await client.post("/api/v1/forums/suggest-reply", json=payload)

    assert response.status_code == 400
    assert "no cumple" in response.json()["detail"].lower()
    mock_embeddings.embed.assert_not_called()
    assert mock_llm.chat_completion.await_count == 1  # solo la del clasificador


async def test_suggest_reply_moderation_failure_is_fail_safe_not_500(client, mock_db, mock_embeddings, mock_llm):
    """Si el servicio de moderación falla por completo (sin API key y el LLM
    de clasificación también revienta), el fail-safe (fail-open por default)
    debe dejar pasar el mensaje en vez de tumbar el endpoint con un 500."""
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings(None)):
        mock_llm.chat_completion.side_effect = [
            Exception("proveedor caído"),                          # clasificador falla
            MagicMock(text="Respuesta sugerida para el alumno."),   # generación real sigue andando
        ]

        response = await client.post("/api/v1/forums/suggest-reply", json=_SUGGEST_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["suggested_reply"] == "Respuesta sugerida para el alumno."
