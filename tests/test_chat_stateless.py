"""
Chat sin estado, el contrato nuevo de la opción C (DATA-04, issue #524).

Con `history` la conversación vive en Moodle: el backend no guarda pregunta,
respuesta, interacción ni gap, y devuelve métricas, la señal de gap con su
embedding, lo que queda del presupuesto y el consumo del pedido. Sin
`history` el flujo viejo sigue igual (lo cubren test_chat_moderation.py y el
resto).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.infrastructure.redis_client import get_redis
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import (
    CompletionResult,
    LLMProvider,
    StreamToken,
    StreamUsage,
    get_llm_provider,
)
from app.shared.middleware import RequestIDMiddleware

_PAYLOAD = {
    "question": "¿Qué es el teorema de Bayes?",
    "course_id": 1,
    "user_id": 7,
    "visible_cmids": [101],
    "history": [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, ¿en qué te ayudo?"},
    ],
    "token_limit_hourly": 5000,
    "token_limit_daily": 20000,
}

_NOT_FLAGGED = CompletionResult(text='{"flagged": false, "categories": []}')


def _answer(text: str = "La respuesta es Bayes.") -> CompletionResult:
    return CompletionResult(
        text=text,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        model="gpt-test",
        provider="openai",
        fallback=True,
    )


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def mock_embeddings():
    emb = AsyncMock(spec=EmbeddingProvider)
    emb.embed.return_value = [0.25] * 4
    return emb


@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMProvider)


@pytest.fixture
def mock_redis():
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1, True])
    redis_mock = MagicMock()
    redis_mock.pipeline.return_value = pipe
    redis_mock.mget = AsyncMock(return_value=[b"300", None])
    redis_mock.decrby = AsyncMock()
    return redis_mock


@pytest.fixture
async def client(mock_db, mock_embeddings, mock_llm, mock_redis):
    from app.chat.router import router

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.include_router(router, prefix="/api/v1/chat")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_redis] = lambda: mock_redis

    settings = SimpleNamespace(
        moderation_enabled=True, moderation_api_key=None, moderation_fail_open=True
    )
    with patch("app.shared.moderation.get_settings", return_value=settings):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


async def test_stateless_messages_saves_nothing_and_returns_what_moodle_stores(
    client, mock_db, mock_llm, mock_embeddings
):
    mock_llm.chat_completion.side_effect = [_NOT_FLAGGED, _answer()]
    with patch("app.chat.router.retrieve_context", new=AsyncMock(return_value=[])):
        response = await client.post("/api/v1/chat/messages", json=_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] is None
    assert body["messages"] == []
    assert body["answer"] == "La respuesta es Bayes."
    assert body["metrics"]["model"] == "gpt-test"
    assert body["metrics"]["fallback"] is True
    assert body["metrics"]["total_tokens"] == 120
    # Sin fragmentos el material no respondió: es gap, con su embedding.
    assert body["gap"]["is_gap"] is True
    assert body["gap"]["embedding"] == [0.25] * 4
    # Presupuesto con los límites que mandó Moodle.
    assert body["budget"]["hourly"] == {
        "limit": 5000,
        "used": 300,
        "remaining": 4700,
        "resets_in_sec": body["budget"]["hourly"]["resets_in_sec"],
    }
    assert body["budget"]["daily"]["limit"] == 20000
    # Nada guardado en la base del backend.
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()


async def test_stateless_messages_sends_moodle_history_to_the_llm(client, mock_llm):
    mock_llm.chat_completion.side_effect = [_NOT_FLAGGED, _answer()]
    with patch("app.chat.router.retrieve_context", new=AsyncMock(return_value=[])):
        await client.post("/api/v1/chat/messages", json=_PAYLOAD)

    sent = mock_llm.chat_completion.await_args_list[1].args[0]
    roles_and_text = [(m["role"], m["content"]) for m in sent[1:]]
    assert roles_and_text[:2] == [
        ("user", "Hola"),
        ("assistant", "Hola, ¿en qué te ayudo?"),
    ]
    assert roles_and_text[-1][0] == "user"
    assert "Bayes" in roles_and_text[-1][1]


async def test_a_good_answer_is_not_a_gap_and_has_no_embedding(
    client, mock_llm, mock_embeddings
):
    chunk = SimpleNamespace(
        similarity=0.9,
        content="Bayes",
        document_filename="a.pdf",
        document_id=None,
        chunk_index=0,
        course_id=1,
    )
    mock_llm.chat_completion.side_effect = [_NOT_FLAGGED, _answer()]
    with (
        patch("app.chat.router.retrieve_context", new=AsyncMock(return_value=[chunk])),
        patch("app.chat.router.format_context_for_prompt", return_value="ctx"),
    ):
        body = (await client.post("/api/v1/chat/messages", json=_PAYLOAD)).json()

    assert body["gap"]["is_gap"] is False
    assert body["gap"]["embedding"] is None
    assert body["metrics"]["has_relevant_context"] is True
    mock_embeddings.embed.assert_not_called()


async def test_history_is_limited_to_ten_messages(client):
    payload = {
        **_PAYLOAD,
        "history": [{"role": "user", "content": f"m{i}"} for i in range(11)],
    }
    response = await client.post("/api/v1/chat/messages", json=payload)
    assert response.status_code == 422


async def test_stateless_stream_puts_everything_in_done(
    client, mock_db, mock_llm, mock_embeddings
):
    mock_llm.chat_completion.return_value = _NOT_FLAGGED

    async def fake_stream(_messages, **_kwargs):
        yield StreamToken(text="Hola ")
        yield StreamToken(text="Bayes")
        yield StreamUsage(
            prompt_tokens=50,
            completion_tokens=5,
            total_tokens=55,
            model="gemini-test",
            provider="google",
        )

    mock_llm.chat_completion_stream = fake_stream

    @asynccontextmanager
    async def session_factory():
        yield mock_db

    with (
        patch("app.chat.router.retrieve_context", new=AsyncMock(return_value=[])),
        patch("app.chat.router.get_session_factory", return_value=session_factory),
    ):
        response = await client.post("/api/v1/chat/stream", json=_PAYLOAD)

    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [e["type"] for e in events]
    assert kinds == ["meta", "token", "token", "answer_meta", "done"]
    assert "session_id" not in events[0]
    done = events[-1]
    assert "assistant_message_id" not in done
    assert done["total_tokens"] == 55
    assert done["metrics"]["model"] == "gemini-test"
    assert done["gap"]["is_gap"] is True
    assert done["gap"]["embedding"] == [0.25] * 4
    assert done["budget"]["daily"]["limit"] == 20000
    assert isinstance(done["usage"], list)
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()
