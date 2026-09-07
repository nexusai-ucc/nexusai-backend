"""
Tests de POST /messages/feedback — voto 👍/👎 del alumno sobre una
respuesta del chat (ASIST-01, #321).

Misma estrategia de aislamiento que test_privacy_router.py: mini FastAPI
solo con el chat router, verify_hmac y get_db reemplazados por mocks, sin
llamadas reales a Postgres.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.analytics.logger import hash_user_id
from app.auth.hmac import verify_hmac
from app.chat.router import router
from app.db.session import get_db


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
async def client(mock_db):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/chat")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _payload(**overrides):
    base = {
        "message_id": str(uuid4()),
        "course_id": 1,
        "user_id": 7,
        "is_helpful": True,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_feedback_creates_new_row_when_no_existing_vote(client, mock_db):
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=lookup_result)
    mock_db.commit = AsyncMock()

    response = await client.post("/api/v1/chat/messages/feedback", json=_payload())

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    mock_db.add.assert_called_once()
    created = mock_db.add.call_args[0][0]
    assert created.is_helpful is True
    assert created.course_id == 1
    assert created.user_id_hash == hash_user_id(7)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_feedback_upserts_when_student_changes_vote(client, mock_db):
    existing = MagicMock(is_helpful=True, comment=None)
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = existing
    mock_db.execute = AsyncMock(return_value=lookup_result)
    mock_db.commit = AsyncMock()

    response = await client.post(
        "/api/v1/chat/messages/feedback",
        json=_payload(is_helpful=False, comment="La respuesta citaba la fuente equivocada."),
    )

    assert response.status_code == 200
    mock_db.add.assert_not_called()  # actualiza la fila existente, no crea una nueva
    assert existing.is_helpful is False
    assert existing.comment == "La respuesta citaba la fuente equivocada."


@pytest.mark.asyncio
async def test_feedback_never_stores_raw_user_id(client, mock_db):
    """Anónimo por diseño (mismo criterio que interaction_logs) — el payload
    de creación no debe contener el user_id en ningún campo persistido."""
    lookup_result = MagicMock()
    lookup_result.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=lookup_result)
    mock_db.commit = AsyncMock()

    await client.post("/api/v1/chat/messages/feedback", json=_payload(user_id=42))

    created = mock_db.add.call_args[0][0]
    assert not hasattr(created, "user_id")
    assert created.user_id_hash == hash_user_id(42)


@pytest.mark.asyncio
async def test_feedback_rejects_invalid_message_id(client):
    response = await client.post(
        "/api/v1/chat/messages/feedback",
        json=_payload(message_id="not-a-uuid"),
    )

    assert response.status_code == 422
