"""
Tests del privacy router — export/delete de datos personales (PRIV-01, #310).

Misma estrategia de aislamiento que test_quiz_router.py:
  - Mini FastAPI solo con el privacy router.
  - verify_hmac y get_db reemplazados con mocks.
  - Sin llamadas reales a Postgres.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.privacy.router import router


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
async def client(mock_db):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/privacy")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _now():
    return datetime.now(timezone.utc)


# ============================================================
# GET /export
# ============================================================

@pytest.mark.asyncio
async def test_export_rejects_non_positive_ids(client):
    resp = await client.get("/api/v1/privacy/export", params={"user_id": 0, "course_id": 5})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_export_returns_messages_from_all_sessions(client, mock_db):
    session1 = SimpleNamespace(
        id=uuid4(),
        messages=[
            SimpleNamespace(role="user", content="hola", created_at=_now()),
            SimpleNamespace(role="assistant", content="hola, ¿en qué te ayudo?", created_at=_now()),
        ],
    )
    session2 = SimpleNamespace(
        id=uuid4(),
        messages=[SimpleNamespace(role="user", content="otra pregunta", created_at=_now())],
    )

    sessions_result = MagicMock()
    sessions_result.scalars.return_value.all.return_value = [session1, session2]

    attempts_result = MagicMock()
    attempts_result.scalars.return_value.all.return_value = []

    errors_result = MagicMock()
    errors_result.scalars.return_value.all.return_value = []

    mock_db.execute = AsyncMock(side_effect=[sessions_result, attempts_result, errors_result])

    resp = await client.get("/api/v1/privacy/export", params={"user_id": 7, "course_id": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 7
    assert body["course_id"] == 3
    assert len(body["messages"]) == 3
    assert body["messages"][0]["content"] == "hola"


@pytest.mark.asyncio
async def test_export_excludes_anonymized_quiz_attempts_at_query_level(client, mock_db):
    """El query de attempts filtra deleted_at IS NULL — verificamos que el
    endpoint solo devuelve lo que el mock (ya filtrado) le da, sin duplicar
    lógica de filtrado en la capa de respuesta."""
    attempt = SimpleNamespace(
        id=uuid4(),
        question_type="multiple_choice",
        difficulty="medium",
        topic="derivadas",
        total_questions=10,
        correct_answers=8,
        score=0.8,
        created_at=_now(),
    )

    sessions_result = MagicMock()
    sessions_result.scalars.return_value.all.return_value = []
    attempts_result = MagicMock()
    attempts_result.scalars.return_value.all.return_value = [attempt]
    errors_result = MagicMock()
    errors_result.scalars.return_value.all.return_value = []

    mock_db.execute = AsyncMock(side_effect=[sessions_result, attempts_result, errors_result])

    resp = await client.get("/api/v1/privacy/export", params={"user_id": 7, "course_id": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["quiz_attempts"]) == 1
    assert body["quiz_attempts"][0]["score"] == 0.8


# ============================================================
# DELETE /data
# ============================================================

@pytest.mark.asyncio
async def test_delete_rejects_non_positive_ids(client):
    resp = await client.request(
        "DELETE", "/api/v1/privacy/data", params={"user_id": -1, "course_id": 5}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_reports_counts_from_each_operation(client, mock_db):
    messages_result = MagicMock(rowcount=4)
    chat_sessions_result = MagicMock(rowcount=2)  # no se reporta, pero se ejecuta
    quiz_errors_result = MagicMock(rowcount=3)
    quiz_attempts_result = MagicMock(rowcount=5)

    mock_db.execute = AsyncMock(
        side_effect=[messages_result, chat_sessions_result, quiz_errors_result, quiz_attempts_result]
    )
    mock_db.commit = AsyncMock()

    resp = await client.request(
        "DELETE", "/api/v1/privacy/data", params={"user_id": 7, "course_id": 3}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["messages_deleted"] == 4
    assert body["quiz_errors_deleted"] == 3
    assert body["quiz_attempts_anonymized"] == 5
    assert mock_db.execute.call_count == 4
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_anonymizes_quiz_attempts_instead_of_deleting(client, mock_db):
    """El 4to execute() tiene que ser un UPDATE (anonimizar), no un DELETE —
    confirma que quiz_attempts nunca se borra crudo (rompería Analytics)."""
    calls = []

    async def fake_execute(stmt):
        calls.append(stmt)
        return MagicMock(rowcount=1)

    mock_db.execute = fake_execute
    mock_db.commit = AsyncMock()

    await client.request("DELETE", "/api/v1/privacy/data", params={"user_id": 7, "course_id": 3})

    assert len(calls) == 4
    # El 4to statement es un Update sobre QuizAttempt, no un Delete.
    fourth_stmt = calls[3]
    assert type(fourth_stmt).__name__ == "Update"
    assert "quiz_attempts" in str(fourth_stmt).lower()
