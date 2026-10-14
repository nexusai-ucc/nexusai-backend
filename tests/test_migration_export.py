"""
Export temporal para migrar los datos del alumno a Moodle (DATA-04, #524).

La consulta real se probó contra PostgreSQL; acá se cubre el interruptor, la
validación de la tabla, el cursor y la forma de las filas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.models import ChatSession
from app.db.session import get_db
from app.migration.router import _decode_cursor, _encode_cursor, _row


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
async def client(mock_db):
    from app.migration.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/migration")
    app.dependency_overrides[verify_hmac] = lambda: b""
    app.dependency_overrides[get_db] = lambda: mock_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _enabled(value: bool):
    return patch(
        "app.migration.router.get_settings",
        return_value=SimpleNamespace(migration_export_enabled=value),
    )


async def test_export_is_off_by_default(client):
    from app.shared.config import Settings

    assert Settings.model_fields["migration_export_enabled"].default is False
    with _enabled(False):
        response = await client.get("/api/v1/migration/export?table=messages")
    assert response.status_code == 404


async def test_unknown_table_is_refused(client):
    with _enabled(True):
        response = await client.get("/api/v1/migration/export?table=llm_usage")
    assert response.status_code == 400


async def test_a_full_page_returns_a_cursor_to_the_next(client, mock_db):
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    sessions = [
        ChatSession(
            id=uuid.uuid4(), user_id=7, course_id=1, created_at=now, updated_at=now
        )
        for _ in range(3)
    ]
    result = MagicMock()
    result.scalars.return_value.all.return_value = sessions
    mock_db.execute.return_value = result

    with _enabled(True):
        response = await client.get(
            "/api/v1/migration/export?table=chat_sessions&limit=2"
        )

    body = response.json()
    assert len(body["rows"]) == 2
    assert body["rows"][0]["id"] == str(sessions[0].id)
    assert body["rows"][0]["created_at"] == now.isoformat()
    assert _decode_cursor(body["next"]) == (now, str(sessions[1].id))


async def test_the_last_page_has_no_cursor(client, mock_db):
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = result
    with _enabled(True):
        body = (await client.get("/api/v1/migration/export?table=messages")).json()
    assert body == {"table": "messages", "rows": [], "next": None}


async def test_a_broken_cursor_is_refused(client):
    with _enabled(True):
        response = await client.get("/api/v1/migration/export?table=messages&after=@@@")
    assert response.status_code == 400


def test_cursor_round_trip():
    when = datetime(2026, 9, 30, 12, 0, 0, 123456, tzinfo=timezone.utc)
    row_id = str(uuid.uuid4())
    assert _decode_cursor(_encode_cursor(when, row_id)) == (when, row_id)


def test_rows_turn_vectors_into_float_lists():
    class FakeVector:
        def tolist(self):
            return [0.5, 0.25]

    from app.db.models import UnansweredQuestion

    gap = UnansweredQuestion(
        id=uuid.uuid4(), course_id=1, user_id=2, question="q", chunks_retrieved=0
    )
    gap.embedding = FakeVector()  # type: ignore[assignment]
    assert _row(UnansweredQuestion, gap)["embedding"] == [0.5, 0.25]


async def test_flashcards_carry_the_activity_of_their_document(client, mock_db):
    from app.db.models import Flashcard

    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    doc_id = uuid.uuid4()
    cards = [
        Flashcard(
            id=uuid.uuid4(),
            course_id=1,
            content_hash="h1",
            question="q1",
            explanation="e1",
            source_document_id=str(doc_id).upper(),
            created_at=now,
        ),
        Flashcard(
            id=uuid.uuid4(),
            course_id=1,
            content_hash="h2",
            question="q2",
            explanation="e2",
            source_document_id="not-a-uuid",
            created_at=now,
        ),
        Flashcard(
            id=uuid.uuid4(),
            course_id=1,
            content_hash="h3",
            question="q3",
            explanation="e3",
            created_at=now,
        ),
    ]
    page = MagicMock()
    page.scalars.return_value.all.return_value = cards
    documents = MagicMock()
    documents.all.return_value = [(doc_id, 42)]
    mock_db.execute.side_effect = [page, documents]

    with _enabled(True):
        body = (await client.get("/api/v1/migration/export?table=flashcards")).json()

    assert [r["source_cmid"] for r in body["rows"]] == [42, None, None]
