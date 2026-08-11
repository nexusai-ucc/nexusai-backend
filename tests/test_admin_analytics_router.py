"""
Tests del dashboard docente — GET /api/v1/admin/analytics (ANALYTICS-01).

Mismo patrón de aislamiento que test_analytics_router.py / test_forums_router.py:
FastAPI mínimo con solo este router, verify_hmac y get_db reemplazados por
mocks, sin conexión real a Postgres. Cada query interna (top_queries, daily
breakdown, quiz score distribution, gaps ratio) se testea vía el orden de
llamadas a `db.execute`, siguiendo el mismo `side_effect` en secuencia que
usa test_study_plan en test_quiz_router.py para combinar múltiples queries.

daily_message_counts y quiz_score_distribution agrupan en SQL (UX-16, #386)
en vez de traer filas crudas — los mocks devuelven directamente el shape ya
agregado que devolvería Postgres (una fila por día, una única fila de
agregados para los buckets de quiz), no filas individuales.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db


def _question_row(question: str, count: int):
    return SimpleNamespace(question=question, count=count)


def _daily_group_row(day: datetime, count: int):
    return SimpleNamespace(day=day, count=count)


def _quiz_buckets_row(total: int, avg_score, bucket_counts: list[int]):
    fields = {"total": total, "avg_score": avg_score}
    for i, c in enumerate(bucket_counts):
        fields[f"bucket_{i}"] = c
    return SimpleNamespace(**fields)


def _scalar_result(value: int):
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _one_result(row):
    result = MagicMock()
    result.one.return_value = row
    return result


@pytest.fixture
def mock_db():
    """execute() se llama 4 veces en orden fijo dentro del endpoint:

    1) top_queries (get_top_questions)      -> .all()
    2) daily_message_counts (agrupado)      -> .all()
    3) quiz score distribution (agregado)   -> .one()
    4) gaps_ratio: gaps_detected             -> .scalar_one()
    5) gaps_ratio: questions_answered        -> .scalar_one()
    """
    db = AsyncMock()
    top_queries_result = MagicMock()
    top_queries_result.all.return_value = [_question_row("¿qué es una integral?", 5)]

    daily_result = MagicMock()
    daily_result.all.return_value = [
        _daily_group_row(datetime(2026, 7, 1, tzinfo=timezone.utc), 2),
        _daily_group_row(datetime(2026, 7, 2, tzinfo=timezone.utc), 1),
    ]

    # scores 1.0, 0.5, 0.1 -> buckets [0-20]=1 (0.1), [40-60]=1 (0.5), [80-100]=1 (1.0)
    quiz_result = _one_result(
        _quiz_buckets_row(total=3, avg_score=(1.0 + 0.5 + 0.1) / 3, bucket_counts=[1, 0, 1, 0, 1])
    )

    db.execute.side_effect = [
        top_queries_result,
        daily_result,
        quiz_result,
        _scalar_result(2),   # gaps_detected
        _scalar_result(10),  # questions_answered
    ]
    return db


@pytest.fixture
async def client(mock_db):
    from app.admin.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/admin")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_analytics_returns_expected_shape(client):
    response = await client.get("/api/v1/admin/analytics?course_id=1")

    assert response.status_code == 200
    data = response.json()
    assert data["course_id"] == 1
    assert data["period_days"] == 30
    assert data["top_queries"] == [{"question": "¿qué es una integral?", "count": 5}]
    assert data["daily_message_counts"] == [
        {"date": "2026-07-01", "message_count": 2},
        {"date": "2026-07-02", "message_count": 1},
    ]


async def test_quiz_score_distribution_buckets_and_average(client):
    response = await client.get("/api/v1/admin/analytics?course_id=1")

    data = response.json()["quiz_score_distribution"]
    assert data["total_attempts"] == 3
    assert data["average_score"] == round((1.0 + 0.5 + 0.1) / 3, 4)
    buckets = {b["range"]: b["count"] for b in data["buckets"]}
    assert buckets["80-100"] == 1  # score 1.0
    assert buckets["40-60"] == 1   # score 0.5
    assert buckets["0-20"] == 1    # score 0.1
    assert buckets["20-40"] == 0
    assert buckets["60-80"] == 0


async def test_gaps_ratio_computed_from_counts(client):
    response = await client.get("/api/v1/admin/analytics?course_id=1")

    data = response.json()["gaps_ratio"]
    assert data == {"gaps_detected": 2, "questions_answered": 10, "ratio": 0.2}


async def test_gaps_ratio_is_zero_when_no_questions_answered(mock_db):
    db = mock_db
    db.execute.side_effect = [
        MagicMock(all=MagicMock(return_value=[])),
        MagicMock(all=MagicMock(return_value=[])),
        _one_result(_quiz_buckets_row(total=0, avg_score=None, bucket_counts=[0, 0, 0, 0, 0])),
        _scalar_result(0),
        _scalar_result(0),
    ]

    from app.admin.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/admin")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/api/v1/admin/analytics?course_id=1")

    data = response.json()["gaps_ratio"]
    assert data == {"gaps_detected": 0, "questions_answered": 0, "ratio": 0.0}


async def test_analytics_rejects_invalid_course_id(client):
    response = await client.get("/api/v1/admin/analytics?course_id=0")

    assert response.status_code == 422


async def test_analytics_rejects_days_out_of_range(client):
    response = await client.get("/api/v1/admin/analytics?course_id=1&days=400")

    assert response.status_code == 422


async def test_analytics_accepts_custom_days(client):
    response = await client.get("/api/v1/admin/analytics?course_id=1&days=7")

    assert response.status_code == 200
    assert response.json()["period_days"] == 7
