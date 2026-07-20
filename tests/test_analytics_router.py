"""
Tests del dashboard de FAQ — DOC-D02.

Router sin cobertura previa (DOC-D01 tampoco tenía tests). El agrupamiento
por texto normalizado se resuelve en SQL (no testeable sin DB real), así que
lo que se cubre acá es la lógica de Python alrededor: el corte temprano sin
interacciones, el parseo/validación de la respuesta del LLM, y que el conteo
por tema SIEMPRE se recalcula a partir de los índices asignados — nunca se
confía en el LLM para los números. Mismo patrón de aislamiento que
test_search_router.py / test_forums_router.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.llm import LLMProvider, get_llm_provider


def _row(question: str, count: int):
    return SimpleNamespace(question=question, count=count)


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.all.return_value = [
        _row("¿qué temas entran en el parcial?", 6),
        _row("¿cómo se calcula el determinante de una matriz 4x4?", 5),
        _row("¿qué es la regla de la cadena?", 3),
    ]
    return db


@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMProvider)


@pytest.fixture
async def client(mock_db, mock_llm):
    from app.analytics.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/analytics")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


_BASE_PAYLOAD = {"course_id": 1, "days": 30}


def _llm_response(payload: dict) -> MagicMock:
    return MagicMock(text=json.dumps(payload))


async def test_no_interactions_returns_empty_without_llm_call(client, mock_db, mock_llm):
    mock_db.execute.return_value.all.return_value = []

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data == {"course_id": 1, "days": 30, "total_questions": 0, "topics": []}
    mock_llm.chat_completion.assert_not_called()


async def test_topics_recompute_count_from_assigned_indices(client, mock_llm):
    mock_llm.chat_completion.return_value = _llm_response({
        "topics": [
            {"label": "Fechas del parcial", "question_indices": [0]},
            {"label": "Álgebra lineal", "question_indices": [1, 2]},
        ]
    })

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["total_questions"] == 14  # 6 + 5 + 3, calculado en Python, no por el LLM
    topics = {t["topic"]: t for t in data["topics"]}
    assert topics["Fechas del parcial"]["count"] == 6
    assert topics["Álgebra lineal"]["count"] == 8  # 5 + 3, no lo que diga el LLM


async def test_topics_sorted_by_recomputed_count_desc(client, mock_llm):
    # El LLM devuelve el grupo chico primero — la respuesta debe reordenar igual.
    mock_llm.chat_completion.return_value = _llm_response({
        "topics": [
            {"label": "Chico", "question_indices": [2]},
            {"label": "Grande", "question_indices": [0, 1]},
        ]
    })

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    data = response.json()
    assert [t["topic"] for t in data["topics"]] == ["Grande", "Chico"]


async def test_invalid_indices_are_dropped_and_topic_skipped_if_empty(client, mock_llm):
    mock_llm.chat_completion.return_value = _llm_response({
        "topics": [
            {"label": "Fuera de rango", "question_indices": [99]},
            {"label": "Válido", "question_indices": [0, 99]},
        ]
    })

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    data = response.json()
    topics = {t["topic"]: t for t in data["topics"]}
    assert "Fuera de rango" not in topics
    assert topics["Válido"]["count"] == 6


async def test_malformed_llm_json_returns_503(client, mock_llm):
    mock_llm.chat_completion.return_value = MagicMock(text="esto no es JSON")

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    assert response.status_code == 503


async def test_llm_call_failure_returns_503(client, mock_llm):
    mock_llm.chat_completion.side_effect = RuntimeError("timeout")

    response = await client.post("/api/v1/analytics/faq-topics", json=_BASE_PAYLOAD)

    assert response.status_code == 503
