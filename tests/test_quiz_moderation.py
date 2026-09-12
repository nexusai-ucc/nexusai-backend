"""
Tests de moderación de contenido integrada en POST /quiz/evaluate.

Misma estrategia de aislamiento que test_quiz_router.py: mini FastAPI solo
con el quiz router, verify_hmac/get_llm_provider reemplazados por mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.providers.llm import LLMProvider, get_llm_provider

_EVALUATE_PAYLOAD = {
    "course_id": 1,
    "user_id": 7,
    "question": "¿Qué es el teorema de Bayes?",
    "model_answer": "Relaciona la probabilidad condicional de A dado B con la de B dado A.",
    "user_answer": "Es una fórmula de probabilidad condicional.",
}


@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMProvider)


@pytest.fixture
async def client(mock_llm):
    from app.quiz.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/quiz")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _fake_settings(*, moderation_api_key=None, moderation_fail_open=True):
    return SimpleNamespace(
        moderation_enabled=True,
        moderation_api_key=moderation_api_key,
        moderation_fail_open=moderation_fail_open,
    )


async def test_evaluate_blocks_inappropriate_user_answer(client, mock_llm):
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.return_value = MagicMock(
            text='{"flagged": true, "categories": ["harassment"]}'
        )

        response = await client.post("/api/v1/quiz/evaluate", json=_EVALUATE_PAYLOAD)

    assert response.status_code == 400
    assert "no cumple" in response.json()["detail"].lower()
    # Solo se llamó al LLM para clasificar — nunca se gastó en el prompt de
    # evaluación real (el objetivo explícito de moderar antes del LLM principal).
    assert mock_llm.chat_completion.await_count == 1


async def test_evaluate_allows_acceptable_answer(client, mock_llm):
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.side_effect = [
            MagicMock(text='{"flagged": false, "categories": []}'),
            MagicMock(text='{"correct": true, "score": 0.9, "feedback": "Buena respuesta."}'),
        ]

        response = await client.post("/api/v1/quiz/evaluate", json=_EVALUATE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["correct"] is True
    assert body["score"] == 0.9


async def test_evaluate_moderation_failure_does_not_return_500(client, mock_llm):
    """Fail-safe: si la clasificación de moderación misma explota, el
    endpoint no debe romperse con un 500 — sigue evaluando la respuesta
    (fail-open, el default) en vez de tumbar el flujo del alumno."""
    with patch("app.shared.moderation.get_settings", return_value=_fake_settings()):
        mock_llm.chat_completion.side_effect = [
            Exception("proveedor caído durante la clasificación"),
            MagicMock(text='{"correct": true, "score": 0.7, "feedback": "Aceptable."}'),
        ]

        response = await client.post("/api/v1/quiz/evaluate", json=_EVALUATE_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["correct"] is True


async def test_evaluate_moderation_fail_closed_blocks_instead_of_500(client, mock_llm):
    """Con MODERATION_FAIL_OPEN=false, la misma falla de moderación bloquea
    con un 400 claro en vez de dejar pasar contenido no verificado o romper
    con un 500."""
    with patch(
        "app.shared.moderation.get_settings",
        return_value=_fake_settings(moderation_fail_open=False),
    ):
        mock_llm.chat_completion.side_effect = Exception("proveedor caído")

        response = await client.post("/api/v1/quiz/evaluate", json=_EVALUATE_PAYLOAD)

    assert response.status_code == 400
    assert response.json()["detail"]
