"""The routers must ask the LLM for English when their source text is English.

The prompt templates are written in Spanish, so without an explicit directive
every AI feature answered in Spanish for English courses. These tests check the
wiring at each call site; the detection itself is covered in test_language.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.providers.llm import LLMProvider, get_llm_provider
from app.shared.language import language_directive

EN_DIRECTIVE = language_directive("en")

EN_MATERIAL = (
    "A primary key uniquely identifies each row of a table and cannot be NULL. "
    "A foreign key references the primary key of another table, and it is used to "
    "enforce referential integrity between the two tables."
)
ES_MATERIAL = (
    "Una clave primaria identifica de forma única cada fila de una tabla y no puede "
    "ser nula. Una clave foránea hace referencia a la clave primaria de otra tabla."
)


@pytest.fixture
def mock_llm():
    return AsyncMock(spec=LLMProvider)


@pytest.fixture
async def forums_client(mock_llm):
    from app.forums.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/forums")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _summary_payload(*contents: str) -> dict:
    return {
        "discussion_id": 3,
        "course_id": 2,
        "posts": [
            {"post_id": i + 1, "author": f"User {i}", "content": text}
            for i, text in enumerate(contents)
        ],
    }


async def test_thread_summary_asks_for_english_for_english_posts(
    forums_client, mock_llm
):
    mock_llm.chat_completion.return_value = MagicMock(
        text='{"summary": "s", "key_points": [], "resolved": true}'
    )

    response = await forums_client.post(
        "/api/v1/forums/summarize-thread",
        json=_summary_payload(
            "I read that two transactions can deadlock if they lock tables in a different order.",
            "Transaction A locks table X then wants Y, and B locks Y then wants X.",
        ),
    )

    assert response.status_code == 200
    messages = mock_llm.chat_completion.await_args.args[0]
    assert messages[-1]["content"].endswith(EN_DIRECTIVE)


async def test_thread_summary_leaves_spanish_posts_alone(forums_client, mock_llm):
    mock_llm.chat_completion.return_value = MagicMock(
        text='{"summary": "s", "key_points": [], "resolved": true}'
    )

    await forums_client.post(
        "/api/v1/forums/summarize-thread",
        json=_summary_payload(
            "Leí que dos transacciones pueden bloquearse si toman las tablas en otro orden.",
            "La transacción A bloquea la tabla X y luego quiere la Y, y la B al revés.",
        ),
    )

    messages = mock_llm.chat_completion.await_args.args[0]
    assert "IMPORTANT: write your entire answer" not in messages[-1]["content"]


async def test_quiz_generation_asks_for_english_for_english_material(mock_llm):
    from app.quiz.router import _run_quiz_generation

    mock_llm.chat_completion.side_effect = RuntimeError("stop after the call")

    with pytest.raises(Exception):  # noqa: B017 - only the prompt matters here
        await _run_quiz_generation(
            [("unit1.pdf", EN_MATERIAL)],
            course_id=2,
            topic=None,
            num_questions=2,
            question_type="multiple_choice",
            difficulty="medium",
            db=AsyncMock(),
            llm=mock_llm,
        )

    messages = mock_llm.chat_completion.await_args.args[0]
    assert messages[-1]["content"].endswith(EN_DIRECTIVE)


async def test_quiz_generation_leaves_spanish_material_alone(mock_llm):
    from app.quiz.router import _run_quiz_generation

    mock_llm.chat_completion.side_effect = RuntimeError("stop after the call")

    with pytest.raises(Exception):  # noqa: B017 - only the prompt matters here
        await _run_quiz_generation(
            [("unidad1.pdf", ES_MATERIAL)],
            course_id=2,
            topic=None,
            num_questions=2,
            question_type="multiple_choice",
            difficulty="medium",
            db=AsyncMock(),
            llm=mock_llm,
        )

    messages = mock_llm.chat_completion.await_args.args[0]
    assert "IMPORTANT: write your entire answer" not in messages[-1]["content"]


def test_summary_cache_key_changes_with_the_prompt_version():
    """Old Spanish summaries of English documents must not be served after the fix."""
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.documents import summarizer

    document = SimpleNamespace(
        id="doc-1", file_hash="abc", updated_at=datetime.now(timezone.utc)
    )

    key = summarizer._cache_key(document, "model-x")

    assert summarizer._PROMPT_VERSION in key
