"""
Tests del quiz router — flashcards (SP-06) y dificultad (SP-08).

Misma estrategia de aislamiento que test_forums_router.py:
  - Mini FastAPI solo con el quiz router.
  - verify_hmac, get_db, get_llm_provider y get_embedding_provider reemplazados con mocks.
  - Sin llamadas reales a Postgres ni a APIs externas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider
from app.quiz.router import ExamGenerateRequest, QuizRequest, _build_quiz_prompt


# ─────────────────────────────────────────────────────────────
# Validación de QuizRequest
# ─────────────────────────────────────────────────────────────

def test_quiz_request_accepts_flashcard_type():
    req = QuizRequest(course_id=1, user_id=1, question_type="flashcard")
    assert req.question_type == "flashcard"
    assert req.difficulty == "medium"  # default


def test_quiz_request_rejects_invalid_question_type():
    with pytest.raises(ValidationError):
        QuizRequest(course_id=1, user_id=1, question_type="not_a_real_type")


def test_quiz_request_accepts_valid_difficulty():
    req = QuizRequest(course_id=1, user_id=1, difficulty="hard")
    assert req.difficulty == "hard"


def test_quiz_request_rejects_invalid_difficulty():
    with pytest.raises(ValidationError):
        QuizRequest(course_id=1, user_id=1, difficulty="impossible")


# ─────────────────────────────────────────────────────────────
# _build_quiz_prompt — lógica pura, sin DB ni LLM
# ─────────────────────────────────────────────────────────────

_CHUNKS = [("apunte1.pdf", "El teorema de Bayes relaciona probabilidades condicionales.")]


def test_build_quiz_prompt_flashcard_schema():
    messages = _build_quiz_prompt(_CHUNKS, num_questions=3, topic=None, question_type="flashcard")
    user_msg = messages[1]["content"]
    assert '"question_type": "flashcard"' in user_msg
    assert '"correct_index": -1' in user_msg
    assert "flashcards" in user_msg.lower()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_build_quiz_prompt_includes_difficulty_instruction(difficulty):
    messages = _build_quiz_prompt(
        _CHUNKS, num_questions=3, topic=None, question_type="multiple_choice", difficulty=difficulty
    )
    system_msg = messages[0]["content"]
    assert difficulty.upper() in system_msg or {
        "easy": "FÁCIL",
        "medium": "MEDIA",
        "hard": "DIFÍCIL",
    }[difficulty] in system_msg


def test_build_quiz_prompt_defaults_to_medium_difficulty():
    with_default = _build_quiz_prompt(_CHUNKS, num_questions=3, topic=None, question_type="open")
    with_explicit = _build_quiz_prompt(_CHUNKS, num_questions=3, topic=None, question_type="open", difficulty="medium")
    assert with_default[0]["content"] == with_explicit[0]["content"]


# ─────────────────────────────────────────────────────────────
# Fixtures — endpoint end-to-end con dependencias mockeadas
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    # Sirve tanto para el sampling de chunks (filename/content) como para el
    # enriquecimiento posterior de source_document_id (filename/id, SP-10).
    db.execute.return_value.all.return_value = [
        SimpleNamespace(
            filename="apunte1.pdf",
            content="El teorema de Bayes relaciona probabilidades condicionales.",
            id="00000000-0000-0000-0000-000000000001",
        )
    ]
    return db


@pytest.fixture
def mock_embeddings():
    return AsyncMock(spec=EmbeddingProvider)


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMProvider)
    flashcard_response = {
        "questions": [
            {
                "question_type": "flashcard",
                "question": "Teorema de Bayes",
                "options": [],
                "correct_index": -1,
                "explanation": "Relaciona la probabilidad condicional de A dado B con la de B dado A.",
                "source_filename": "apunte1.pdf",
            }
        ]
    }
    llm.chat_completion.return_value = MagicMock(text=json.dumps(flashcard_response))
    return llm


@pytest.fixture
async def client(mock_db, mock_embeddings, mock_llm):
    from app.quiz.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/quiz")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ─────────────────────────────────────────────────────────────
# POST /generate — flashcards + dificultad
# ─────────────────────────────────────────────────────────────

async def test_generate_flashcards_returns_200_with_expected_shape(client):
    payload = {
        "course_id": 1,
        "user_id": 1,
        "num_questions": 1,
        "question_type": "flashcard",
        "difficulty": "hard",
    }

    response = await client.post("/api/v1/quiz/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["questions"]) == 1
    q = data["questions"][0]
    assert q["question_type"] == "flashcard"
    assert q["options"] == []
    assert q["correct_index"] == -1


async def test_generate_rejects_invalid_difficulty_at_http_level(client):
    payload = {"course_id": 1, "user_id": 1, "difficulty": "impossible"}

    response = await client.post("/api/v1/quiz/generate", json=payload)

    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# POST /study-plan — combina QuizError + UnansweredQuestion
# ─────────────────────────────────────────────────────────────

def _quiz_error_row(**kwargs):
    defaults = dict(
        source_filename="apunte1.pdf",
        question="¿Cuál es la derivada de x^2?",
        explanation="2x",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _gap_row(**kwargs):
    defaults = dict(
        question="que es una integral impropia",
        count=3,
        last_asked_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mock_quiz_result(rows):
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    return result


def _mock_gap_result(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


_STUDY_PLAN_PAYLOAD = {"course_id": 1, "user_id": 1}


async def test_study_plan_skips_llm_when_no_signal(client, mock_db, mock_llm):
    mock_db.execute.side_effect = [_mock_quiz_result([]), _mock_gap_result([])]

    response = await client.post("/api/v1/quiz/study-plan", json=_STUDY_PLAN_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["topics"] == []
    mock_llm.chat_completion.assert_not_called()


async def test_study_plan_combines_quiz_errors_and_gaps(client, mock_db, mock_llm):
    mock_db.execute.side_effect = [
        _mock_quiz_result([_quiz_error_row(), _quiz_error_row()]),
        _mock_gap_result([_gap_row()]),
    ]
    llm_response = {
        "topics": [
            {
                "topic": "Derivadas",
                "quiz_groups": [0],
                "gap_groups": [],
                "reason": "Fallaste 2 preguntas sobre derivadas.",
                "suggested_quiz_topic": "derivadas",
            },
            {
                "topic": "Integrales impropias",
                "quiz_groups": [],
                "gap_groups": [0],
                "reason": "Preguntaste esto 3 veces sin buena respuesta.",
                "suggested_quiz_topic": "integrales impropias",
            },
        ]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))

    response = await client.post("/api/v1/quiz/study-plan", json=_STUDY_PLAN_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert len(data["topics"]) == 2
    # Orden por (quiz_error_count + gap_count) desc: "Integrales impropias" (3)
    # queda antes que "Derivadas" (2), aunque el LLM las haya listado al revés.
    assert data["topics"][0]["topic"] == "Integrales impropias"
    assert data["topics"][0]["gap_count"] == 3
    assert data["topics"][1]["topic"] == "Derivadas"
    assert data["topics"][1]["quiz_error_count"] == 2
    assert data["topics"][1]["gap_count"] == 0


async def test_study_plan_never_trusts_llm_counts(client, mock_db, mock_llm):
    """El LLM podría devolver counts propios — Python siempre los recalcula."""
    mock_db.execute.side_effect = [
        _mock_quiz_result([_quiz_error_row()]),
        _mock_gap_result([]),
    ]
    llm_response = {
        "topics": [
            {
                "topic": "Derivadas",
                "quiz_groups": [0],
                "gap_groups": [],
                "quiz_error_count": 999,
                "gap_count": 999,
            }
        ]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))

    response = await client.post("/api/v1/quiz/study-plan", json=_STUDY_PLAN_PAYLOAD)

    data = response.json()
    assert data["topics"][0]["quiz_error_count"] == 1
    assert data["topics"][0]["gap_count"] == 0


async def test_study_plan_ignores_out_of_range_group_indices(client, mock_db, mock_llm):
    mock_db.execute.side_effect = [
        _mock_quiz_result([_quiz_error_row()]),
        _mock_gap_result([]),
    ]
    llm_response = {
        "topics": [
            {"topic": "Tema inventado", "quiz_groups": [99], "gap_groups": [5]},
            {"topic": "Derivadas", "quiz_groups": [0], "gap_groups": []},
        ]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))

    response = await client.post("/api/v1/quiz/study-plan", json=_STUDY_PLAN_PAYLOAD)

    data = response.json()
    # "Tema inventado" no cita ningún grupo válido -> queda afuera.
    assert len(data["topics"]) == 1
    assert data["topics"][0]["topic"] == "Derivadas"


# ─────────────────────────────────────────────────────────────
# ExamGenerateRequest — validación (EVAL-01 / issue #235)
# ─────────────────────────────────────────────────────────────

_VALID_DOC_ID = "00000000-0000-0000-0000-000000000001"


def test_exam_request_accepts_valid_payload():
    req = ExamGenerateRequest(
        course_id=1, user_id=5, document_ids=[_VALID_DOC_ID], question_type="true_false"
    )
    assert req.document_ids == [_VALID_DOC_ID]
    assert req.question_type == "true_false"
    assert req.num_questions == 10  # default


def test_exam_request_rejects_empty_document_ids():
    with pytest.raises(ValidationError):
        ExamGenerateRequest(course_id=1, user_id=5, document_ids=[])


def test_exam_request_rejects_non_uuid_document_id():
    with pytest.raises(ValidationError):
        ExamGenerateRequest(course_id=1, user_id=5, document_ids=["not-a-uuid"])


def test_exam_request_rejects_flashcard_type():
    # flashcard es válido para el quiz de alumno pero no tiene sentido en un examen.
    with pytest.raises(ValidationError):
        ExamGenerateRequest(course_id=1, user_id=5, document_ids=[_VALID_DOC_ID], question_type="flashcard")


# ─────────────────────────────────────────────────────────────
# POST /generate-exam — endpoint end-to-end
# ─────────────────────────────────────────────────────────────

async def test_generate_exam_returns_200_with_expected_shape(client):
    payload = {
        "course_id": 1,
        "user_id": 9,
        "document_ids": [_VALID_DOC_ID],
        "num_questions": 1,
        "question_type": "multiple_choice",
    }

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["course_id"] == 1
    assert len(data["questions"]) == 1


async def test_generate_exam_rejects_empty_document_ids_at_http_level(client):
    payload = {"course_id": 1, "user_id": 9, "document_ids": []}

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 422


async def test_generate_exam_404_when_no_chunks_for_selected_documents(client, mock_db):
    mock_db.execute.return_value.all.return_value = []
    payload = {"course_id": 1, "user_id": 9, "document_ids": [_VALID_DOC_ID]}

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 404


# ─────────────────────────────────────────────────────────────
# POST /attempts — ANALYTICS-01, fuente de datos para el histograma
# de quiz scores del dashboard docente.
# ─────────────────────────────────────────────────────────────

async def test_record_attempt_recomputes_score_from_answers(client, mock_db):
    # db.add() es sync en SQLAlchemy AsyncSession — MagicMock evita coroutine warning.
    mock_db.add = MagicMock()
    payload = {"course_id": 1, "user_id": 9, "total_questions": 10, "correct_answers": 7}

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["score"] == 0.7
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


async def test_record_attempt_ignores_score_sent_by_client(client, mock_db):
    """El score nunca viene del cliente — solo total_questions/correct_answers."""
    mock_db.add = MagicMock()
    payload = {
        "course_id": 1,
        "user_id": 9,
        "total_questions": 4,
        "correct_answers": 1,
        "score": 1.0,  # campo extra, ignorado por el schema
    }

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 200
    assert response.json()["score"] == 0.25


async def test_record_attempt_perfect_score(client, mock_db):
    mock_db.add = MagicMock()
    payload = {"course_id": 1, "user_id": 9, "total_questions": 5, "correct_answers": 5}

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 200
    assert response.json()["score"] == 1.0


async def test_record_attempt_rejects_correct_greater_than_total(client):
    payload = {"course_id": 1, "user_id": 9, "total_questions": 3, "correct_answers": 4}

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 422


async def test_record_attempt_rejects_zero_total_questions(client):
    payload = {"course_id": 1, "user_id": 9, "total_questions": 0, "correct_answers": 0}

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 422


async def test_record_attempt_rejects_negative_course_id(client):
    payload = {"course_id": -1, "user_id": 9, "total_questions": 5, "correct_answers": 3}

    response = await client.post("/api/v1/quiz/attempts", json=payload)

    assert response.status_code == 422
