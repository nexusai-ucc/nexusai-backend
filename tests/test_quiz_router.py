"""
Tests del quiz router — flashcards (SP-06) y dificultad (SP-08).

Misma estrategia de aislamiento que test_forums_router.py:
  - Mini FastAPI solo con el quiz router.
  - verify_hmac, get_db, get_llm_provider y get_embedding_provider reemplazados con mocks.
  - Sin llamadas reales a Postgres ni a APIs externas.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider
from app.quiz.router import ExamGenerateRequest, FocusTopic, QuizRequest, _build_quiz_prompt


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
            # SP-11 (#315): mismo mock genérico reusado por el upsert de
            # flashcards (_persist_flashcards) — content_hash no matchea el
            # real así que el id de la flashcard queda None, pero no rompe
            # el shape de la respuesta (los tests de flashcards no lo assertan).
            content_hash="dummy-hash",
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
# POST /suggest-difficulty (SP-12 / #322)
# ─────────────────────────────────────────────────────────────

def _attempt_row(**kwargs):
    defaults = dict(topic="derivadas", score=0.5, created_at=datetime.now(timezone.utc))
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


_SUGGEST_PAYLOAD = {"course_id": 1, "user_id": 1}


async def test_suggest_difficulty_no_history_returns_null(client, mock_db):
    mock_db.execute.return_value = _mock_quiz_result([])

    response = await client.post("/api/v1/quiz/suggest-difficulty", json=_SUGGEST_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["difficulty"] is None
    assert data["based_on_attempts"] == 0


async def test_suggest_difficulty_high_scores_suggest_hard(client, mock_db):
    mock_db.execute.return_value = _mock_quiz_result([
        _attempt_row(score=0.9), _attempt_row(score=1.0), _attempt_row(score=0.85),
    ])

    response = await client.post("/api/v1/quiz/suggest-difficulty", json=_SUGGEST_PAYLOAD)

    data = response.json()
    assert data["difficulty"] == "hard"
    assert data["based_on_attempts"] == 3
    assert "difícil" in data["reason"]


async def test_suggest_difficulty_low_scores_suggest_easy(client, mock_db):
    mock_db.execute.return_value = _mock_quiz_result([
        _attempt_row(score=0.2), _attempt_row(score=0.3),
    ])

    response = await client.post("/api/v1/quiz/suggest-difficulty", json=_SUGGEST_PAYLOAD)

    data = response.json()
    assert data["difficulty"] == "easy"


async def test_suggest_difficulty_mid_scores_suggest_medium(client, mock_db):
    mock_db.execute.return_value = _mock_quiz_result([_attempt_row(score=0.6)])

    response = await client.post("/api/v1/quiz/suggest-difficulty", json=_SUGGEST_PAYLOAD)

    assert response.json()["difficulty"] == "medium"


async def test_suggest_difficulty_filters_by_topic_case_insensitive(client, mock_db):
    """El mock no valida la query en sí — esto confirma que el payload con
    topic llega bien al endpoint y no rompe nada (el filtro real se prueba
    end-to-end contra una DB real, fuera del alcance de este entorno)."""
    mock_db.execute.return_value = _mock_quiz_result([_attempt_row(topic="Derivadas", score=0.9)])

    response = await client.post("/api/v1/quiz/suggest-difficulty", json={
        **_SUGGEST_PAYLOAD, "topic": "derivadas",
    })

    assert response.status_code == 200
    assert response.json()["difficulty"] == "hard"


# ─────────────────────────────────────────────────────────────
# POST /study-plan — combina QuizError + UnansweredQuestion
# ─────────────────────────────────────────────────────────────

def _quiz_error_row(**kwargs):
    defaults = dict(
        id=uuid4(),
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
        ids=[uuid4(), uuid4(), uuid4()],
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
# POST /study-plan — quiz_error_ids/gap_question_ids (SP-13 / #323)
# ─────────────────────────────────────────────────────────────

async def test_study_plan_topic_exposes_underlying_row_ids(client, mock_db, mock_llm):
    """El topic es texto del LLM, no una clave estable — el descarte (SP-13)
    opera sobre estos IDs reales, no sobre el texto."""
    quiz_row = _quiz_error_row()
    gap_row = _gap_row()
    mock_db.execute.side_effect = [
        _mock_quiz_result([quiz_row]),
        _mock_gap_result([gap_row]),
    ]
    llm_response = {
        "topics": [
            {"topic": "Derivadas", "quiz_groups": [0], "gap_groups": [0]},
        ]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))

    response = await client.post("/api/v1/quiz/study-plan", json=_STUDY_PLAN_PAYLOAD)

    data = response.json()
    topic = data["topics"][0]
    assert topic["quiz_error_ids"] == [str(quiz_row.id)]
    assert set(topic["gap_question_ids"]) == {str(i) for i in gap_row.ids}


# ─────────────────────────────────────────────────────────────
# POST /study-plan/dismiss (SP-13 / #323)
# ─────────────────────────────────────────────────────────────

_DISMISS_ID_1 = str(uuid4())
_DISMISS_ID_2 = str(uuid4())


async def test_study_plan_dismiss_updates_quiz_errors_only(client, mock_db):
    mock_db.execute.return_value = MagicMock(rowcount=2)

    response = await client.post("/api/v1/quiz/study-plan/dismiss", json={
        "course_id": 1,
        "user_id": 1,
        "quiz_error_ids": [_DISMISS_ID_1, _DISMISS_ID_2],
        "gap_question_ids": [],
    })

    assert response.status_code == 200
    assert response.json()["affected"] == 2
    # Un solo UPDATE (quiz_error_ids) — gap_question_ids vacío no dispara
    # una segunda query con un IN () vacío.
    mock_db.execute.assert_called_once()
    stmt_sql = str(mock_db.execute.call_args.args[0]).lower()
    assert "quiz_errors" in stmt_sql
    assert "dismissed_at" in stmt_sql


async def test_study_plan_dismiss_updates_both_tables(client, mock_db):
    mock_db.execute.return_value = MagicMock(rowcount=1)

    response = await client.post("/api/v1/quiz/study-plan/dismiss", json={
        "course_id": 1,
        "user_id": 1,
        "quiz_error_ids": [_DISMISS_ID_1],
        "gap_question_ids": [_DISMISS_ID_2],
    })

    assert response.status_code == 200
    assert response.json()["affected"] == 2  # 1 + 1
    assert mock_db.execute.call_count == 2
    second_stmt_sql = str(mock_db.execute.call_args_list[1].args[0]).lower()
    assert "unanswered_questions" in second_stmt_sql
    assert "student_dismissed_at" in second_stmt_sql


async def test_study_plan_dismiss_noop_without_ids(client, mock_db):
    response = await client.post("/api/v1/quiz/study-plan/dismiss", json={
        "course_id": 1,
        "user_id": 1,
        "quiz_error_ids": [],
        "gap_question_ids": [],
    })

    assert response.status_code == 200
    assert response.json()["affected"] == 0
    mock_db.execute.assert_not_called()


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
# focus_topics — temas con dificultad detectada (DOC-D09 / issue #390)
# ─────────────────────────────────────────────────────────────

def test_exam_request_defaults_to_no_focus_topics():
    req = ExamGenerateRequest(course_id=1, user_id=5, document_ids=[_VALID_DOC_ID])
    assert req.focus_topics == []


def test_exam_request_accepts_gap_and_faq_focus_topics():
    req = ExamGenerateRequest(
        course_id=1, user_id=5, document_ids=[_VALID_DOC_ID],
        focus_topics=[
            {"label": "derivadas de orden superior", "source": "gap"},
            {"label": "teorema de Bayes", "source": "faq"},
        ],
    )
    assert [t.source for t in req.focus_topics] == ["gap", "faq"]


def test_exam_request_rejects_invalid_focus_topic_source():
    with pytest.raises(ValidationError):
        ExamGenerateRequest(
            course_id=1, user_id=5, document_ids=[_VALID_DOC_ID],
            focus_topics=[{"label": "x", "source": "made_up"}],
        )


def test_exam_request_rejects_too_many_focus_topics():
    topics = [{"label": f"tema {i}", "source": "gap"} for i in range(16)]
    with pytest.raises(ValidationError):
        ExamGenerateRequest(course_id=1, user_id=5, document_ids=[_VALID_DOC_ID], focus_topics=topics)


def test_build_quiz_prompt_includes_focus_topics_block():
    messages = _build_quiz_prompt(
        _CHUNKS, num_questions=3, topic=None, question_type="multiple_choice",
        focus_topics=["derivadas de orden superior", "teorema de Bayes"],
    )
    user_msg = messages[1]["content"]
    assert "derivadas de orden superior" in user_msg
    assert "teorema de Bayes" in user_msg
    assert "source_topic" in user_msg


def test_build_quiz_prompt_omits_focus_topics_block_when_empty():
    messages = _build_quiz_prompt(_CHUNKS, num_questions=3, topic=None, question_type="multiple_choice")
    user_msg = messages[1]["content"]
    assert "TEMAS CON DIFICULTAD DETECTADA" not in user_msg


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


async def test_generate_exam_keeps_source_topic_matching_focus_topics(client, mock_llm):
    llm_response = {
        "questions": [{
            "question_type": "multiple_choice",
            "question": "¿Cuál es la derivada de x^2?",
            "options": ["2x", "x", "x^2", "0"],
            "correct_index": 0,
            "explanation": "2x",
            "source_filename": "apunte1.pdf",
            "source_topic": "derivadas de orden superior",
        }]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))
    payload = {
        "course_id": 1, "user_id": 9, "document_ids": [_VALID_DOC_ID], "num_questions": 1,
        "focus_topics": [{"label": "derivadas de orden superior", "source": "gap"}],
    }

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 200
    assert response.json()["questions"][0]["source_topic"] == "derivadas de orden superior"


async def test_generate_exam_discards_hallucinated_source_topic(client, mock_llm):
    """DOC-D09: si el LLM inventa un source_topic que no está en la lista
    provista, se descarta en vez de citar un tema falso."""
    llm_response = {
        "questions": [{
            "question_type": "multiple_choice",
            "question": "¿Cuál es la derivada de x^2?",
            "options": ["2x", "x", "x^2", "0"],
            "correct_index": 0,
            "explanation": "2x",
            "source_filename": "apunte1.pdf",
            "source_topic": "tema inventado por el LLM",
        }]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))
    payload = {
        "course_id": 1, "user_id": 9, "document_ids": [_VALID_DOC_ID], "num_questions": 1,
        "focus_topics": [{"label": "derivadas de orden superior", "source": "gap"}],
    }

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 200
    assert response.json()["questions"][0]["source_topic"] is None


async def test_generate_exam_without_focus_topics_never_sets_source_topic(client, mock_llm):
    llm_response = {
        "questions": [{
            "question_type": "multiple_choice",
            "question": "¿Cuál es la derivada de x^2?",
            "options": ["2x", "x", "x^2", "0"],
            "correct_index": 0,
            "explanation": "2x",
            "source_filename": "apunte1.pdf",
            "source_topic": "algo que el LLM puso igual",
        }]
    }
    mock_llm.chat_completion.return_value = MagicMock(text=json.dumps(llm_response))
    payload = {"course_id": 1, "user_id": 9, "document_ids": [_VALID_DOC_ID], "num_questions": 1}

    response = await client.post("/api/v1/quiz/generate-exam", json=payload)

    assert response.status_code == 200
    assert response.json()["questions"][0]["source_topic"] is None


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


# ─────────────────────────────────────────────────────────────
# POST /errors/list — paginación (UX-19 / issue #389)
# ─────────────────────────────────────────────────────────────

def _stored_quiz_error_row(**kwargs):
    defaults = dict(
        id="00000000-0000-0000-0000-000000000001",
        created_at=datetime.now(timezone.utc),
        question_type="multiple_choice",
        question="¿Cuál es la derivada de x^2?",
        explanation="2x",
        source_filename="apunte1.pdf",
        source_document_id=None,
        options=["2x", "x", "x^2", "0"],
        correct_index=0,
        user_selected_index=1,
        user_answer=None,
        ai_feedback=None,
        ai_score=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


_ERRORS_LIST_PAYLOAD = {"course_id": 1, "user_id": 1}


async def test_list_quiz_errors_total_reflects_real_count_not_page_size(client, mock_db):
    """Regresión UX-19: `total` debe ser el COUNT(*) real, no `len(items)`
    de la página devuelta — si no, el frontend nunca sabe que hay más."""
    mock_db.scalar.return_value = 37  # el alumno tiene 37 errores en total
    mock_db.execute.return_value = _mock_quiz_result([_stored_quiz_error_row(), _stored_quiz_error_row()])

    response = await client.post(
        "/api/v1/quiz/errors/list",
        json={**_ERRORS_LIST_PAYLOAD, "limit": 2, "offset": 0},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["total"] == 37


async def test_list_quiz_errors_defaults_offset_to_zero(client, mock_db):
    mock_db.scalar.return_value = 1
    mock_db.execute.return_value = _mock_quiz_result([_stored_quiz_error_row()])

    response = await client.post("/api/v1/quiz/errors/list", json=_ERRORS_LIST_PAYLOAD)

    assert response.status_code == 200


async def test_list_quiz_errors_rejects_negative_offset(client):
    response = await client.post(
        "/api/v1/quiz/errors/list",
        json={**_ERRORS_LIST_PAYLOAD, "offset": -1},
    )

    assert response.status_code == 422

    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# Repetición espaciada de flashcards — SP-11 (#315)
# ─────────────────────────────────────────────────────────────

from app.db.models import FlashcardReview  # noqa: E402
from app.quiz.router import _apply_sm2, _flashcard_content_hash  # noqa: E402


def _new_review(**kwargs):
    defaults = dict(ease_factor=2.5, interval_days=0, repetitions=0)
    defaults.update(kwargs)
    return FlashcardReview(**defaults)


def test_sm2_first_correct_review_sets_interval_to_one_day():
    review = _new_review()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_sm2(review, knew_it=True, now=now)

    assert review.repetitions == 1
    assert review.interval_days == 1
    assert review.next_review_at == now + timedelta(days=1)
    assert review.ease_factor > 2.5  # quality=5 sube el ease


def test_sm2_second_correct_review_sets_interval_to_six_days():
    review = _new_review(repetitions=1, interval_days=1)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_sm2(review, knew_it=True, now=now)

    assert review.repetitions == 2
    assert review.interval_days == 6


def test_sm2_third_correct_review_multiplies_interval_by_ease_factor():
    review = _new_review(repetitions=2, interval_days=6, ease_factor=2.5)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_sm2(review, knew_it=True, now=now)

    assert review.repetitions == 3
    assert review.interval_days == round(6 * 2.5)


def test_sm2_incorrect_review_resets_to_short_term():
    review = _new_review(repetitions=5, interval_days=40, ease_factor=2.8)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_sm2(review, knew_it=False, now=now)

    assert review.repetitions == 0
    assert review.interval_days == 1
    assert review.next_review_at == now + timedelta(days=1)
    assert review.ease_factor < 2.8  # quality=2 baja el ease


def test_sm2_ease_factor_never_drops_below_minimum():
    review = _new_review(ease_factor=1.3)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_sm2(review, knew_it=False, now=now)

    assert review.ease_factor == 1.3


def test_flashcard_content_hash_is_stable_and_order_sensitive():
    h1 = _flashcard_content_hash("¿Qué es una derivada?", "La tasa de cambio.")
    h2 = _flashcard_content_hash("¿Qué es una derivada?", "La tasa de cambio.")
    h3 = _flashcard_content_hash("otra pregunta", "otra respuesta")

    assert h1 == h2
    assert h1 != h3


async def test_flashcards_summary_returns_due_and_total_counts(client, mock_db):
    mock_db.scalar.side_effect = [7, 3]  # total_count, due_count

    response = await client.post(
        "/api/v1/quiz/flashcards/summary", json={"course_id": 1, "user_id": 1}
    )

    assert response.status_code == 200
    assert response.json() == {"due_count": 3, "total_count": 7}


async def test_flashcards_due_returns_questions_shaped_like_generate(client, mock_db):
    fc_id = uuid4()
    fc = SimpleNamespace(
        id=fc_id,
        question="Teorema de Bayes",
        explanation="Relaciona P(A|B) con P(B|A).",
        source_filename="apunte1.pdf",
        source_document_id=None,
    )
    due_result = MagicMock()
    due_result.all.return_value = [(fc, None)]
    mock_db.execute.return_value = due_result

    response = await client.post(
        "/api/v1/quiz/flashcards/due",
        json={"course_id": 1, "user_id": 1, "limit": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["questions"]) == 1
    q = data["questions"][0]
    assert q["id"] == str(fc_id)
    assert q["question_type"] == "flashcard"
    assert q["question"] == "Teorema de Bayes"
    assert q["options"] == []
    assert q["correct_index"] == -1


async def test_flashcards_review_batch_creates_new_review_and_applies_sm2(client, mock_db):
    flashcard_id = uuid4()

    valid_ids_result = MagicMock()
    valid_ids_result.all.return_value = [SimpleNamespace(id=flashcard_id)]

    review_lookup_result = MagicMock()
    review_lookup_result.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [valid_ids_result, review_lookup_result]

    response = await client.post(
        "/api/v1/quiz/flashcards/review-batch",
        json={
            "course_id": 1,
            "user_id": 1,
            "reviews": [{"flashcard_id": str(flashcard_id), "knew_it": True}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"updated": 1}
    mock_db.add.assert_called_once()
    created = mock_db.add.call_args[0][0]
    assert created.flashcard_id == flashcard_id
    assert created.user_id == 1
    assert created.repetitions == 1
    assert created.interval_days == 1


async def test_flashcards_review_batch_skips_flashcard_not_in_course(client, mock_db):
    """flashcard_id que no pertenece a este course_id se ignora — no confiamos
    ciegamente en un id mandado por el cliente."""
    flashcard_id = uuid4()
    other_course_flashcard_id = uuid4()

    valid_ids_result = MagicMock()
    valid_ids_result.all.return_value = [SimpleNamespace(id=other_course_flashcard_id)]
    mock_db.execute.side_effect = [valid_ids_result]

    response = await client.post(
        "/api/v1/quiz/flashcards/review-batch",
        json={
            "course_id": 1,
            "user_id": 1,
            "reviews": [{"flashcard_id": str(flashcard_id), "knew_it": True}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"updated": 0}
    mock_db.add.assert_not_called()
