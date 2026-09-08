"""
Quiz Generator — Feature F.

POST /api/v1/quiz/generate
  Genera un quiz de práctica a partir del material indexado del curso.
  Soporta múltiples tipos de pregunta: opción múltiple, verdadero/falso,
  preguntas abiertas, flashcards (SP-06) y mix. El LLM produce JSON
  estructurado. Acepta un nivel de dificultad (easy|medium|hard, SP-08)
  que ajusta la complejidad de las preguntas generadas.

  Modos:
  - topic provisto → retrieve_context con esa consulta (chunks relevantes)
  - topic vacío    → chunks aleatorios del curso (variedad de temas)

POST /api/v1/quiz/generate-exam
  Genera un banco de preguntas de EXAMEN para el docente (EVAL-01, issue #235).
  El docente elige explícitamente document_ids del curso en vez de un topic
  libre o sampling aleatorio. Devuelve el mismo shape que /generate — el
  export a formato GIFT (Moodle) se hace del lado del frontend.

POST /api/v1/quiz/evaluate
  Evalúa la respuesta libre de un alumno a una pregunta abierta usando LLM.
  Devuelve { correct, score, feedback }.

Uso del LLM:
  - `response_format={"type":"json_object"}` para que el provider fuerce
    JSON parseable (Gemini OpenAI-compat lo soporta).
  - System prompt explícito con el schema esperado y reglas de calidad.
  - Si el parse falla → 503 con mensaje claro (no inventar quiz fake).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import (
    ChatSession,
    Chunk,
    Document,
    Flashcard,
    FlashcardReview,
    Message,
    QuizAttempt,
    QuizError,
    UnansweredQuestion,
)
from app.db.session import get_db
from app.documents.retriever import retrieve_context
from app.gaps.recorder import WEAK_MATCH_THRESHOLD
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider
from app.shared.config import get_settings

logger = logging.getLogger("nexusai.quiz")

# Minimum cosine similarity for a chunk to count as covering a topic.
# Higher than WEAK_MATCH_THRESHOLD (0.4) to reject spurious cross-lingual matches.
QUIZ_TOPIC_MIN_SIMILARITY = 0.5

_VALID_QUESTION_TYPES = {"multiple_choice", "true_false", "open", "mix", "flashcard", "fill_blank"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class QuizRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    topic: Optional[str] = Field(default=None, max_length=200)
    num_questions: int = Field(default=5, ge=1, le=10)
    question_type: str = Field(default="multiple_choice")
    difficulty: str = Field(default="medium")

    @field_validator("question_type")
    @classmethod
    def check_question_type(cls, v: str) -> str:
        if v not in _VALID_QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {_VALID_QUESTION_TYPES}")
        return v

    @field_validator("difficulty")
    @classmethod
    def check_difficulty(cls, v: str) -> str:
        if v not in _VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {_VALID_DIFFICULTIES}")
        return v


class QuizQuestion(BaseModel):
    # SP-11 (#315): id real de `flashcards`, solo poblado cuando
    # question_type='flashcard' (persistidas para poder aplicar repetición
    # espaciada). None para el resto de los tipos, que siguen siendo efímeros.
    id: Optional[str] = Field(default=None)
    question_type: str = Field(default="multiple_choice")
    question: str = Field(min_length=1, max_length=500)
    options: List[str] = Field(default=[])   # 4 for MC, 2 for T/F, [] for open
    correct_index: int = Field(default=-1, ge=-1, le=3)  # -1 for open questions
    explanation: str = Field(min_length=1, max_length=1500)
    source_filename: str = Field(default="")
    source_document_id: Optional[str] = Field(default=None)  # filled after generation; Document.id is a UUID
    # DOC-D09 (#390): tema débil (Gap/FAQ) del que salió esta pregunta, si
    # el docente pidió incluir esos temas — texto exacto de FocusTopic.label,
    # o None si la pregunta no cubre ninguno de los temas provistos.
    source_topic: Optional[str] = Field(default=None, max_length=200)


class QuizResponse(BaseModel):
    course_id: int
    topic: Optional[str]
    questions: List[QuizQuestion]


_VALID_EXAM_QUESTION_TYPES = {"multiple_choice", "true_false", "open", "mix"}
_VALID_FOCUS_TOPIC_SOURCES = {"gap", "faq"}


class FocusTopic(BaseModel):
    """Tema débil detectado (Gap sin responder o tópico de FAQ) que el
    docente eligió incluir como contexto extra al generar el examen —
    DOC-D09 (#390)."""
    label: str = Field(min_length=1, max_length=200)
    source: str = Field(default="gap")

    @field_validator("source")
    @classmethod
    def check_source(cls, v: str) -> str:
        if v not in _VALID_FOCUS_TOPIC_SOURCES:
            raise ValueError(f"source must be one of {_VALID_FOCUS_TOPIC_SOURCES}")
        return v


class ExamGenerateRequest(BaseModel):
    """Generador de exámenes para docentes — EVAL-01 / issue #235 (DOC-D04).

    A diferencia de QuizRequest (alumno, material aleatorio o por topic libre),
    el docente elige explícitamente de qué archivos del curso quiere sacar las
    preguntas.
    """
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)  # docente — $USER->id real, validado por la capability del lado Moodle
    document_ids: List[str] = Field(min_length=1, max_length=20)
    topic: Optional[str] = Field(default=None, max_length=200)
    num_questions: int = Field(default=10, ge=1, le=20)
    question_type: str = Field(default="multiple_choice")
    difficulty: str = Field(default="medium")
    # DOC-D09 (#390): temas con dificultad detectada (Gaps sin responder /
    # FAQ agrupada) que el docente eligió priorizar. El caller (Moodle) es
    # responsable de no mandar temas de gaps archivados — ver gaps/router.py.
    focus_topics: List[FocusTopic] = Field(default_factory=list, max_length=15)

    @field_validator("question_type")
    @classmethod
    def check_question_type(cls, v: str) -> str:
        if v not in _VALID_EXAM_QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {_VALID_EXAM_QUESTION_TYPES}")
        return v

    @field_validator("difficulty")
    @classmethod
    def check_difficulty(cls, v: str) -> str:
        if v not in _VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {_VALID_DIFFICULTIES}")
        return v

    @field_validator("document_ids")
    @classmethod
    def check_document_ids(cls, v: List[str]) -> List[str]:
        for doc_id in v:
            try:
                uuid.UUID(doc_id)
            except ValueError as exc:
                raise ValueError(f"invalid document id: {doc_id}") from exc
        return v


class EvaluateRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=500)
    model_answer: str = Field(min_length=1, max_length=1500)
    user_answer: str = Field(min_length=1, max_length=3000)


class EvaluateResponse(BaseModel):
    correct: bool
    score: float
    feedback: str


class QuizErrorItem(BaseModel):
    """Una pregunta que el alumno respondió mal, tal como la arma QuizPanel."""
    question_type: str = Field(default="multiple_choice", max_length=20)
    question: str = Field(min_length=1, max_length=1000)
    explanation: str = Field(default="", max_length=3000)
    source_filename: Optional[str] = Field(default=None, max_length=255)
    # No es un UUID confiable — ver comentario en app/db/models.py::QuizError.
    source_document_id: Optional[str] = Field(default=None, max_length=64)
    options: List[str] = Field(default=[])
    correct_index: int = Field(default=-1, ge=-1, le=3)
    user_selected_index: Optional[int] = Field(default=None, ge=0, le=3)
    user_answer: Optional[str] = Field(default=None, max_length=3000)
    ai_feedback: Optional[str] = Field(default=None, max_length=3000)
    ai_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RecordErrorsRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    errors: List[QuizErrorItem] = Field(min_length=1, max_length=10)


class RecordErrorsResponse(BaseModel):
    stored: int


class ErrorsListRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    days: int = Field(default=90, ge=1, le=365)
    limit: int = Field(default=100, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class StoredQuizError(QuizErrorItem):
    id: str
    created_at: datetime


class ErrorsListResponse(BaseModel):
    course_id: int
    total: int
    items: List[StoredQuizError]


class ClearErrorsRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)


class ClearErrorsResponse(BaseModel):
    deleted: int


class ReviewSuggestionsRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    days: int = Field(default=90, ge=1, le=365)


class ReviewSuggestion(BaseModel):
    source_filename: Optional[str] = None
    source_document_id: Optional[str] = None
    error_count: int
    last_error_at: datetime
    topic: str
    suggestion: str


class ReviewSuggestionsResponse(BaseModel):
    course_id: int
    total_errors: int
    suggestions: List[ReviewSuggestion]


class RecordAttemptRequest(BaseModel):
    """Intento de quiz completado por el alumno (SP-09 + ANALYTICS-01)."""
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    question_type: Optional[str] = Field(default=None, max_length=20)
    difficulty: str = Field(default="medium", max_length=10)
    topic: Optional[str] = Field(default=None, max_length=200)
    total_questions: int = Field(ge=1, le=50)
    correct_answers: int = Field(ge=0, le=50)

    @field_validator("correct_answers")
    @classmethod
    def check_correct_not_greater_than_total(cls, v: int, info) -> int:
        total = info.data.get("total_questions")
        if total is not None and v > total:
            raise ValueError("correct_answers no puede ser mayor que total_questions")
        return v


class RecordAttemptResponse(BaseModel):
    id: str
    score: float


class AttemptsListRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    days: int = Field(default=90, ge=1, le=365)
    limit: int = Field(default=20, ge=1, le=100)


class AttemptItem(BaseModel):
    id: str
    question_type: Optional[str]
    difficulty: str
    topic: Optional[str]
    total_questions: int
    correct_answers: int
    score: float
    created_at: datetime


class AttemptsListResponse(BaseModel):
    course_id: int
    total: int
    items: List[AttemptItem]


class SuggestDifficultyRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    # Sin topic: sugerencia sobre el historial general del curso (alumno
    # practicando "de lo que sea"). Con topic: solo cuenta el historial de
    # ESE tema — sin match ahí, no hay sugerencia (SP-12, ver criterio de
    # aceptación: "sin historial previo en ese tema, comportamiento actual").
    topic: Optional[str] = Field(default=None, max_length=200)


class SuggestDifficultyResponse(BaseModel):
    difficulty: Optional[str] = None
    reason: Optional[str] = None
    based_on_attempts: int = 0
    # % de aciertos redondeado — se manda aparte (además de embebido en
    # `reason`, que está fijo en español) para que el front arme el texto en
    # inglés sin tener que parsear el string.
    accuracy_pct: Optional[int] = None


class StudyPlanRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    days: int = Field(default=30, ge=1, le=365)


class StudyPlanTopic(BaseModel):
    topic: str
    quiz_error_count: int
    gap_count: int
    reason: str
    suggested_quiz_topic: str
    # SP-13 (#323): IDs reales de fila que sustentan este tema — el `topic`
    # es texto generado por el LLM en cada llamada, no una clave estable, así
    # que "descartar este tema" opera sobre estos IDs (mismo patrón que
    # `question_ids` en app/gaps/router.py::GapItem).
    quiz_error_ids: List[str] = Field(default_factory=list)
    gap_question_ids: List[str] = Field(default_factory=list)


class StudyPlanResponse(BaseModel):
    course_id: int
    topics: List[StudyPlanTopic]


class StudyPlanDismissRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    quiz_error_ids: List[str] = Field(default_factory=list)
    gap_question_ids: List[str] = Field(default_factory=list)


class StudyPlanDismissResponse(BaseModel):
    affected: int


class FlashcardsSummaryRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    topic: Optional[str] = Field(default=None, max_length=200)


class FlashcardsSummaryResponse(BaseModel):
    due_count: int
    total_count: int


class FlashcardsDueRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    topic: Optional[str] = Field(default=None, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)


class FlashcardsDueResponse(BaseModel):
    course_id: int
    questions: List[QuizQuestion]


class FlashcardReviewItem(BaseModel):
    flashcard_id: str
    knew_it: bool


class FlashcardsReviewBatchRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    reviews: List[FlashcardReviewItem] = Field(min_length=1, max_length=50)


class FlashcardsReviewBatchResponse(BaseModel):
    updated: int


class StreakRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)


class StreakResponse(BaseModel):
    current_streak: int
    practiced_today: bool


# ============================================================
# Helpers
# ============================================================

def _flashcard_content_hash(question: str, explanation: str) -> str:
    return hashlib.sha256(f"{question.strip()}\n{explanation.strip()}".encode()).hexdigest()


async def _persist_flashcards(
    db: AsyncSession,
    course_id: int,
    topic: Optional[str],
    questions: list["QuizQuestion"],
) -> None:
    """Upsert de las flashcards generadas en `flashcards`, adjuntando el id real a cada una.

    SP-11 (#315): les da identidad estable para poder aplicar repetición
    espaciada. Dedup por (course_id, content_hash) vía ON CONFLICT DO NOTHING
    — regenerar el mismo contenido no crea filas duplicadas.
    """
    hashes = [_flashcard_content_hash(q.question, q.explanation) for q in questions]
    values = [
        {
            "id": uuid.uuid4(),
            "course_id": course_id,
            "topic": topic,
            "content_hash": h,
            "question": q.question,
            "explanation": q.explanation,
            "source_filename": q.source_filename or None,
            "source_document_id": q.source_document_id,
        }
        for q, h in zip(questions, hashes)
    ]
    stmt = pg_insert(Flashcard).values(values).on_conflict_do_nothing(
        index_elements=["course_id", "content_hash"]
    )
    await db.execute(stmt)
    await db.commit()

    id_rows = await db.execute(
        select(Flashcard.id, Flashcard.content_hash).where(
            Flashcard.course_id == course_id,
            Flashcard.content_hash.in_(hashes),
        )
    )
    hash_to_id = {row.content_hash: str(row.id) for row in id_rows.all()}
    for q, h in zip(questions, hashes):
        q.id = hash_to_id.get(h)


# SP-11 (#315): fórmula SM-2 estándar (el algoritmo detrás de Anki),
# simplificada porque la UI de autoevaluación es binaria ("Sabía"/"No
# sabía") en vez de una escala 0-5. Mapeo de quality: knew_it=True → 5,
# knew_it=False → 2 (no 0, para no destruir ease_factor de un solo
# tropiezo — mismo criterio de apps de repetición espaciada con 2 botones).
def _apply_sm2(review: FlashcardReview, knew_it: bool, now: datetime) -> None:
    quality = 5 if knew_it else 2
    if knew_it:
        if review.repetitions == 0:
            review.interval_days = 1
        elif review.repetitions == 1:
            review.interval_days = 6
        else:
            review.interval_days = max(1, round(review.interval_days * review.ease_factor))
        review.repetitions += 1
    else:
        # Reseteo a corto plazo — criterio de aceptación explícito de SP-11.
        review.repetitions = 0
        review.interval_days = 1

    new_ease = review.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    review.ease_factor = max(1.3, round(new_ease, 4))
    review.last_reviewed_at = now
    review.next_review_at = now + timedelta(days=review.interval_days)


async def _sample_chunks_for_quiz(
    db: AsyncSession,
    course_id: int,
    limit: int = 12,
) -> list[tuple[str, str]]:
    """Devuelve [(filename, content)] de chunks aleatorios indexed del curso.

    No usa embeddings — random sample. Útil cuando el alumno NO especifica
    topic y queremos variedad de temas en el quiz.
    """
    stmt = (
        select(Document.filename, Chunk.content)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.course_id == course_id)
        .where(Document.status == "indexed")
        .order_by(func.random())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [(row.filename, row.content) for row in result.all()]


async def _fetch_chunks_for_documents(
    db: AsyncSession,
    course_id: int,
    document_ids: list[str],
    limit: int = 20,
) -> list[tuple[str, str]]:
    """Devuelve [(filename, content)] de chunks de los archivos elegidos por el docente.

    A diferencia de `_sample_chunks_for_quiz` (todo el curso), acá se restringe
    explícitamente a `document_ids` — y siempre se re-filtra por `course_id`
    para que un docente no pueda pedir material de un curso ajeno pasando IDs
    de otro curso a mano.
    """
    stmt = (
        select(Document.filename, Chunk.content)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.course_id == course_id)
        .where(Document.status == "indexed")
        .where(Document.id.in_([uuid.UUID(d) for d in document_ids]))
        .order_by(func.random())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [(row.filename, row.content) for row in result.all()]


async def _run_quiz_generation(
    chunks: list[tuple[str, str]],
    course_id: int,
    topic: Optional[str],
    num_questions: int,
    question_type: str,
    difficulty: str,
    db: AsyncSession,
    llm: LLMProvider,
    focus_topics: Optional[list[FocusTopic]] = None,
) -> list[QuizQuestion]:
    """Llama al LLM con los chunks dados y devuelve preguntas validadas + enriquecidas.

    Compartido entre el quiz de práctica del alumno (`/generate`) y el
    generador de exámenes del docente (`/generate-exam`) — misma pipeline de
    prompt → JSON → validación → shuffle → mapeo a document_id, solo cambia
    de dónde salen los chunks.
    """
    focus_labels = [t.label for t in focus_topics] if focus_topics else None
    messages = _build_quiz_prompt(chunks, num_questions, topic, question_type, difficulty, focus_labels)
    try:
        result = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.6,
            # PERF-01: única llamada del sistema que pisa el default global de
            # thinking ("none"). Armar distractores plausibles y explicaciones
            # correctas mejora con algo de razonamiento, y acá el alumno ya
            # espera una pantalla de carga — no es una respuesta que se
            # streamee token a token como el chat.
            reasoning_effort=get_settings().llm_reasoning_effort_generation,
        )
    except Exception as exc:
        logger.error("Quiz LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar el quiz en este momento. Intentá de nuevo.",
        ) from exc

    raw = result.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Quiz JSON parse failed. Raw response: %.500s", raw)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El generador devolvió una respuesta inválida. Intentá de nuevo.",
        ) from exc

    if isinstance(parsed, dict) and "error" in parsed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=parsed.get(
                "detail",
                "El material del curso no contiene suficiente contenido sobre este tema para generar preguntas.",
            ),
        )

    questions_raw = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions_raw, list) or not questions_raw:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El generador no devolvió preguntas válidas.",
        )

    questions: list[QuizQuestion] = []
    for q in questions_raw:
        try:
            questions.append(QuizQuestion.model_validate(q))
        except ValidationError:
            # Saltear preguntas malformadas en lugar de tirar 503 — degradación graceful.
            continue

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ninguna pregunta generada pasó validación.",
        )

    # Truncar al número pedido (el LLM a veces se pasa por uno).
    questions = questions[:num_questions]

    # DOC-D09 (#390): descartar source_topic que no matchea ninguno de los
    # temas provistos — evita citar un tema inventado por el LLM.
    if focus_labels:
        valid_labels = set(focus_labels)
        for q in questions:
            if q.source_topic and q.source_topic not in valid_labels:
                q.source_topic = None
    else:
        for q in questions:
            q.source_topic = None

    # Shuffle de opciones para no dejar siempre la correcta en el mismo lugar.
    for q in questions:
        if q.question_type == "multiple_choice" and len(q.options) == 4 and q.correct_index >= 0:
            original_correct = q.options[q.correct_index]
            random.shuffle(q.options)
            q.correct_index = q.options.index(original_correct)

    # Enriquecer preguntas con el document_id del archivo fuente.
    source_filenames = {q.source_filename for q in questions if q.source_filename}
    if source_filenames:
        doc_stmt = select(Document.filename, Document.id).where(
            Document.course_id == course_id,
            Document.filename.in_(source_filenames),
        )
        doc_rows = await db.execute(doc_stmt)
        doc_id_map: dict[str, str] = {row.filename: str(row.id) for row in doc_rows.all()}
        for q in questions:
            if q.source_filename and q.source_filename in doc_id_map:
                q.source_document_id = doc_id_map[q.source_filename]

    return questions


_DIFFICULTY_INSTRUCTIONS = {
    "easy": "Nivel de dificultad: FÁCIL. Usá definiciones y conceptos básicos citados directamente en el material.",
    "medium": "Nivel de dificultad: MEDIA. Requerí aplicar o relacionar un concepto, no solo repetir una definición.",
    "hard": "Nivel de dificultad: DIFÍCIL. Requerí relacionar varios conceptos entre sí o razonar sobre un caso, no una definición aislada.",
}


def _build_quiz_prompt(
    chunks: list[tuple[str, str]],
    num_questions: int,
    topic: Optional[str],
    question_type: str = "multiple_choice",
    difficulty: str = "medium",
    focus_topics: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Arma los mensajes para el LLM según el tipo de pregunta solicitado."""

    error_clause = (
        "O, si el material no contiene suficiente contenido directo sobre el tema:\n"
        '{"error": "insufficient_content", "detail": "El material del curso no contiene suficiente contenido sobre este tema para generar preguntas."}\n'
    )

    if question_type == "true_false":
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "true_false",\n'
            '      "question": "<afirmación verdadera o falsa>",\n'
            '      "options": ["Verdadero", "Falso"],\n'
            '      "correct_index": 0,\n'
            '      "explanation": "<por qué es verdadero o falso>",\n'
            '      "source_filename": "<nombre del archivo fuente>"\n'
            '    }\n  ]\n}\n' + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. Cada pregunta es una AFIRMACIÓN (sin signo de interrogación al final).\n"
            "2. options SIEMPRE es exactamente [\"Verdadero\", \"Falso\"].\n"
            "3. correct_index es 0 si la afirmación es verdadera, 1 si es falsa.\n"
            "4. Las afirmaciones DEBEN basarse en el material entregado.\n"
            "5. Variá entre afirmaciones verdaderas y falsas en el conjunto.\n"
            "6. Las afirmaciones falsas deben ser plausibles (no obviamente absurdas).\n"
            "7. La explicación justifica por qué es verdadero o falso citando el material.\n"
            "8. source_filename DEBE ser uno de los nombres de archivo del material.\n"
            "9. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = f"Generá {num_questions} preguntas de Verdadero/Falso de práctica.\n\n"

    elif question_type == "open":
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "open",\n'
            '      "question": "<pregunta abierta que requiere explicar o desarrollar>",\n'
            '      "options": [],\n'
            '      "correct_index": -1,\n'
            '      "explanation": "<respuesta modelo completa y detallada, basada en el material>",\n'
            '      "source_filename": "<nombre del archivo fuente>"\n'
            '    }\n  ]\n}\n' + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. options SIEMPRE es [] (array vacío).\n"
            "2. correct_index SIEMPRE es -1.\n"
            "3. Las preguntas deben requerir que el alumno explique, describa o analice un concepto.\n"
            "4. Las preguntas DEBEN basarse en el material entregado.\n"
            "5. explanation DEBE ser una respuesta modelo completa que sirva como criterio de evaluación.\n"
            "6. source_filename DEBE ser uno de los nombres de archivo del material.\n"
            "7. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = f"Generá {num_questions} preguntas abiertas de práctica.\n\n"

    elif question_type == "mix":
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "multiple_choice",\n'
            '      "question": "<texto>",\n'
            '      "options": ["<A>", "<B>", "<C>", "<D>"],\n'
            '      "correct_index": 0,\n'
            '      "explanation": "<explicación>",\n'
            '      "source_filename": "<archivo>"\n'
            '    }\n  ]\n}\n'
            "Cada pregunta puede ser de tipo 'multiple_choice', 'true_false' u 'open'.\n"
            "Para true_false: options=[\"Verdadero\",\"Falso\"], correct_index 0 o 1.\n"
            "Para open: options=[], correct_index=-1, explanation=respuesta modelo.\n"
            + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. Incluí los tres tipos: multiple_choice, true_false y open. Distribuílos de forma pareja.\n"
            "2. Para multiple_choice: exactamente 4 opciones, correct_index 0-3.\n"
            "3. Para true_false: options=[\"Verdadero\",\"Falso\"], correct_index 0 o 1.\n"
            "4. Para open: options=[], correct_index=-1, explanation=respuesta modelo completa.\n"
            "5. Todas las preguntas DEBEN basarse en el material entregado.\n"
            "6. source_filename DEBE ser uno de los nombres de archivo del material.\n"
            "7. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = (
            f"Generá {num_questions} preguntas de práctica variando los tipos "
            "(opción múltiple, verdadero/falso y preguntas abiertas).\n\n"
        )

    elif question_type == "flashcard":
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "flashcard",\n'
            '      "question": "<frente de la tarjeta: pregunta o término corto>",\n'
            '      "options": [],\n'
            '      "correct_index": -1,\n'
            '      "explanation": "<dorso de la tarjeta: respuesta corta y directa>",\n'
            '      "source_filename": "<nombre del archivo fuente>"\n'
            '    }\n  ]\n}\n' + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. options SIEMPRE es [] (array vacío).\n"
            "2. correct_index SIEMPRE es -1.\n"
            "3. El 'question' (frente) es CORTO: un término, una pregunta puntual o una definición a completar — "
            "NO una consigna de desarrollo.\n"
            "4. El 'explanation' (dorso) es la respuesta CORTA y directa a ese frente, no un párrafo largo.\n"
            "5. Las tarjetas DEBEN basarse en el material entregado.\n"
            "6. source_filename DEBE ser uno de los nombres de archivo del material.\n"
            "7. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = f"Generá {num_questions} flashcards de práctica (pregunta corta / respuesta corta).\n\n"

    elif question_type == "fill_blank":
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "fill_blank",\n'
            '      "question": "<oración con exactamente un _____ donde falta la palabra>",\n'
            '      "options": [],\n'
            '      "correct_index": -1,\n'
            '      "explanation": "<palabra_correcta> — <justificación breve basada en el material>",\n'
            '      "source_filename": "<nombre del archivo fuente>"\n'
            '    }\n  ]\n}\n' + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. options SIEMPRE es [] (array vacío).\n"
            "2. correct_index SIEMPRE es -1.\n"
            "3. La 'question' es una oración con EXACTAMENTE UN espacio en blanco marcado como '_____' (5 guiones bajos).\n"
            "4. La palabra omitida DEBE ser un término técnico, conceptual o relevante del material.\n"
            "5. La oración debe tener suficiente contexto para que se pueda deducir la palabra correcta.\n"
            "6. NO uses oraciones ambiguas donde varias palabras serían igualmente correctas.\n"
            "7. El 'explanation' comienza con la palabra correcta, seguida de ' — ' y luego una justificación breve.\n"
            "   Ejemplo: \"mitocondria — La mitocondria es el orgánulo encargado de la respiración celular aeróbica.\"\n"
            "8. Las oraciones DEBEN basarse en el material entregado.\n"
            "9. source_filename DEBE ser uno de los nombres de archivo del material.\n"
            "10. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = f"Generá {num_questions} ejercicios de completar espacios en blanco.\n\n"

    else:  # multiple_choice (default)
        schema_hint = (
            "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
            '{\n  "questions": [\n    {\n'
            '      "question_type": "multiple_choice",\n'
            '      "question": "<texto de la pregunta>",\n'
            '      "options": ["<opción A>", "<opción B>", "<opción C>", "<opción D>"],\n'
            '      "correct_index": 0,\n'
            '      "explanation": "<por qué la opción correcta es correcta>",\n'
            '      "source_filename": "<nombre del archivo del que sale la respuesta>"\n'
            '    }\n  ]\n}\n' + error_clause
        )
        rules = (
            "Reglas:\n"
            "1. Cada pregunta tiene EXACTAMENTE 4 opciones.\n"
            "2. correct_index es un entero entre 0 y 3 que indica la opción correcta.\n"
            "3. Las preguntas y respuestas DEBEN basarse ÚNICAMENTE en el material entregado.\n"
            "4. NUNCA inventes contenido que no esté en el material.\n"
            "5. Las opciones incorrectas (distractores) tienen que ser plausibles, no obviamente absurdas.\n"
            "6. La explicación debe citar implícitamente el fragmento que justifica la respuesta.\n"
            "7. source_filename DEBE ser uno de los nombres de archivo que aparecen en el material.\n"
            "8. NO usar markdown ni texto fuera del JSON.\n"
        )
        type_instruction = f"Generá {num_questions} preguntas de opción múltiple de práctica.\n\n"

    topic_line = (
        f"Tema solicitado: {topic.strip()}\n\n"
        if topic and topic.strip()
        else "El alumno no especificó tema — variá temas para cubrir el material disponible.\n\n"
    )

    # DOC-D09 (#390): temas con dificultad detectada (Gaps/FAQ) que el
    # docente pidió priorizar al generar un examen. Además del schema por
    # tipo de pregunta de arriba, cada pregunta debe declarar de qué tema de
    # esta lista salió (campo extra "source_topic", fuera del schema_hint
    # porque aplica igual a los 6 tipos de pregunta).
    focus_topics_block = ""
    if focus_topics:
        topics_list = "\n".join(f"- {t}" for t in focus_topics)
        focus_topics_block = (
            "TEMAS CON DIFICULTAD DETECTADA (priorizar cobertura de estos temas, "
            "en base a preguntas de alumnos sin responder o frecuentes):\n"
            f"{topics_list}\n\n"
            "Además de los campos del schema de arriba, agregá en cada pregunta un "
            'campo "source_topic": el texto EXACTO del tema de la lista anterior que '
            'esa pregunta cubre, o "" si la pregunta no cubre ninguno de esos temas '
            "puntualmente. No inventes un tema que no esté en la lista.\n\n"
        )

    material_block = "\n\n".join(
        f'FRAGMENTO {i + 1} (de "{filename}"):\n{content[:600].strip()}'
        for i, (filename, content) in enumerate(chunks)
    )

    type_desc = {
        "multiple_choice": "opción múltiple",
        "true_false": "Verdadero/Falso",
        "open": "preguntas abiertas evaluadas por IA",
        "mix": "mixto (opción múltiple, V/F y preguntas abiertas)",
        "flashcard": "flashcards de pregunta y respuesta corta",
        "fill_blank": "completar espacios en blanco",
    }.get(question_type, "opción múltiple")

    difficulty_line = _DIFFICULTY_INSTRUCTIONS.get(difficulty, _DIFFICULTY_INSTRUCTIONS["medium"])

    system = (
        f"Sos un generador de quizzes académicos de NexusAI. "
        f"Producís preguntas de {type_desc} en español, basadas estrictamente "
        "en el material académico del curso del alumno. Tu salida es JSON.\n\n"
        f"{difficulty_line}\n\n"
        "REGLA CRÍTICA: Solo podés generar preguntas sobre contenido que esté "
        "explícita y directamente presente en el material entregado.\n"
        "- NO generes preguntas sobre la ausencia de un tema.\n"
        "- NO generes preguntas del tipo \"¿Qué información sobre X se puede encontrar?\", "
        "\"¿Se menciona X en el material?\" o \"¿Dónde se habla de X?\".\n"
        f"- Si el material no contiene suficiente contenido directo sobre el tema pedido "
        f"para formar {num_questions} preguntas legítimas, respondé con el objeto de error "
        "descripto en el formato de salida en lugar del array de preguntas."
    )

    user_msg = (
        f"{type_instruction}"
        f"{topic_line}"
        f"{focus_topics_block}"
        f"{schema_hint}\n"
        f"{rules}\n"
        f"--- MATERIAL DEL CURSO ---\n\n{material_block}\n\n--- FIN DEL MATERIAL ---"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ============================================================
# Endpoint
# ============================================================

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    payload: QuizRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> QuizResponse:
    """Genera un quiz de práctica con N preguntas basadas en el material del curso.

    Si no hay material indexado en el curso → 404.
    Si el LLM devuelve JSON inválido o falla → 503.
    """
    # 1) Conseguir material para el quiz.
    has_topic = bool(payload.topic and payload.topic.strip())

    if has_topic:
        # Modo dirigido: el alumno pidió un tema específico.
        # Validación en dos pasos para evitar falsos positivos por similitud
        # semántica cruzada (ej. "derivadas" matchea débilmente contra cualquier PDF).
        try:
            retrieved = await retrieve_context(
                question=payload.topic,
                course_id=payload.course_id,
                db=db,
                embeddings=embeddings,
                top_k=5,
                min_similarity=QUIZ_TOPIC_MIN_SIMILARITY,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No se pudo procesar el tema en este momento. Intentá de nuevo.",
            ) from exc

        # Step 1: rechazo semántico — ningún chunk superó el umbral mínimo.
        max_sim = max((c.similarity for c in retrieved), default=0.0)
        if max_sim < QUIZ_TOPIC_MIN_SIMILARITY:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "No encontré material sobre ese tema en el curso. "
                    "Intentá con un tema que esté cubierto en los archivos indexados."
                ),
            )

        # Step 2: verificación LLM — confirma que los chunks realmente cubren el tema.
        # Previene falsos positivos donde la similitud semántica pasa el umbral
        # pero el contenido no tiene relación directa con el topic pedido.
        excerpts = "\n---\n".join(c.content[:200].strip() for c in retrieved)
        relevance_messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic content relevance checker. Answer only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {payload.topic.strip()}\n\n"
                    f"Course material excerpts:\n{excerpts}\n\n"
                    f"Is '{payload.topic.strip()}' meaningfully present in this course material — "
                    "either as a main subject, a key concept explained, or a named entity directly discussed?\n"
                    "Answer NO only if the topic has no real presence in the excerpts at all.\n"
                    "Answer only YES or NO."
                ),
            },
        ]
        try:
            relevance_result = await llm.chat_completion(
                relevance_messages,
                max_tokens=5,
                temperature=0.0,
            )
            answer = relevance_result.text.strip().upper()
        except Exception as exc:
            logger.warning("LLM relevance check failed, proceeding with generation: %s", exc)
            answer = "YES"

        if not answer.startswith("YES"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"No encontré material sobre '{payload.topic.strip()}' en los archivos del curso. "
                    "Intentá con un tema que esté cubierto en los archivos indexados."
                ),
            )

        chunks = [(c.document_filename, c.content) for c in retrieved]
    else:
        # Modo variedad: sampling aleatorio del material del curso.
        chunks = await _sample_chunks_for_quiz(db, payload.course_id, limit=12)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Este curso todavía no tiene material indexado para generar un quiz.",
        )

    # 2) Pedir al LLM la generación + parseo/validación/enriquecimiento (compartido con /generate-exam).
    questions = await _run_quiz_generation(
        chunks=chunks,
        course_id=payload.course_id,
        topic=payload.topic,
        num_questions=payload.num_questions,
        question_type=payload.question_type,
        difficulty=payload.difficulty,
        db=db,
        llm=llm,
    )

    # SP-11 (#315): persistir flashcards generadas para darles identidad
    # estable — habilita repetición espaciada (/flashcards/*). Solo aplica
    # a este endpoint (el alumno practicando); /generate-exam no admite
    # question_type='flashcard'.
    if payload.question_type == "flashcard" and questions:
        await _persist_flashcards(db, payload.course_id, payload.topic, questions)

    return QuizResponse(
        course_id=payload.course_id,
        topic=payload.topic,
        questions=questions,
    )


@router.post("/generate-exam", response_model=QuizResponse)
async def generate_exam(
    payload: ExamGenerateRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
) -> QuizResponse:
    """Genera un banco de preguntas de examen para el docente (EVAL-01 / issue #235).

    A diferencia de `/generate` (quiz de práctica del alumno), el docente elige
    explícitamente de qué archivos del curso salen las preguntas — no hay
    sampling aleatorio de todo el curso ni búsqueda semántica por tema.

    La autorización de rol (solo docentes) se hace en el plugin de Moodle vía
    `require_capability('local/nexusai:manage', ...)`, igual que el resto de
    las funciones del dashboard docente — este endpoint solo valida HMAC.

    Si ninguno de los `document_ids` tiene material indexado en ese curso → 404.
    Si el LLM devuelve JSON inválido o falla → 503.
    """
    chunks = await _fetch_chunks_for_documents(
        db, payload.course_id, payload.document_ids, limit=20
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Los archivos seleccionados no tienen material indexado en este curso.",
        )

    questions = await _run_quiz_generation(
        chunks=chunks,
        course_id=payload.course_id,
        topic=payload.topic,
        num_questions=payload.num_questions,
        question_type=payload.question_type,
        difficulty=payload.difficulty,
        db=db,
        llm=llm,
        focus_topics=payload.focus_topics,
    )

    return QuizResponse(
        course_id=payload.course_id,
        topic=payload.topic,
        questions=questions,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_open_answer(
    payload: EvaluateRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    llm: LLMProvider = Depends(get_llm_provider),
) -> EvaluateResponse:
    """Evalúa la respuesta libre de un alumno usando LLM (SP-05).

    Devuelve { correct, score 0-1, feedback } con justificación detallada.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Sos un evaluador de respuestas académicas de NexusAI. "
                "Evaluás respuestas abiertas de alumnos comparándolas con el contenido del curso. "
                "Tu salida es JSON.\n\n"
                "Devolvé EXCLUSIVAMENTE un JSON con esta forma exacta:\n"
                '{"correct": true, "score": 0.85, "feedback": "<feedback detallado en español>"}\n'
                "- correct: true si el alumno demostró comprensión del concepto principal.\n"
                "- score: valor entre 0.0 y 1.0 representando la calidad de la respuesta.\n"
                "- feedback: feedback constructivo en español; mencioná qué estuvo bien, "
                "qué faltó y cuál es la respuesta correcta completa."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pregunta: {payload.question}\n\n"
                f"Respuesta esperada (basada en el material del curso):\n{payload.model_answer}\n\n"
                f"Respuesta del alumno:\n{payload.user_answer}\n\n"
                "Evaluá la respuesta del alumno y devolvé el JSON."
            ),
        },
    ]

    try:
        result = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as exc:
        logger.error("Evaluate LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo evaluar la respuesta en este momento. Intentá de nuevo.",
        ) from exc

    raw = result.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    try:
        parsed = json.loads(raw)
        return EvaluateResponse(
            correct=bool(parsed.get("correct", False)),
            score=max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
            feedback=str(parsed.get("feedback", "Sin feedback disponible.")),
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Evaluate JSON parse failed. Raw: %.300s", raw)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo procesar la evaluación. Intentá de nuevo.",
        ) from exc


# ============================================================
# Repaso de errores — SP-10
# ============================================================

@router.post("/errors", response_model=RecordErrorsResponse)
async def record_quiz_errors(
    payload: RecordErrorsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> RecordErrorsResponse:
    """Persiste las preguntas que el alumno respondió mal en un quiz (SP-10).

    Reemplaza el localStorage efímero que usaba antes el frontend: el
    historial de errores ahora vive en Postgres, por usuario+curso, y
    sobrevive entre dispositivos/sesiones.
    """
    rows = [
        QuizError(
            course_id=payload.course_id,
            user_id=payload.user_id,
            question_type=e.question_type,
            question=e.question,
            explanation=e.explanation,
            source_filename=e.source_filename,
            source_document_id=e.source_document_id,
            options=e.options,
            correct_index=e.correct_index,
            user_selected_index=e.user_selected_index,
            user_answer=e.user_answer,
            ai_feedback=e.ai_feedback,
            ai_score=e.ai_score,
        )
        for e in payload.errors
    ]
    db.add_all(rows)
    await db.commit()
    return RecordErrorsResponse(stored=len(rows))


@router.post("/errors/list", response_model=ErrorsListResponse)
async def list_quiz_errors(
    payload: ErrorsListRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> ErrorsListResponse:
    """Historial de errores del alumno en un curso, más recientes primero.

    UX-19 (#389): `total` es el conteo real de errores en el rango de
    `days` (no solo el tamaño de la página devuelta), para que el frontend
    pueda saber si hay más resultados y pedirlos con `offset`.
    """
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    base_filters = (
        QuizError.course_id == payload.course_id,
        QuizError.user_id == payload.user_id,
        QuizError.created_at >= since,
    )

    total = await db.scalar(
        select(func.count()).select_from(QuizError).where(*base_filters)
    )

    stmt = (
        select(QuizError)
        .where(*base_filters)
        .order_by(desc(QuizError.created_at))
        .offset(payload.offset)
        .limit(payload.limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    items = [
        StoredQuizError(
            id=str(r.id),
            created_at=r.created_at,
            question_type=r.question_type,
            question=r.question,
            explanation=r.explanation,
            source_filename=r.source_filename,
            source_document_id=r.source_document_id,
            options=r.options or [],
            correct_index=r.correct_index,
            user_selected_index=r.user_selected_index,
            user_answer=r.user_answer,
            ai_feedback=r.ai_feedback,
            ai_score=r.ai_score,
        )
        for r in rows
    ]
    return ErrorsListResponse(course_id=payload.course_id, total=total or 0, items=items)


@router.post("/errors/clear", response_model=ClearErrorsResponse)
async def clear_quiz_errors(
    payload: ClearErrorsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> ClearErrorsResponse:
    """Borra todo el historial de errores del alumno en un curso."""
    stmt = delete(QuizError).where(
        QuizError.course_id == payload.course_id,
        QuizError.user_id == payload.user_id,
    )
    result = await db.execute(stmt)
    await db.commit()
    return ClearErrorsResponse(deleted=result.rowcount or 0)


@router.post("/review-suggestions", response_model=ReviewSuggestionsResponse)
async def review_suggestions(
    payload: ReviewSuggestionsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
) -> ReviewSuggestionsResponse:
    """Analiza el historial de errores del alumno y sugiere qué repasar (SP-10).

    No hay taxonomía de temas en el schema, así que se usa `source_filename`
    como proxy de tema: se agrupan los errores recientes por archivo fuente,
    se toman los grupos más frecuentes, y se le pide al LLM una síntesis
    accionable por grupo. El conteo y el orden se calculan acá — nunca se
    confía en lo que devuelva el LLM para eso.
    """
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    stmt = (
        select(QuizError)
        .where(QuizError.course_id == payload.course_id)
        .where(QuizError.user_id == payload.user_id)
        .where(QuizError.created_at >= since)
        .order_by(desc(QuizError.created_at))
        .limit(300)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return ReviewSuggestionsResponse(course_id=payload.course_id, total_errors=0, suggestions=[])

    # Agrupar por archivo fuente (fallback a "material general" si no hay filename).
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r.source_filename or "__general__"
        g = groups.setdefault(key, {
            "filename": r.source_filename,
            "document_id": r.source_document_id,
            "count": 0,
            "last_at": r.created_at,
            "samples": [],
        })
        g["count"] += 1
        if r.created_at > g["last_at"]:
            g["last_at"] = r.created_at
        if not g["document_id"] and r.source_document_id:
            g["document_id"] = r.source_document_id
        if len(g["samples"]) < 3:
            g["samples"].append({"question": r.question, "explanation": r.explanation})

    top_groups = sorted(groups.values(), key=lambda g: g["count"], reverse=True)[:5]

    prompt_blocks = []
    for i, g in enumerate(top_groups):
        label = g["filename"] or "Material general del curso"
        samples_text = "\n".join(
            f'  - Pregunta: {s["question"]}\n    Respuesta correcta: {s["explanation"]}'
            for s in g["samples"]
        )
        prompt_blocks.append(f'Grupo {i} — fuente: "{label}" ({g["count"]} errores)\n{samples_text}')

    messages = [
        {
            "role": "system",
            "content": (
                "Sos un tutor de NexusAI que ayuda a un alumno a priorizar su repaso. "
                "Te paso grupos de preguntas que el alumno respondió mal, agrupadas por archivo "
                "fuente del curso. Para cada grupo, identificá el/los subtemas puntuales que el "
                "alumno no domina y dale una sugerencia concreta de qué repasar y cómo. "
                "Tu salida es JSON.\n\n"
                "Devolvé EXCLUSIVAMENTE un JSON con esta forma exacta:\n"
                '{"suggestions": [{"group": 0, "topic": "<subtema en 3-6 palabras>", '
                '"suggestion": "<2-4 oraciones en español, concretas y accionables>"}]}\n'
                "- topic: nombrá el/los conceptos puntuales que fallan, NO el nombre del archivo.\n"
                "- suggestion: explicá qué patrón de error ves y qué debería releer/practicar.\n"
                "- Devolvé un objeto de 'suggestions' por cada grupo recibido, con el mismo índice 'group'."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(prompt_blocks),
        },
    ]

    try:
        result_llm = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as exc:
        logger.error("Review suggestions LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudieron generar las sugerencias de repaso en este momento. Intentá de nuevo.",
        ) from exc

    raw = result_llm.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    try:
        parsed = json.loads(raw)
        by_group = {int(item.get("group", -1)): item for item in parsed.get("suggestions", [])}
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        logger.error("Review suggestions JSON parse failed. Raw: %.300s", raw)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudieron procesar las sugerencias de repaso. Intentá de nuevo.",
        ) from exc

    suggestions = [
        ReviewSuggestion(
            source_filename=g["filename"],
            source_document_id=g["document_id"],
            error_count=g["count"],
            last_error_at=g["last_at"],
            topic=str(by_group.get(i, {}).get("topic") or (g["filename"] or "Repaso general")),
            suggestion=str(
                by_group.get(i, {}).get("suggestion")
                or "Revisá el material relacionado con estas preguntas."
            ),
        )
        for i, g in enumerate(top_groups)
    ]

    return ReviewSuggestionsResponse(
        course_id=payload.course_id,
        total_errors=len(rows),
        suggestions=suggestions,
    )


# ============================================================
# Historial de quizzes — SP-09
# ============================================================

@router.post("/attempts", response_model=RecordAttemptResponse)
async def record_quiz_attempt(
    payload: RecordAttemptRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> RecordAttemptResponse:
    """Persiste el resultado de un quiz completado por el alumno (SP-09 + ANALYTICS-01).

    SP-09: registra tipo, dificultad y tema para el historial del alumno.
    ANALYTICS-01: score calculado server-side (correct_answers/total_questions)
    para el histograma del dashboard docente — nunca se confía en un score
    enviado por el cliente.
    """
    score = round(payload.correct_answers / payload.total_questions, 4)
    row = QuizAttempt(
        course_id=payload.course_id,
        user_id=payload.user_id,
        question_type=payload.question_type,
        difficulty=payload.difficulty,
        topic=payload.topic,
        total_questions=payload.total_questions,
        correct_answers=payload.correct_answers,
        score=score,
    )
    db.add(row)
    await db.commit()
    return RecordAttemptResponse(id=str(row.id), score=score)


@router.post("/attempts/list", response_model=AttemptsListResponse)
async def list_quiz_attempts(
    payload: AttemptsListRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> AttemptsListResponse:
    """Historial de quizzes completados por el alumno en un curso, más recientes primero (SP-09)."""
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    stmt = (
        select(QuizAttempt)
        .where(QuizAttempt.course_id == payload.course_id)
        .where(QuizAttempt.user_id == payload.user_id)
        .where(QuizAttempt.created_at >= since)
        .order_by(desc(QuizAttempt.created_at))
        .limit(payload.limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    items = [
        AttemptItem(
            id=str(r.id),
            question_type=r.question_type,
            difficulty=r.difficulty,
            topic=r.topic,
            total_questions=r.total_questions,
            correct_answers=r.correct_answers,
            score=r.score,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return AttemptsListResponse(course_id=payload.course_id, total=len(items), items=items)


# SP-12 (#322): umbrales de sugerencia — el issue no fija un número exacto,
# así que quedan documentados acá. >=80% de aciertos sugiere subir a hard,
# <=40% sugiere bajar a easy; el resto se queda en medium (el default actual).
_SUGGEST_HARD_THRESHOLD = 0.8
_SUGGEST_EASY_THRESHOLD = 0.4
_SUGGEST_ATTEMPT_LIMIT = 10


@router.post("/suggest-difficulty", response_model=SuggestDifficultyResponse)
async def suggest_difficulty(
    payload: SuggestDifficultyRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> SuggestDifficultyResponse:
    """Sugiere una dificultad de partida para el generador de quiz (SP-12).

    Solo lectura sobre `quiz_attempts` (ya persistido, SP-09/ANALYTICS-01) —
    no hay tabla ni migración nueva. Es una SUGERENCIA, nunca una
    restricción: el alumno siempre puede elegir otra dificultad a mano.
    """
    stmt = (
        select(QuizAttempt)
        .where(QuizAttempt.course_id == payload.course_id)
        .where(QuizAttempt.user_id == payload.user_id)
        .order_by(desc(QuizAttempt.created_at))
        .limit(_SUGGEST_ATTEMPT_LIMIT)
    )
    if payload.topic:
        stmt = stmt.where(func.lower(QuizAttempt.topic) == payload.topic.strip().lower())

    rows = (await db.execute(stmt)).scalars().all()

    if not rows:
        return SuggestDifficultyResponse()

    avg_score = sum(r.score for r in rows) / len(rows)
    if avg_score >= _SUGGEST_HARD_THRESHOLD:
        difficulty = "hard"
    elif avg_score <= _SUGGEST_EASY_THRESHOLD:
        difficulty = "easy"
    else:
        difficulty = "medium"

    pct = round(avg_score * 100)
    difficulty_label = {"easy": "fácil", "medium": "media", "hard": "difícil"}[difficulty]
    reason = (
        f"Basado en tus últimos {len(rows)} intento{'s' if len(rows) != 1 else ''} "
        f"({pct}% de aciertos), te sugerimos dificultad {difficulty_label}."
    )

    return SuggestDifficultyResponse(
        difficulty=difficulty,
        reason=reason,
        based_on_attempts=len(rows),
        accuracy_pct=pct,
    )


@router.post("/study-plan", response_model=StudyPlanResponse)
async def study_plan(
    payload: StudyPlanRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
) -> StudyPlanResponse:
    """Plan de estudio personalizado: combina errores de quiz + gaps del chat.

    Dos señales de "dónde le cuesta al alumno" que hoy viven separadas:
    errores de quiz (QuizError, agrupados por source_filename como proxy de
    tema — igual que review_suggestions) y preguntas del chat que el
    material no pudo responder bien (UnansweredQuestion, sin ninguna
    columna de tema, agrupadas por texto normalizado — igual que
    gaps/router.py, pero acá filtrado también por user_id porque esto es
    personal del alumno, no agregado de todo el curso). Un único LLM call
    sintetiza ambas señales en una lista unificada de temas débiles. El
    conteo y el orden se calculan acá — nunca se confía en lo que
    devuelva el LLM para eso.
    """
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    quiz_stmt = (
        select(QuizError)
        .where(QuizError.course_id == payload.course_id)
        .where(QuizError.user_id == payload.user_id)
        .where(QuizError.created_at >= since)
        .where(QuizError.dismissed_at.is_(None))  # SP-13: descartado por el alumno
        .order_by(desc(QuizError.created_at))
        .limit(300)
    )
    quiz_rows = (await db.execute(quiz_stmt)).scalars().all()

    norm_question = func.lower(func.trim(UnansweredQuestion.question))
    gap_stmt = (
        select(
            norm_question.label("question"),
            func.count().label("count"),
            func.max(UnansweredQuestion.created_at).label("last_asked_at"),
            # SP-13: IDs reales detrás del grupo, para poder descartar el tema
            # sin depender del texto (ver StudyPlanTopic.gap_question_ids).
            func.array_agg(UnansweredQuestion.id).label("ids"),
        )
        .where(UnansweredQuestion.course_id == payload.course_id)
        .where(UnansweredQuestion.user_id == payload.user_id)
        .where(UnansweredQuestion.created_at >= since)
        .where(UnansweredQuestion.student_dismissed_at.is_(None))
        .group_by(norm_question)
        .order_by(desc("count"), desc("last_asked_at"))
        .limit(20)
    )
    gap_rows = (await db.execute(gap_stmt)).all()

    if not quiz_rows and not gap_rows:
        return StudyPlanResponse(course_id=payload.course_id, topics=[])

    # Agrupar errores de quiz por archivo fuente (idéntico a review_suggestions).
    quiz_groups: dict[str, dict[str, Any]] = {}
    for r in quiz_rows:
        key = r.source_filename or "__general__"
        g = quiz_groups.setdefault(key, {"filename": r.source_filename, "count": 0, "samples": [], "ids": []})
        g["count"] += 1
        g["ids"].append(str(r.id))
        if len(g["samples"]) < 3:
            g["samples"].append({"question": r.question, "explanation": r.explanation})
    top_quiz_groups = sorted(quiz_groups.values(), key=lambda g: g["count"], reverse=True)[:5]

    top_gap_groups = [
        {"question": row.question, "count": int(row.count), "ids": [str(i) for i in row.ids]}
        for row in gap_rows
    ]

    prompt_blocks = []
    for i, g in enumerate(top_quiz_groups):
        label = g["filename"] or "Material general del curso"
        samples_text = "\n".join(
            f'  - Pregunta: {s["question"]}\n    Respuesta correcta: {s["explanation"]}'
            for s in g["samples"]
        )
        prompt_blocks.append(f'Grupo Q{i} — errores de quiz, fuente: "{label}" ({g["count"]} errores)\n{samples_text}')
    for j, g in enumerate(top_gap_groups):
        prompt_blocks.append(f'Grupo G{j} — pregunta del chat sin responder bien: "{g["question"]}" ({g["count"]} veces)')

    messages = [
        {
            "role": "system",
            "content": (
                "Sos un tutor de NexusAI que arma un plan de estudio personalizado para un "
                "alumno. Te paso dos tipos de evidencia: grupos de preguntas de quiz que "
                "respondió mal (prefijo Q), y preguntas que le hizo al asistente de chat y "
                "el material del curso no pudo responder bien (prefijo G). Identificá los "
                "temas débiles más importantes combinando ambas señales — un mismo tema "
                "puede aparecer en evidencia Q y G a la vez. Tu salida es JSON.\n\n"
                "Devolvé EXCLUSIVAMENTE un JSON con esta forma exacta:\n"
                '{"topics": [{"topic": "<tema en 3-6 palabras>", '
                '"quiz_groups": [0, 2], "gap_groups": [1], '
                '"reason": "<1-2 oraciones explicando por qué es un tema débil>", '
                '"suggested_quiz_topic": "<2-4 palabras para buscar este tema en un generador de quiz>"}]}\n'
                "- Máximo 6 temas, ordenados del más al menos urgente.\n"
                "- quiz_groups/gap_groups: índices (0-based) de los grupos Q/G que sustentan ese tema — "
                "pueden estar vacíos si el tema solo tiene evidencia de un tipo.\n"
                "- No repitas el mismo grupo en dos temas distintos."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(prompt_blocks),
        },
    ]

    try:
        result_llm = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as exc:
        logger.error("Study plan LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar el plan de estudio en este momento. Intentá de nuevo.",
        ) from exc

    raw = result_llm.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    try:
        parsed = json.loads(raw)
        llm_topics = parsed.get("topics", [])
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        logger.error("Study plan JSON parse failed. Raw: %.300s", raw)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo procesar el plan de estudio. Intentá de nuevo.",
        ) from exc

    topics: List[StudyPlanTopic] = []
    for item in llm_topics:
        if not isinstance(item, dict) or not item.get("topic"):
            continue
        quiz_idx = [i for i in item.get("quiz_groups", []) if isinstance(i, int) and 0 <= i < len(top_quiz_groups)]
        gap_idx = [j for j in item.get("gap_groups", []) if isinstance(j, int) and 0 <= j < len(top_gap_groups)]
        quiz_error_count = sum(top_quiz_groups[i]["count"] for i in quiz_idx)
        gap_count = sum(top_gap_groups[j]["count"] for j in gap_idx)
        if quiz_error_count == 0 and gap_count == 0:
            continue
        quiz_error_ids = [id_ for i in quiz_idx for id_ in top_quiz_groups[i]["ids"]]
        gap_question_ids = [id_ for j in gap_idx for id_ in top_gap_groups[j]["ids"]]
        topics.append(
            StudyPlanTopic(
                topic=str(item["topic"]),
                quiz_error_count=quiz_error_count,
                gap_count=gap_count,
                reason=str(item.get("reason") or ""),
                suggested_quiz_topic=str(item.get("suggested_quiz_topic") or item["topic"]),
                quiz_error_ids=quiz_error_ids,
                gap_question_ids=gap_question_ids,
            )
        )

    topics.sort(key=lambda t: t.quiz_error_count + t.gap_count, reverse=True)

    return StudyPlanResponse(course_id=payload.course_id, topics=topics[:6])


@router.post("/study-plan/dismiss", response_model=StudyPlanDismissResponse)
async def study_plan_dismiss(
    payload: StudyPlanDismissRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> StudyPlanDismissResponse:
    """Descarta un tema puntual del plan de estudio (SP-13, issue #323).

    Opera sobre los IDs reales de fila (`quiz_error_ids`/`gap_question_ids`,
    devueltos por /study-plan) — el `topic` que ve el alumno es texto
    generado por el LLM en cada llamada, no una clave estable. Mismo patrón
    que app/gaps/router.py::gaps_archive, pero con columnas propias
    (`QuizError.dismissed_at` / `UnansweredQuestion.student_dismissed_at`)
    que NO tocan `archived_at` (archivado del docente, DOC-D08) — descartar
    del lado del alumno no debe cambiar lo que ve el docente en Gaps/Analytics.
    """
    affected = 0

    if payload.quiz_error_ids:
        quiz_ids = [uuid.UUID(i) for i in payload.quiz_error_ids]
        result = await db.execute(
            update(QuizError)
            .where(QuizError.course_id == payload.course_id)
            .where(QuizError.user_id == payload.user_id)
            .where(QuizError.id.in_(quiz_ids))
            .values(dismissed_at=datetime.now(timezone.utc))
        )
        affected += result.rowcount

    if payload.gap_question_ids:
        gap_ids = [uuid.UUID(i) for i in payload.gap_question_ids]
        result = await db.execute(
            update(UnansweredQuestion)
            .where(UnansweredQuestion.course_id == payload.course_id)
            .where(UnansweredQuestion.user_id == payload.user_id)
            .where(UnansweredQuestion.id.in_(gap_ids))
            .values(student_dismissed_at=datetime.now(timezone.utc))
        )
        affected += result.rowcount

    await db.commit()

    return StudyPlanDismissResponse(affected=affected)


# ============================================================
# Repetición espaciada de flashcards — SP-11 (#315)
# ============================================================

def _flashcard_due_filter(user_id: int):
    """Condición ON del LEFT JOIN + WHERE reusada por summary/due: "toca hoy"
    significa sin fila de review, o next_review_at NULL, o ya vencido."""
    join_cond = (
        (FlashcardReview.flashcard_id == Flashcard.id)
        & (FlashcardReview.user_id == user_id)
        & (FlashcardReview.deleted_at.is_(None))
    )
    due_cond = (
        FlashcardReview.id.is_(None)
        | FlashcardReview.next_review_at.is_(None)
        | (FlashcardReview.next_review_at <= func.now())
    )
    return join_cond, due_cond


@router.post("/flashcards/summary", response_model=FlashcardsSummaryResponse)
async def flashcards_summary(
    payload: FlashcardsSummaryRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> FlashcardsSummaryResponse:
    """Cuántas flashcards generadas hasta ahora "tocan hoy" vs. el total (SP-11)."""
    base_filters = [Flashcard.course_id == payload.course_id]
    if payload.topic:
        base_filters.append(func.lower(Flashcard.topic) == payload.topic.strip().lower())

    total_count = await db.scalar(
        select(func.count()).select_from(Flashcard).where(*base_filters)
    )

    join_cond, due_cond = _flashcard_due_filter(payload.user_id)
    due_count = await db.scalar(
        select(func.count())
        .select_from(Flashcard)
        .outerjoin(FlashcardReview, join_cond)
        .where(*base_filters)
        .where(due_cond)
    )

    return FlashcardsSummaryResponse(due_count=due_count or 0, total_count=total_count or 0)


@router.post("/flashcards/due", response_model=FlashcardsDueResponse)
async def flashcards_due(
    payload: FlashcardsDueRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> FlashcardsDueResponse:
    """Flashcards ya generadas que "tocan hoy", más vencidas primero (SP-11).

    No llama al LLM — sirve del banco ya persistido por /generate. El
    frontend completa con generación nueva solo si esto no alcanza para
    la cantidad pedida (ver QuizPanel.jsx).
    """
    join_cond, due_cond = _flashcard_due_filter(payload.user_id)
    stmt = (
        select(Flashcard, FlashcardReview.next_review_at)
        .outerjoin(FlashcardReview, join_cond)
        .where(Flashcard.course_id == payload.course_id)
        .where(due_cond)
    )
    if payload.topic:
        stmt = stmt.where(func.lower(Flashcard.topic) == payload.topic.strip().lower())
    stmt = stmt.order_by(FlashcardReview.next_review_at.asc().nulls_last()).limit(payload.limit)

    rows = (await db.execute(stmt)).all()

    questions = [
        QuizQuestion(
            id=str(fc.id),
            question_type="flashcard",
            question=fc.question,
            options=[],
            correct_index=-1,
            explanation=fc.explanation,
            source_filename=fc.source_filename or "",
            source_document_id=fc.source_document_id,
        )
        for fc, _next_review_at in rows
    ]

    return FlashcardsDueResponse(course_id=payload.course_id, questions=questions)


@router.post("/flashcards/review-batch", response_model=FlashcardsReviewBatchResponse)
async def flashcards_review_batch(
    payload: FlashcardsReviewBatchRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> FlashcardsReviewBatchResponse:
    """Aplica SM-2 sobre el resultado de autoevaluación de una sesión de flashcards (SP-11).

    Se llama una sola vez al final de la sesión (mismo patrón que
    /attempts + /errors), no por-tarjeta.
    """
    now = datetime.now(timezone.utc)

    parsed_ids: dict[str, uuid.UUID] = {}
    for item in payload.reviews:
        try:
            parsed_ids[item.flashcard_id] = uuid.UUID(item.flashcard_id)
        except ValueError:
            continue

    if not parsed_ids:
        return FlashcardsReviewBatchResponse(updated=0)

    # Solo se aplica repaso sobre flashcards que realmente pertenecen a este
    # curso — flashcard_id es dato del cliente, no confiar ciegamente.
    valid_rows = await db.execute(
        select(Flashcard.id).where(
            Flashcard.id.in_(parsed_ids.values()),
            Flashcard.course_id == payload.course_id,
        )
    )
    valid_ids = {row.id for row in valid_rows.all()}

    updated = 0
    for item in payload.reviews:
        flashcard_id = parsed_ids.get(item.flashcard_id)
        if flashcard_id is None or flashcard_id not in valid_ids:
            continue

        review_stmt = select(FlashcardReview).where(
            FlashcardReview.flashcard_id == flashcard_id,
            FlashcardReview.user_id == payload.user_id,
        )
        review = (await db.execute(review_stmt)).scalar_one_or_none()
        if review is None:
            # Los defaults de mapped_column solo se aplican al hacer INSERT
            # (flush), no al construir el objeto — _apply_sm2 los necesita
            # en memoria YA, antes del commit, así que se setean acá a mano
            # (mismos valores que los defaults declarados en el modelo).
            review = FlashcardReview(
                flashcard_id=flashcard_id,
                user_id=payload.user_id,
                ease_factor=2.5,
                interval_days=0,
                repetitions=0,
            )
            db.add(review)

        _apply_sm2(review, item.knew_it, now)
        updated += 1

    await db.commit()
    return FlashcardsReviewBatchResponse(updated=updated)


# ============================================================
# Racha de estudio — SP-16 (#354)
# ============================================================

def _compute_streak(active_dates: set[date], today: date) -> int:
    """Días consecutivos de actividad terminando hoy o ayer.

    Si la actividad más reciente es de hace 2+ días, la racha se considera
    rota (0) — criterio de aceptación explícito ("se resetea si el alumno
    deja de usar el asistente un día"). Si hubo actividad hoy o ayer
    (el alumno puede no haber practicado TODAVÍA hoy y seguir con la racha
    viva desde ayer), cuenta hacia atrás mientras los días sean consecutivos.
    """
    if not active_dates:
        return 0

    most_recent = max(active_dates)
    if (today - most_recent).days > 1:
        return 0

    streak = 0
    cursor = most_recent
    while cursor in active_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


@router.post("/streak", response_model=StreakResponse)
async def study_streak(
    payload: StreakRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> StreakResponse:
    """Días consecutivos de actividad del alumno en el curso (SP-16, #354).

    Combina dos señales ya persistidas, sin tabla ni migración nueva:
    - QuizAttempt.user_id/course_id/created_at — anonimizado por PRIV-01
      cae solo del filtro (user_id queda NULL), mismo criterio que
      suggest_difficulty/list_quiz_attempts.
    - Message.created_at (role='user') join ChatSession.user_id/course_id.
      InteractionLog NO sirve acá: es anónimo por diseño (solo
      user_id_hash, sin user_id real) — no hay forma de agrupar por
      alumno. Sesiones multi-curso (ChatSession.course_id == 0, Feature B)
      se excluyen: no pertenecen a un curso puntual.
    """
    quiz_days_stmt = (
        select(func.date_trunc("day", QuizAttempt.created_at).label("day"))
        .where(
            QuizAttempt.user_id == payload.user_id,
            QuizAttempt.course_id == payload.course_id,
        )
        .distinct()
    )
    chat_days_stmt = (
        select(func.date_trunc("day", Message.created_at).label("day"))
        .join(ChatSession, Message.session_id == ChatSession.id)
        .where(
            ChatSession.user_id == payload.user_id,
            ChatSession.course_id == payload.course_id,
            Message.role == "user",
        )
        .distinct()
    )

    quiz_rows = (await db.execute(quiz_days_stmt)).all()
    chat_rows = (await db.execute(chat_days_stmt)).all()

    active_dates = {row.day.date() for row in quiz_rows} | {row.day.date() for row in chat_rows}

    today = datetime.now(timezone.utc).date()
    streak = _compute_streak(active_dates, today)

    return StreakResponse(current_streak=streak, practiced_today=today in active_dates)

