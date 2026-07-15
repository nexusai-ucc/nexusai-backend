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

import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import Chunk, Document, QuizError
from app.db.session import get_db
from app.documents.retriever import retrieve_context
from app.gaps.recorder import WEAK_MATCH_THRESHOLD
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

logger = logging.getLogger("nexusai.quiz")

# Minimum cosine similarity for a chunk to count as covering a topic.
# Higher than WEAK_MATCH_THRESHOLD (0.4) to reject spurious cross-lingual matches.
QUIZ_TOPIC_MIN_SIMILARITY = 0.5

_VALID_QUESTION_TYPES = {"multiple_choice", "true_false", "open", "mix", "flashcard"}
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
    question_type: str = Field(default="multiple_choice")
    question: str = Field(min_length=1, max_length=500)
    options: List[str] = Field(default=[])   # 4 for MC, 2 for T/F, [] for open
    correct_index: int = Field(default=-1, ge=-1, le=3)  # -1 for open questions
    explanation: str = Field(min_length=1, max_length=1500)
    source_filename: str = Field(default="")
    source_document_id: Optional[str] = Field(default=None)  # filled after generation; Document.id is a UUID


class QuizResponse(BaseModel):
    course_id: int
    topic: Optional[str]
    questions: List[QuizQuestion]


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


# ============================================================
# Helpers
# ============================================================

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

    # 2) Pedir al LLM la generación con JSON estricto.
    messages = _build_quiz_prompt(chunks, payload.num_questions, payload.topic, payload.question_type, payload.difficulty)
    try:
        result = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.6,
        )
    except Exception as exc:
        logger.error("Quiz LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar el quiz en este momento. Intentá de nuevo.",
        ) from exc

    # 3) Parsear y validar.
    raw = result.text.strip()
    # Defensa: si el LLM envuelve el JSON en ```json ... ```, sacar las cercas.
    if raw.startswith("```"):
        # quitar primera línea (```json o ```) y última (```)
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
    questions = questions[: payload.num_questions]

    # Shuffle de opciones para no dejar siempre la correcta en el mismo lugar.
    # Solo aplica a opción múltiple; T/F y preguntas abiertas no se modifican.
    for q in questions:
        if q.question_type == "multiple_choice" and len(q.options) == 4 and q.correct_index >= 0:
            original_correct = q.options[q.correct_index]
            random.shuffle(q.options)
            q.correct_index = q.options.index(original_correct)

    # Enriquecer preguntas con el document_id del archivo fuente (SP-10).
    # Permite al frontend ofrecer descarga directa del material relacionado.
    source_filenames = {q.source_filename for q in questions if q.source_filename}
    if source_filenames:
        doc_stmt = (
            select(Document.filename, Document.id)
            .where(
                Document.course_id == payload.course_id,
                Document.filename.in_(source_filenames),
            )
        )
        doc_rows = await db.execute(doc_stmt)
        doc_id_map: dict[str, str] = {row.filename: str(row.id) for row in doc_rows.all()}
        for q in questions:
            if q.source_filename and q.source_filename in doc_id_map:
                q.source_document_id = doc_id_map[q.source_filename]

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
    """Historial de errores del alumno en un curso, más recientes primero."""
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    stmt = (
        select(QuizError)
        .where(QuizError.course_id == payload.course_id)
        .where(QuizError.user_id == payload.user_id)
        .where(QuizError.created_at >= since)
        .order_by(desc(QuizError.created_at))
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
    return ErrorsListResponse(course_id=payload.course_id, total=len(items), items=items)


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
