"""
Quiz Generator — Feature F.

POST /api/v1/quiz/generate
  Genera un quiz de práctica a partir del material indexado del curso.
  Soporta múltiples tipos de pregunta: opción múltiple, verdadero/falso,
  preguntas abiertas y mix. El LLM produce JSON estructurado.

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
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import Chunk, Document
from app.db.session import get_db
from app.documents.retriever import retrieve_context
from app.gaps.recorder import WEAK_MATCH_THRESHOLD
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

logger = logging.getLogger("nexusai.quiz")

# Minimum cosine similarity for a chunk to count as covering a topic.
# Higher than WEAK_MATCH_THRESHOLD (0.4) to reject spurious cross-lingual matches.
QUIZ_TOPIC_MIN_SIMILARITY = 0.5

_VALID_QUESTION_TYPES = {"multiple_choice", "true_false", "open", "mix"}

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

    @field_validator("question_type")
    @classmethod
    def check_question_type(cls, v: str) -> str:
        if v not in _VALID_QUESTION_TYPES:
            raise ValueError(f"question_type must be one of {_VALID_QUESTION_TYPES}")
        return v


class QuizQuestion(BaseModel):
    question_type: str = Field(default="multiple_choice")
    question: str = Field(min_length=1, max_length=500)
    options: List[str] = Field(default=[])   # 4 for MC, 2 for T/F, [] for open
    correct_index: int = Field(default=-1, ge=-1, le=3)  # -1 for open questions
    explanation: str = Field(min_length=1, max_length=1500)
    source_filename: str = Field(default="")
    source_document_id: Optional[int] = Field(default=None)  # filled after generation


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


def _build_quiz_prompt(
    chunks: list[tuple[str, str]],
    num_questions: int,
    topic: Optional[str],
    question_type: str = "multiple_choice",
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
    }.get(question_type, "opción múltiple")

    system = (
        f"Sos un generador de quizzes académicos de NexusAI. "
        f"Producís preguntas de {type_desc} en español, basadas estrictamente "
        "en el material académico del curso del alumno. Tu salida es JSON.\n\n"
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
    messages = _build_quiz_prompt(chunks, payload.num_questions, payload.topic, payload.question_type)
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
        doc_id_map: dict[str, int] = {row.filename: row.id for row in doc_rows.all()}
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
