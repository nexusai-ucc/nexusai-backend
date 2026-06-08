"""
Quiz Generator — Feature F.

POST /api/v1/quiz/generate
  Genera un quiz de práctica con preguntas de opción múltiple a partir
  del material indexado del curso. El LLM produce JSON estructurado:
  pregunta, 4 opciones, índice correcto, explicación + archivo fuente.

  Modos:
  - topic provisto → retrieve_context con esa consulta (chunks relevantes)
  - topic vacío    → chunks aleatorios del curso (variedad de temas)

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
from pydantic import BaseModel, Field, ValidationError
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

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class QuizRequest(BaseModel):
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    topic: Optional[str] = Field(default=None, max_length=200)
    num_questions: int = Field(default=5, ge=1, le=10)
    course_ids: Optional[List[int]] = Field(default=None)


class QuizQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    options: List[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=1, max_length=600)
    source_filename: str = Field(default="")


class QuizResponse(BaseModel):
    course_id: int
    topic: Optional[str]
    questions: List[QuizQuestion]


# ============================================================
# Helpers
# ============================================================

async def _sample_chunks_for_quiz(
    db: AsyncSession,
    course_id: int,
    limit: int = 12,
    course_ids: list[int] | None = None,
) -> list[tuple[str, str]]:
    """Devuelve [(filename, content)] de chunks aleatorios indexed del/los curso/s.

    No usa embeddings — random sample. Útil cuando el alumno NO especifica
    topic y queremos variedad de temas en el quiz.
    """
    ids = [i for i in (course_ids or [course_id]) if i > 0] or [course_id]
    stmt = (
        select(Document.filename, Chunk.content)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.course_id.in_(ids))
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
) -> list[dict[str, str]]:
    """Arma los mensajes para el LLM con instrucciones estrictas de JSON."""
    schema_hint = (
        "Devolvé EXCLUSIVAMENTE un JSON válido con esta forma exacta:\n"
        '{\n'
        '  "questions": [\n'
        '    {\n'
        '      "question": "<texto de la pregunta>",\n'
        '      "options": ["<opción A>", "<opción B>", "<opción C>", "<opción D>"],\n'
        '      "correct_index": 0,\n'
        '      "explanation": "<por qué la opción correcta es correcta>",\n'
        '      "source_filename": "<nombre del archivo del que sale la respuesta>"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "O, si el material no contiene suficiente contenido directo sobre el tema:\n"
        '{"error": "insufficient_content", "detail": "El material del curso no contiene suficiente contenido sobre este tema para generar preguntas."}\n'
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

    topic_line = (
        f"Tema solicitado: {topic.strip()}\n\n"
        if topic and topic.strip()
        else "El alumno no especificó tema — variá temas para cubrir el material disponible.\n\n"
    )

    material_block = "\n\n".join(
        f'FRAGMENTO {i + 1} (de "{filename}"):\n{content[:600].strip()}'
        for i, (filename, content) in enumerate(chunks)
    )

    system = (
        "Sos un generador de quizzes académicos de NexusAI. "
        "Producís preguntas de opción múltiple en español, basadas estrictamente "
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
        f"Generá {num_questions} preguntas de opción múltiple de práctica.\n\n"
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

    # Resolver qué cursos usar (multi-curso si viene course_ids).
    effective_course_ids: list[int] | None = None
    if payload.course_ids:
        effective_course_ids = [i for i in payload.course_ids if i > 0] or None

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
                course_ids=effective_course_ids,
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
        # Modo variedad: sampling aleatorio del material del/los curso/s.
        chunks = await _sample_chunks_for_quiz(db, payload.course_id, limit=12, course_ids=effective_course_ids)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Este curso todavía no tiene material indexado para generar un quiz.",
        )

    # 2) Pedir al LLM la generación con JSON estricto.
    messages = _build_quiz_prompt(chunks, payload.num_questions, payload.topic)
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
    # (El LLM tiende a poner la correcta en index 0 — esto rompe ese patrón.)
    for q in questions:
        original_correct = q.options[q.correct_index]
        random.shuffle(q.options)
        q.correct_index = q.options.index(original_correct)

    return QuizResponse(
        course_id=payload.course_id,
        topic=payload.topic,
        questions=questions,
    )
