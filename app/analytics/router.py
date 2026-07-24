"""Analytics endpoints para el dashboard docente — DOC-D01 / DOC-D02.

GET /api/v1/analytics/course-stats?course_id=X&days=30
  Devuelve métricas agregadas de interacciones para un curso.
  Solo accesible con HMAC válido.

POST /api/v1/analytics/faq-topics
  Agrupa las preguntas más frecuentes de los alumnos por tema (DOC-D02).
  No hay clustering semántico en el schema, así que se agrupan primero por
  texto normalizado (igual que gaps/router.py) y después se le pide a un
  LLM que sintetice esos grupos en temas — el conteo por tema siempre se
  recalcula en Python a partir de la asignación del LLM, nunca se confía
  en el LLM para los números (mismo principio que las sugerencias de
  repaso de quiz/router.py).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import InteractionLog, Message
from app.db.session import get_db
from app.providers.llm import LLMProvider, get_llm_provider

logger = logging.getLogger("nexusai.analytics")

router = APIRouter()

_FAQ_SAMPLE_LIMIT = 40
_FAQ_MAX_TOPICS = 8


class DailyCount(BaseModel):
    date: str
    interactions: int


class CourseStats(BaseModel):
    course_id: int
    period_days: int
    total_interactions: int
    unique_users: int
    grounding_rate: float
    avg_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    daily_breakdown: list[DailyCount]


@router.get("/course-stats", response_model=CourseStats)
async def course_stats(
    course_id: int = Query(..., gt=0),
    days: int = Query(default=30, ge=1, le=365),
    _body: Annotated[bytes, Depends(verify_hmac)] = b"",
    db: AsyncSession = Depends(get_db),
) -> CourseStats:
    """Métricas agregadas de interacciones para un curso en los últimos N días."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows_result = await db.execute(
        select(InteractionLog)
        .where(
            InteractionLog.course_id == course_id,
            InteractionLog.created_at >= since,
        )
        .order_by(InteractionLog.created_at)
    )
    rows = rows_result.scalars().all()

    if not rows:
        return CourseStats(
            course_id=course_id,
            period_days=days,
            total_interactions=0,
            unique_users=0,
            grounding_rate=0.0,
            avg_latency_ms=0.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            daily_breakdown=[],
        )

    total = len(rows)
    unique_users = len({r.user_id_hash for r in rows})
    grounded = sum(1 for r in rows if r.has_relevant_context)
    grounding_rate = round(grounded / total, 3)
    avg_latency = round(sum(r.latency_ms for r in rows) / total, 1)
    total_prompt = sum(r.prompt_tokens or 0 for r in rows)
    total_completion = sum(r.completion_tokens or 0 for r in rows)

    # Breakdown diario — agrupa por fecha UTC
    daily: dict[str, int] = {}
    for r in rows:
        day = r.created_at.strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + 1

    daily_breakdown = [DailyCount(date=d, interactions=c) for d, c in sorted(daily.items())]

    return CourseStats(
        course_id=course_id,
        period_days=days,
        total_interactions=total,
        unique_users=unique_users,
        grounding_rate=grounding_rate,
        avg_latency_ms=avg_latency,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        daily_breakdown=daily_breakdown,
    )


# ============================================================
# FAQ agrupado por tema — DOC-D02
# ============================================================

class FaqTopicsRequest(BaseModel):
    course_id: int = Field(gt=0)
    days: int = Field(default=30, ge=1, le=365)


class FaqTopic(BaseModel):
    topic: str
    count: int
    example_questions: List[str]


class FaqTopicsResponse(BaseModel):
    course_id: int
    days: int
    total_questions: int
    topics: List[FaqTopic]


@router.post("/faq-topics", response_model=FaqTopicsResponse)
async def faq_topics(
    payload: FaqTopicsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
) -> FaqTopicsResponse:
    """Preguntas más frecuentes de los alumnos, agrupadas por tema (DOC-D02)."""
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    # 1) Agrupar por pregunta normalizada, uniendo interaction_logs con el
    # mensaje de usuario real para recuperar el texto (interaction_logs no
    # guarda el texto de la pregunta, solo métricas — ver InteractionLog).
    norm_question = func.lower(func.trim(Message.content))
    stmt = (
        select(
            norm_question.label("question"),
            func.count().label("count"),
        )
        .select_from(InteractionLog)
        .join(Message, Message.id == InteractionLog.user_message_id)
        .where(InteractionLog.course_id == payload.course_id)
        .where(InteractionLog.created_at >= since)
        .where(Message.role == "user")
        .group_by(norm_question)
        .order_by(func.count().desc())
        .limit(_FAQ_SAMPLE_LIMIT)
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return FaqTopicsResponse(
            course_id=payload.course_id,
            days=payload.days,
            total_questions=0,
            topics=[],
        )

    total_questions = sum(int(row.count) for row in rows)

    # 2) Un único LLM call: agrupa y etiqueta por tema. El conteo y el orden
    # se recalculan en Python a partir de los índices que devuelve el LLM —
    # nunca se confía en el LLM para los números.
    questions_block = "\n".join(
        f'{i}. "{row.question}" (preguntada {row.count} {"vez" if row.count == 1 else "veces"})'
        for i, row in enumerate(rows)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Sos un asistente que ayuda a un docente a entender qué preguntan sus "
                "alumnos. Te paso una lista numerada de preguntas frecuentes de un curso, "
                "cada una con la cantidad de veces que se hizo. Agrupalas por tema/subtema "
                f"en como máximo {_FAQ_MAX_TOPICS} grupos. Tu salida es JSON.\n\n"
                "Devolvé EXCLUSIVAMENTE un JSON con esta forma exacta:\n"
                '{"topics": [{"label": "<tema en 3-6 palabras>", "question_indices": [0, 2, 5]}]}\n'
                "- label: nombrá el tema/subtema concreto que agrupa esas preguntas.\n"
                "- question_indices: los índices (de la lista numerada) que pertenecen a ese tema.\n"
                "- Cada índice de la lista debe aparecer en como máximo un grupo.\n"
                "- No inventes preguntas ni cambies su texto — solo agrupá y etiquetá."
            ),
        },
        {
            "role": "user",
            "content": questions_block,
        },
    ]

    try:
        result_llm = await llm.chat_completion(
            messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as exc:
        logger.error("FAQ topics LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudieron agrupar las preguntas frecuentes en este momento. Intentá de nuevo.",
        ) from exc

    raw = result_llm.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    try:
        parsed = json.loads(raw)
        raw_topics = parsed.get("topics", [])
        if not isinstance(raw_topics, list):
            raise ValueError("'topics' no es una lista")
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.error("FAQ topics JSON parse failed. Raw: %.300s", raw)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudieron procesar los temas frecuentes. Intentá de nuevo.",
        ) from exc

    topics: list[FaqTopic] = []
    for item in raw_topics:
        label = str(item.get("label") or "").strip()
        indices = item.get("question_indices")
        if not label or not isinstance(indices, list):
            continue
        assigned = [rows[i] for i in indices if isinstance(i, int) and 0 <= i < len(rows)]
        if not assigned:
            continue
        topics.append(
            FaqTopic(
                topic=label,
                count=sum(int(r.count) for r in assigned),
                example_questions=[r.question for r in assigned[:3]],
            )
        )

    topics.sort(key=lambda t: t.count, reverse=True)

    return FaqTopicsResponse(
        course_id=payload.course_id,
        days=payload.days,
        total_questions=total_questions,
        topics=topics,
    )
