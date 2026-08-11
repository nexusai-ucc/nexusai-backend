"""
Endpoints de monitoreo y analytics para docentes — BACK-12 / ANALYTICS-01.

GET /api/v1/admin/usage?course_id=X
  Devuelve:
    - total_tokens del curso (suma de prompt + completion en mensajes del LLM)
    - total_messages (solo mensajes tipo 'assistant', es decir, respuestas del LLM)
    - desglose por sesión: session_id, tokens, message_count

GET /api/v1/admin/analytics?course_id=X&days=30
  Dashboard agregado de métricas de un curso para el docente (ANALYTICS-01):
    - top_queries: hasta 10 preguntas más frecuentes (agrupadas por texto
      normalizado — bag-of-words literal, reusa la misma query que
      analytics/faq-topics vía app.analytics.service, sin llamar al LLM
      porque acá no hace falta sintetizar temas, solo listar lo más pedido)
    - daily_message_counts: serie temporal {date, message_count} de los
      últimos N días (reusa app.analytics.service, misma fuente que
      analytics/course-stats: interaction_logs, DOC-D01)
    - quiz_score_distribution: histograma de puntajes sobre quiz_attempts
      (ANALYTICS-01) — a diferencia de quiz_errors, que solo registra
      respuestas incorrectas, quiz_attempts persiste el score de CADA
      intento completo vía POST /api/v1/quiz/attempts
    - gaps_ratio: gaps detectados (unanswered_questions, DOC-D03) sobre
      preguntas respondidas (interaction_logs) en el mismo período

Ambos endpoints solo requieren HMAC válido — la validación de rol docente
(capability local/nexusai:viewanalytics) se hace del lado del plugin PHP
antes de llamar al backend, como el resto de la API (arquitectura Hybrid
PHP Proxy, ver app/auth/hmac.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.service import get_daily_message_counts, get_distinct_topic_count, get_top_questions
from app.auth.hmac import verify_hmac
from app.db.models import ChatSession, InteractionLog, Message, QuizAttempt, UnansweredQuestion
from app.db.session import get_db

router = APIRouter()

_TOP_QUERIES_LIMIT = 10
_QUIZ_SCORE_BUCKETS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
# Límites de score [lo, hi) por bucket — score=0.2 exacto cae en "20-40", no
# en "0-20" (idéntico al mapeo previo vía idx = int(pct // 20)). El último
# límite (1.01) es solo para incluir score=1.0 en "80-100" con un `<` uniforme.
_QUIZ_SCORE_BUCKET_BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]


# ============================================================
# Schemas de respuesta
# ============================================================

class SessionUsage(BaseModel):
    session_id: UUID
    tokens: int
    message_count: int


class CourseUsage(BaseModel):
    course_id: int
    total_tokens: int
    total_messages: int
    sessions: List[SessionUsage]


class TopQuery(BaseModel):
    question: str
    count: int


class DailyMessageCount(BaseModel):
    date: str
    message_count: int


class QuizScoreBucket(BaseModel):
    range: str
    count: int


class QuizScoreDistribution(BaseModel):
    total_attempts: int
    average_score: float
    buckets: List[QuizScoreBucket]


class GapsRatio(BaseModel):
    gaps_detected: int
    questions_answered: int
    ratio: float


class CourseAnalytics(BaseModel):
    course_id: int
    period_days: int
    top_queries: List[TopQuery]
    daily_message_counts: List[DailyMessageCount]
    quiz_score_distribution: QuizScoreDistribution
    gaps_ratio: GapsRatio
    topics_consulted: int


# ============================================================
# Endpoint
# ============================================================

@router.get("/usage", response_model=CourseUsage)
async def get_usage(
    course_id: int,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> CourseUsage:
    """Retorna el consumo de tokens del LLM para un curso.

    Solo cuenta mensajes de role='assistant' (los que generó el LLM).
    Los mensajes de role='user' no consumen tokens de completions.
    Usa COALESCE para manejar correctamente filas anteriores a la migración 003
    que tienen token_count_prompt = NULL.
    """
    if course_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="course_id debe ser positivo",
        )

    # ----- Totales del curso -----
    total_stmt = (
        select(
            func.coalesce(
                func.sum(
                    func.coalesce(Message.token_count_prompt, 0)
                    + func.coalesce(Message.token_count_completion, 0)
                ),
                0,
            ).label("total_tokens"),
            func.count(Message.id).label("total_messages"),
        )
        .join(ChatSession, Message.session_id == ChatSession.id)
        .where(ChatSession.course_id == course_id)
        .where(Message.role == "assistant")
    )
    total_result = await db.execute(total_stmt)
    total_row = total_result.one()

    # ----- Desglose por sesión -----
    sessions_stmt = (
        select(
            ChatSession.id.label("session_id"),
            func.coalesce(
                func.sum(
                    func.coalesce(Message.token_count_prompt, 0)
                    + func.coalesce(Message.token_count_completion, 0)
                ),
                0,
            ).label("tokens"),
            func.count(Message.id).label("message_count"),
        )
        .join(Message, Message.session_id == ChatSession.id)
        .where(ChatSession.course_id == course_id)
        .where(Message.role == "assistant")
        .group_by(ChatSession.id)
        .order_by(ChatSession.updated_at.desc())
    )
    sessions_result = await db.execute(sessions_stmt)
    sessions_rows = sessions_result.all()

    return CourseUsage(
        course_id=course_id,
        total_tokens=int(total_row.total_tokens),
        total_messages=int(total_row.total_messages),
        sessions=[
            SessionUsage(
                session_id=row.session_id,
                tokens=int(row.tokens),
                message_count=int(row.message_count),
            )
            for row in sessions_rows
        ],
    )


# ============================================================
# Analytics — ANALYTICS-01
# ============================================================

async def _get_quiz_score_distribution(
    db: AsyncSession, course_id: int, since: datetime
) -> QuizScoreDistribution:
    """Histograma de puntajes armado en una sola query de agregados (UX-16).

    Antes traía cada QuizAttempt.score del período a Python para bucketearlo
    ahí — con muchos alumnos rindiendo muchos quizzes esa lista podía crecer
    sin límite. `FILTER` calcula los 5 buckets + el promedio en el motor,
    sin importar cuántos intentos haya.
    """
    bucket_cols = [
        func.count()
        .filter(QuizAttempt.score >= lo, QuizAttempt.score < hi)
        .label(f"bucket_{i}")
        for i, (lo, hi) in enumerate(
            zip(_QUIZ_SCORE_BUCKET_BOUNDS, _QUIZ_SCORE_BUCKET_BOUNDS[1:])
        )
    ]
    stmt = select(
        func.count().label("total"),
        func.avg(QuizAttempt.score).label("avg_score"),
        *bucket_cols,
    ).where(
        QuizAttempt.course_id == course_id,
        QuizAttempt.created_at >= since,
    )
    row = (await db.execute(stmt)).one()

    total = int(row.total)
    average = round(float(row.avg_score), 4) if total else 0.0
    bucket_counts = [int(getattr(row, f"bucket_{i}")) for i in range(len(_QUIZ_SCORE_BUCKETS))]

    return QuizScoreDistribution(
        total_attempts=total,
        average_score=average,
        buckets=[
            QuizScoreBucket(range=b, count=c) for b, c in zip(_QUIZ_SCORE_BUCKETS, bucket_counts)
        ],
    )


async def _get_gaps_ratio(db: AsyncSession, course_id: int, since: datetime) -> GapsRatio:
    gaps_stmt = (
        select(func.count())
        .select_from(UnansweredQuestion)
        .where(
            UnansweredQuestion.course_id == course_id,
            UnansweredQuestion.created_at >= since,
        )
    )
    gaps_detected = (await db.execute(gaps_stmt)).scalar_one()

    answered_stmt = (
        select(func.count())
        .select_from(InteractionLog)
        .where(
            InteractionLog.course_id == course_id,
            InteractionLog.created_at >= since,
        )
    )
    questions_answered = (await db.execute(answered_stmt)).scalar_one()

    ratio = round(gaps_detected / questions_answered, 4) if questions_answered else 0.0

    return GapsRatio(
        gaps_detected=gaps_detected,
        questions_answered=questions_answered,
        ratio=ratio,
    )


@router.get("/analytics", response_model=CourseAnalytics)
async def get_course_analytics(
    course_id: int = Query(..., gt=0),
    days: int = Query(default=30, ge=1, le=365),
    _body: Annotated[bytes, Depends(verify_hmac)] = b"",
    db: AsyncSession = Depends(get_db),
) -> CourseAnalytics:
    """Dashboard agregado de métricas de un curso para el docente (ANALYTICS-01)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    top_rows = await get_top_questions(db, course_id, since, limit=_TOP_QUERIES_LIMIT)
    top_queries = [TopQuery(question=r.question, count=r.count) for r in top_rows]

    daily = await get_daily_message_counts(db, course_id, since)
    daily_message_counts = [
        DailyMessageCount(date=d, message_count=c) for d, c in sorted(daily.items())
    ]

    quiz_score_distribution = await _get_quiz_score_distribution(db, course_id, since)
    gaps_ratio = await _get_gaps_ratio(db, course_id, since)
    topics_consulted = await get_distinct_topic_count(db, course_id, since)

    return CourseAnalytics(
        course_id=course_id,
        period_days=days,
        top_queries=top_queries,
        daily_message_counts=daily_message_counts,
        quiz_score_distribution=quiz_score_distribution,
        gaps_ratio=gaps_ratio,
        topics_consulted=topics_consulted,
    )
