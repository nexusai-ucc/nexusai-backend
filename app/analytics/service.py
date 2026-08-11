"""Queries de agregación compartidas entre analytics/router.py y admin/router.py.

Extraído para que ANALYTICS-01 (dashboard docente en /admin/analytics) pueda
reusar la misma serie temporal y el mismo agrupado de preguntas que ya usan
/analytics/course-stats y /analytics/faq-topics, sin duplicar las queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InteractionLog, Message


class QuestionCount(NamedTuple):
    question: str
    count: int


async def get_daily_message_counts(
    db: AsyncSession,
    course_id: int,
    since: datetime,
) -> dict[str, int]:
    """Cantidad de interacciones por día (fecha UTC) desde `since`.

    Basado en interaction_logs (DOC-D01), que registra una fila por cada
    interacción con el asistente — un proxy directo de "mensajes" sin
    depender de que el logging best-effort haya corrido para cada mensaje
    de la tabla `messages`.

    Agrupa en SQL (UX-16) en vez de traer cada fila a Python — con mucha
    actividad en el curso, `days=365` podía significar miles de filas
    viajando solo para contarlas de a una.
    """
    day = func.date_trunc("day", InteractionLog.created_at, "UTC").label("day")
    stmt = (
        select(day, func.count().label("count"))
        .where(
            InteractionLog.course_id == course_id,
            InteractionLog.created_at >= since,
        )
        .group_by(day)
    )
    result = await db.execute(stmt)

    return {row.day.strftime("%Y-%m-%d"): int(row.count) for row in result.all()}


async def get_top_questions(
    db: AsyncSession,
    course_id: int,
    since: datetime,
    limit: int,
) -> list[QuestionCount]:
    """Preguntas de alumnos agrupadas por texto normalizado, más frecuentes primero.

    interaction_logs no guarda el texto de la pregunta (anonimizado), así
    que hay que joinear con el mensaje de usuario real vía user_message_id.
    Agrupar por `lower(trim(content))` es el mismo bag-of-words literal que
    usa gaps/router.py para deduplicar preguntas iguales escritas distinto.
    """
    norm_question = func.lower(func.trim(Message.content))
    stmt = (
        select(
            norm_question.label("question"),
            func.count().label("count"),
        )
        .select_from(InteractionLog)
        .join(Message, Message.id == InteractionLog.user_message_id)
        .where(InteractionLog.course_id == course_id)
        .where(InteractionLog.created_at >= since)
        .where(Message.role == "user")
        .group_by(norm_question)
        .order_by(func.count().desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [QuestionCount(question=row.question, count=int(row.count)) for row in result.all()]


async def get_distinct_topic_count(
    db: AsyncSession,
    course_id: int,
    since: datetime,
) -> int:
    """Cantidad de preguntas distintas (texto normalizado) en el período.

    Mismo agrupado que get_top_questions (lower(trim(content))), pero
    cuenta grupos distintos en vez de traer las N más frecuentes con su
    texto — COUNT(DISTINCT ...) sin LLM, para el stat-card "Temas
    consultados" (RDS-06).
    """
    norm_question = func.lower(func.trim(Message.content))
    stmt = (
        select(func.count(func.distinct(norm_question)))
        .select_from(InteractionLog)
        .join(Message, Message.id == InteractionLog.user_message_id)
        .where(InteractionLog.course_id == course_id)
        .where(InteractionLog.created_at >= since)
        .where(Message.role == "user")
    )
    result = await db.execute(stmt)
    return int(result.scalar_one())
