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
    """
    stmt = select(InteractionLog.created_at).where(
        InteractionLog.course_id == course_id,
        InteractionLog.created_at >= since,
    )
    result = await db.execute(stmt)
    rows = result.all()

    daily: dict[str, int] = {}
    for (created_at,) in rows:
        day = created_at.strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + 1
    return daily


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
