"""Analytics endpoints para el dashboard docente — DOC-D01.

GET /api/v1/analytics/course-stats?course_id=X&days=30
  Devuelve métricas agregadas de interacciones para un curso.
  Solo accesible con HMAC válido.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import InteractionLog
from app.db.session import get_db

router = APIRouter()


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
