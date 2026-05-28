"""
Gaps router — Feature G.

Endpoints para que el docente consulte qué preguntas de los alumnos NO
pudo responder el material indexado del curso. Es el feedback loop
pedagógico: "estos temas no están cubiertos por tu material".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import UnansweredQuestion
from app.db.session import get_db

router = APIRouter()


class GapsListRequest(BaseModel):
    course_id: int = Field(gt=0)
    days: int = Field(default=30, ge=1, le=365)
    limit: int = Field(default=20, ge=1, le=100)


class GapItem(BaseModel):
    question: str
    count: int
    last_asked_at: datetime
    avg_similarity: Optional[float] = None


class GapsListResponse(BaseModel):
    course_id: int
    days: int
    total: int
    items: List[GapItem]


@router.post("/list", response_model=GapsListResponse)
async def gaps_list(
    payload: GapsListRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> GapsListResponse:
    """Devuelve los gaps del curso agrupados por pregunta normalizada.

    Agrupa por `lower(trim(question))` para deduplicar preguntas
    semánticamente iguales escritas con casing distinto. Para deduplicación
    semántica más sofisticada (clustering por embedding) habría que correr
    un proceso aparte — esto es suficiente para el MVP.
    """
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    # Grouping por pregunta normalizada (lowercased, trimmed).
    norm_question = func.lower(func.trim(UnansweredQuestion.question))

    stmt = (
        select(
            norm_question.label("question"),
            func.count().label("count"),
            func.max(UnansweredQuestion.created_at).label("last_asked_at"),
            func.avg(UnansweredQuestion.max_similarity).label("avg_similarity"),
        )
        .where(UnansweredQuestion.course_id == payload.course_id)
        .where(UnansweredQuestion.created_at >= since)
        .group_by(norm_question)
        .order_by(desc("count"), desc("last_asked_at"))
        .limit(payload.limit)
    )

    result = await db.execute(stmt)
    rows = result.all()

    items = [
        GapItem(
            question=row.question,
            count=int(row.count),
            last_asked_at=row.last_asked_at,
            avg_similarity=float(row.avg_similarity) if row.avg_similarity is not None else None,
        )
        for row in rows
    ]

    return GapsListResponse(
        course_id=payload.course_id,
        days=payload.days,
        total=len(items),
        items=items,
    )
