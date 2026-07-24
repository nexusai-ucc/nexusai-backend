"""
Calendar Alerts router — CAL-02.

POST /api/v1/calendar/alerts/save          Upsert de alerta (0 = eliminar)
POST /api/v1/calendar/alerts/list          Alertas del alumno en el curso
POST /api/v1/calendar/alerts/due           Alertas vencidas para el cron PHP
POST /api/v1/calendar/alerts/mark-notified Marca alerta como notificada
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import CalendarAlert
from app.db.session import get_db

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class SaveAlertRequest(BaseModel):
    user_id: int = Field(gt=0)
    course_id: int = Field(gt=0)
    event_id: int = Field(gt=0)
    event_name: str = Field(max_length=200)
    event_timestamp: int
    days_before: int = Field(ge=0)


class SaveAlertResponse(BaseModel):
    id: str | None
    days_before: int


class AlertItem(BaseModel):
    event_id: int
    days_before: int
    notified: bool


class AlertsListRequest(BaseModel):
    user_id: int = Field(gt=0)
    course_id: int = Field(gt=0)


class AlertsListResponse(BaseModel):
    alerts: List[AlertItem]


class AlertsDueItem(BaseModel):
    id: str
    user_id: int
    event_name: str
    event_timestamp: int
    days_before: int


class AlertsDueResponse(BaseModel):
    alerts: List[AlertsDueItem]


class MarkNotifiedRequest(BaseModel):
    alert_id: str


class MarkNotifiedResponse(BaseModel):
    ok: bool


# ============================================================
# Endpoints
# ============================================================

@router.post("/save", response_model=SaveAlertResponse)
async def save_alert(
    payload: SaveAlertRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> SaveAlertResponse:
    """Upsert de alerta de calendario. days_before=0 elimina la alerta."""
    if payload.days_before == 0:
        await db.execute(
            delete(CalendarAlert).where(
                CalendarAlert.user_id == payload.user_id,
                CalendarAlert.event_id == payload.event_id,
            )
        )
        await db.commit()
        return SaveAlertResponse(id=None, days_before=0)

    stmt = (
        pg_insert(CalendarAlert)
        .values(
            id=uuid.uuid4(),
            user_id=payload.user_id,
            course_id=payload.course_id,
            event_id=payload.event_id,
            event_name=payload.event_name,
            event_timestamp=payload.event_timestamp,
            days_before=payload.days_before,
            notified=False,
        )
        .on_conflict_do_update(
            constraint="uq_calendar_alerts_user_event",
            set_={
                "days_before": payload.days_before,
                "event_name": payload.event_name,
                "event_timestamp": payload.event_timestamp,
                "notified": False,
            },
        )
        .returning(CalendarAlert.id, CalendarAlert.days_before)
    )

    result = await db.execute(stmt)
    await db.commit()
    row = result.fetchone()
    return SaveAlertResponse(id=str(row.id), days_before=row.days_before)


@router.post("/list", response_model=AlertsListResponse)
async def list_alerts(
    payload: AlertsListRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> AlertsListResponse:
    """Devuelve todas las alertas activas del alumno en el curso."""
    stmt = select(CalendarAlert).where(
        CalendarAlert.user_id == payload.user_id,
        CalendarAlert.course_id == payload.course_id,
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return AlertsListResponse(
        alerts=[
            AlertItem(event_id=r.event_id, days_before=r.days_before, notified=r.notified)
            for r in rows
        ]
    )


@router.post("/due", response_model=AlertsDueResponse)
async def alerts_due(
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> AlertsDueResponse:
    """Alertas cuyo momento de notificación ya llegó (para el cron PHP)."""
    stmt = select(CalendarAlert).where(
        CalendarAlert.days_before > 0,
        CalendarAlert.notified.is_(False),
        text(
            "to_timestamp(event_timestamp) - days_before * interval '1 day' <= now()"
        ),
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return AlertsDueResponse(
        alerts=[
            AlertsDueItem(
                id=str(r.id),
                user_id=r.user_id,
                event_name=r.event_name,
                event_timestamp=r.event_timestamp,
                days_before=r.days_before,
            )
            for r in rows
        ]
    )


@router.post("/mark-notified", response_model=MarkNotifiedResponse)
async def mark_notified(
    payload: MarkNotifiedRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> MarkNotifiedResponse:
    """Marca una alerta como ya notificada para que el cron no la reenvíe."""
    try:
        alert_uuid = uuid.UUID(payload.alert_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="alert_id inválido")

    await db.execute(
        update(CalendarAlert)
        .where(CalendarAlert.id == alert_uuid)
        .values(notified=True)
    )
    await db.commit()
    return MarkNotifiedResponse(ok=True)
