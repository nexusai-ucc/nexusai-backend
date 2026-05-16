"""
Endpoints de monitoreo de uso de tokens — BACK-12.

GET /api/v1/admin/usage?course_id=X
  Devuelve:
    - total_tokens del curso (suma de prompt + completion en mensajes del LLM)
    - total_messages (solo mensajes tipo 'assistant', es decir, respuestas del LLM)
    - desglose por sesión: session_id, tokens, message_count

Solo accesible con HMAC válido (mismo esquema que el resto de la API).
"""

from __future__ import annotations

from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import ChatSession, Message
from app.db.session import get_db

router = APIRouter()


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
