"""
Privacy — exportación y eliminación de datos personales (PRIV-01, issue #310).

`classes/privacy/provider.php` del plugin declaraba `null_provider` con el
argumento de que "el plugin no guarda datos personales en la DB de Moodle".
Cierto para Moodle, falso para este backend: acá SÍ vive el historial real
de un alumno (mensajes de chat, intentos y errores de quiz). Este router le
da al alumno una forma de pedir su propio historial o borrarlo.

GET    /api/v1/privacy/export?user_id=&course_id=
  Devuelve todo lo que el alumno mandó: mensajes de chat, intentos de quiz
  (los no anonimizados) y errores de quiz registrados.

DELETE /api/v1/privacy/data?user_id=&course_id=
  Borra ese historial. Distinto por tabla, a propósito:
    - messages/chat_sessions → DELETE real (nada del lado de Analytics
      depende de estas filas sobreviviendo: interaction_logs ya es anónimo
      y su FK a messages es ON DELETE SET NULL).
    - quiz_errors → DELETE real (no hay ningún agregado de Analytics que
      lea quiz_errors directamente).
    - quiz_attempts → ANONIMIZA in-place (user_id=NULL, deleted_at=now()),
      nunca DELETE. Su columna `score` alimenta en vivo el histograma de
      Analytics del docente (_get_quiz_score_distribution en
      app/admin/router.py), que filtra por course_id/created_at — nunca por
      user_id. Un DELETE le bajaría el promedio/total al curso cada vez que
      alguien pide el borrado. Anonimizar conserva la métrica intacta.

Seguridad: igual que el resto de los routers, este endpoint confía en el
`user_id` que manda el plugin dentro del body/query firmado con HMAC — la
verificación real de "sos vos mismo" pasa en PHP con $USER->id (ver
classes/external/privacy_export.php / privacy_delete.php del plugin). El
plugin nunca debe exponer un parámetro user_id editable por el alumno.
"""

from __future__ import annotations

from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.auth.hmac import verify_hmac
from app.db.models import ChatSession, Message, QuizAttempt, QuizError
from app.db.session import get_db

router = APIRouter()


# ============================================================
# Schemas
# ============================================================

class ExportedMessage(BaseModel):
    session_id: UUID
    role: str
    content: str
    created_at: datetime


class ExportedQuizAttempt(BaseModel):
    id: UUID
    question_type: Optional[str]
    difficulty: str
    topic: Optional[str]
    total_questions: int
    correct_answers: int
    score: float
    created_at: datetime


class ExportedQuizError(BaseModel):
    id: UUID
    question_type: str
    question: str
    explanation: str
    user_answer: Optional[str]
    ai_feedback: Optional[str]
    ai_score: Optional[float]
    created_at: datetime


class PrivacyExportResponse(BaseModel):
    user_id: int
    course_id: int
    messages: List[ExportedMessage]
    quiz_attempts: List[ExportedQuizAttempt]
    quiz_errors: List[ExportedQuizError]


class PrivacyDeleteResponse(BaseModel):
    messages_deleted: int
    quiz_errors_deleted: int
    quiz_attempts_anonymized: int


def _validate_ids(user_id: int, course_id: int) -> None:
    if user_id <= 0 or course_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id and course_id must be positive",
        )


# ============================================================
# GET /export
# ============================================================

@router.get("/export", response_model=PrivacyExportResponse)
async def export_personal_data(
    user_id: int,
    course_id: int,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> PrivacyExportResponse:
    """Exporta todo el historial personal del alumno en un curso."""
    _validate_ids(user_id, course_id)

    sessions_result = await db.execute(
        select(ChatSession).where(
            ChatSession.user_id == user_id,
            ChatSession.course_id == course_id,
        )
    )
    sessions = sessions_result.scalars().all()
    messages = [
        ExportedMessage(
            session_id=session.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
        )
        for session in sessions
        for m in session.messages
    ]

    attempts_result = await db.execute(
        select(QuizAttempt)
        .where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.course_id == course_id,
            QuizAttempt.deleted_at.is_(None),
        )
        .order_by(QuizAttempt.created_at.desc())
    )
    quiz_attempts = [
        ExportedQuizAttempt(
            id=a.id,
            question_type=a.question_type,
            difficulty=a.difficulty,
            topic=a.topic,
            total_questions=a.total_questions,
            correct_answers=a.correct_answers,
            score=a.score,
            created_at=a.created_at,
        )
        for a in attempts_result.scalars().all()
    ]

    errors_result = await db.execute(
        select(QuizError)
        .where(QuizError.user_id == user_id, QuizError.course_id == course_id)
        .order_by(QuizError.created_at.desc())
    )
    quiz_errors = [
        ExportedQuizError(
            id=e.id,
            question_type=e.question_type,
            question=e.question,
            explanation=e.explanation,
            user_answer=e.user_answer,
            ai_feedback=e.ai_feedback,
            ai_score=e.ai_score,
            created_at=e.created_at,
        )
        for e in errors_result.scalars().all()
    ]

    return PrivacyExportResponse(
        user_id=user_id,
        course_id=course_id,
        messages=messages,
        quiz_attempts=quiz_attempts,
        quiz_errors=quiz_errors,
    )


# ============================================================
# DELETE /data
# ============================================================

@router.delete("/data", response_model=PrivacyDeleteResponse)
async def delete_personal_data(
    user_id: int,
    course_id: int,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> PrivacyDeleteResponse:
    """Borra el historial personal del alumno en un curso.

    Ver docstring del módulo: messages/quiz_errors se borran de verdad,
    quiz_attempts se anonimiza in-place para no romper Analytics.
    """
    _validate_ids(user_id, course_id)

    # Contamos los mensajes ANTES de borrar chat_sessions: el DELETE de
    # ChatSession cascadea a Message a nivel de DB (ON DELETE CASCADE), pero
    # eso no se refleja en el rowcount del DELETE de ChatSession — hay que
    # borrar Message explícitamente primero para tener el conteo real.
    messages_stmt = delete(Message).where(
        Message.session_id.in_(
            select(ChatSession.id).where(
                ChatSession.user_id == user_id,
                ChatSession.course_id == course_id,
            )
        )
    )
    messages_result = await db.execute(messages_stmt)
    messages_deleted = messages_result.rowcount or 0

    await db.execute(
        delete(ChatSession).where(
            ChatSession.user_id == user_id,
            ChatSession.course_id == course_id,
        )
    )

    errors_result = await db.execute(
        delete(QuizError).where(
            QuizError.user_id == user_id,
            QuizError.course_id == course_id,
        )
    )
    quiz_errors_deleted = errors_result.rowcount or 0

    attempts_result = await db.execute(
        update(QuizAttempt)
        .where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.course_id == course_id,
            QuizAttempt.deleted_at.is_(None),
        )
        .values(user_id=None, deleted_at=func.now())
    )
    quiz_attempts_anonymized = attempts_result.rowcount or 0

    await db.commit()

    return PrivacyDeleteResponse(
        messages_deleted=messages_deleted,
        quiz_errors_deleted=quiz_errors_deleted,
        quiz_attempts_anonymized=quiz_attempts_anonymized,
    )
