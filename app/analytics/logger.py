"""Logging anonimizado de interacciones con el asistente — DOC-D01."""

from __future__ import annotations

import hashlib
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InteractionLog

logger = logging.getLogger("nexusai.analytics")


def _hash_user_id(user_id: int) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()


async def log_interaction(
    db: AsyncSession,
    *,
    course_id: int,
    user_id: int,
    user_message_id: uuid.UUID | None,
    question: str,
    answer: str,
    chunks_retrieved: int,
    has_relevant_context: bool,
    is_multicourse: bool,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: float,
    endpoint: str,
) -> None:
    """Persiste una fila anonimizada en interaction_logs.

    Se llama al final de /messages y /stream, después del commit principal,
    como best-effort — los errores se loguean pero no propagan al cliente.
    """
    try:
        log = InteractionLog(
            course_id=course_id,
            user_id_hash=_hash_user_id(user_id),
            user_message_id=user_message_id,
            question_char_count=len(question),
            answer_char_count=len(answer),
            chunks_retrieved=chunks_retrieved,
            has_relevant_context=has_relevant_context,
            is_multicourse=is_multicourse,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            endpoint=endpoint,
        )
        db.add(log)
        await db.commit()
    except Exception as exc:
        logger.warning("log_interaction failed (non-fatal): %s", exc)
