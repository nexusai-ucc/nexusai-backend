"""Logging anonimizado de interacciones con el asistente — DOC-D01."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import InteractionLog

logger = logging.getLogger("nexusai.analytics")


def hash_user_id(user_id: int) -> str:
    """Hash SHA-256 de un user_id — reusado por otras tablas anónimas-por-diseño
    (message_feedback, ASIST-01) para poder hacer upsert sin guardar identidad."""
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
            user_id_hash=hash_user_id(user_id),
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


def log_moderation_block(
    *,
    endpoint: str,
    course_id: int,
    user_id: int | None,
    source: str,
    categories: list[str],
) -> None:
    """Loguea (structured JSON, mismo formato que chat/router.py) un bloqueo
    de contenido por la capa de moderación — ver app/shared/moderation.py.

    `user_id` es opcional: algunos endpoints (p. ej. forums.suggest_reply) no
    reciben el user_id del alumno en el payload.

    No persiste en DB: es un evento de seguridad, no una interacción exitosa,
    y no requiere una migración de schema para un piloto. Si más adelante se
    necesita un dashboard de estos eventos, agregar una tabla dedicada
    (ver InteractionLog) en vez de forzarlo en el modelo actual.
    """
    logger.info(
        json.dumps(
            {
                "event": "content_moderation_blocked",
                "endpoint": endpoint,
                "course_id": course_id,
                "user_id_hash": hash_user_id(user_id) if user_id is not None else None,
                "source": source,
                "categories": categories,
            },
            ensure_ascii=False,
        )
    )
