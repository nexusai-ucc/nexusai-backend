"""Presupuesto de tokens por user_id usando ventanas fijas en Redis.

Complementa a rate_limit.py: ese limita CANTIDAD de requests. Este limita
CANTIDAD DE TOKENS consumidos — proxy directo del costo real del LLM. Un
usuario podría hacer pocas consultas pero muy largas (o con historial
extenso) y agotar presupuesto real sin llegar nunca al límite de requests;
este módulo cubre ese caso.

Dos operaciones separadas porque el costo en tokens de una request recién se
conoce DESPUÉS de llamar al LLM:

  - check_token_budget: se llama ANTES del LLM (igual que check_rate_limit).
    Lee el acumulado de la ventana vigente y corta con 429 si ya se alcanzó
    el límite. No incrementa nada — solo lee.
  - record_token_usage: se llama DESPUÉS del LLM, con los tokens reales
    consumidos (prompt + completion), y los suma al acumulado de la ventana.

Mismo esquema de bucket de ventana fija que rate_limit.py — ver ese archivo
para el razonamiento sobre epoch UTC vs. medianoche local, que aplica igual
acá.
"""

from __future__ import annotations

import logging
import time

import redis.asyncio as redis_async
from fastapi import HTTPException, status

logger = logging.getLogger("nexusai.token_budget")

_MESSAGES = {
    "hourly": (
        "Alcanzaste tu límite de {limit} tokens por hora. "
        "Esperá un momento y volvé a intentarlo."
    ),
    "daily": "Alcanzaste tu límite de {limit} tokens de hoy. Volvé a intentarlo mañana.",
}

# El alumno ve este texto tal cual (el frontend muestra el mensaje de nuestro
# propio limitador), así que sigue el idioma de la pregunta cuando es inglés
# — mismo criterio que rate_limit.py.
_MESSAGES_EN = {
    "hourly": (
        "You reached your limit of {limit} tokens for this hour. "
        "Wait a moment and try again."
    ),
    "daily": "You reached your limit of {limit} tokens for today. Try again tomorrow.",
}


def _bucket_key(user_id: int, window_sec: int) -> str:
    bucket = int(time.time()) // window_sec
    return f"nexusai:tokenbudget:{window_sec}:{user_id}:{bucket}"


async def check_token_budget(
    user_id: int,
    redis: redis_async.Redis,
    limit: int,
    window_sec: int,
    scope: str,
    language: str | None = None,
) -> None:
    """Lanza HTTP 429 si user_id ya consumió `limit` tokens (o más) en la ventana vigente.

    Args:
        user_id: ID del usuario de Moodle (del payload, no de la sesión HTTP).
        redis: Cliente Redis async (inyectado como FastAPI Dependency).
        limit: Techo de tokens permitidos en la ventana (varía por rol —
            ver token_budget_student_*/token_budget_teacher_* en Settings).
        window_sec: Tamaño de la ventana en segundos (ej. 3600 = por hora).
        scope: Identificador lógico ("hourly", "daily", ...) usado para armar
            un mensaje de error distinguible por el frontend.
        language: "en" para mensaje en inglés, cualquier otro valor (o None)
            para español.
    """
    key = _bucket_key(user_id, window_sec)
    try:
        raw = await redis.get(key)
    except Exception:
        # Fail-open a propósito, mismo criterio que check_rate_limit: el
        # presupuesto de tokens es un guardrail de costo, no un control de
        # seguridad crítico — no tiene sentido tumbar con 500 a TODOS los
        # usuarios por la caída de una dependencia auxiliar.
        logger.error(
            "Token budget check falló (Redis no disponible), dejando pasar: scope=%s user_id=%s",
            scope,
            user_id,
        )
        return

    used = int(raw) if raw is not None else 0
    if used >= limit:
        retry_after = window_sec - (int(time.time()) % window_sec)
        messages = _MESSAGES_EN if language == "en" else _MESSAGES
        message_template = messages.get(scope, messages["daily"])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "token_budget_exceeded",
                "scope": scope,
                "message": message_template.format(limit=limit),
                "limit": limit,
                "used": used,
                "window_sec": window_sec,
            },
            headers={"Retry-After": str(retry_after)},
        )


async def record_token_usage(
    user_id: int,
    redis: redis_async.Redis,
    tokens: int,
    window_sec: int,
) -> None:
    """Suma `tokens` al acumulado de la ventana vigente de user_id.

    Se llama una vez por ventana (ej. una vez con window_sec=3600 y otra con
    window_sec=86400) para que el consumo quede reflejado en ambos contadores
    independientes, igual que check_token_budget se llama una vez por ventana.

    Fail-open igual que check_token_budget: si Redis falla acá se pierde el
    conteo de ESTA request pero no tiene sentido romper la respuesta ya
    generada — el LLM ya fue llamado (y facturado), devolver 500 al usuario
    por un fallo de contabilidad interna sería peor que el riesgo residual
    de subcontar durante la caída.
    """
    if tokens <= 0:
        return
    key = _bucket_key(user_id, window_sec)
    try:
        pipe = redis.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, window_sec + 10)
        await pipe.execute()
    except Exception:
        logger.error(
            "No se pudo registrar consumo de tokens (Redis no disponible): user_id=%s tokens=%s window_sec=%s",
            user_id,
            tokens,
            window_sec,
        )
