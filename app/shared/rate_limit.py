"""Rate limiting por user_id usando ventanas fijas en Redis."""

from __future__ import annotations

import logging
import time

import redis.asyncio as redis_async
from fastapi import HTTPException, status

logger = logging.getLogger("nexusai.rate_limit")

_MESSAGES = {
    "minute": "Superaste el límite de {limit} consultas por minuto. Esperá un momento y volvé a intentarlo.",
    "daily": "Alcanzaste tu límite de {limit} consultas de hoy. Volvé a intentarlo mañana.",
}


async def check_rate_limit(
    user_id: int,
    redis: redis_async.Redis,
    limit: int,
    window_sec: int = 60,
    scope: str = "minute",
) -> None:
    """Lanza HTTP 429 si user_id superó `limit` llamadas en `window_sec` segundos.

    Implementación: ventana fija (fixed window). El bucket cambia cada `window_sec`
    segundos, así que en el peor caso un usuario puede hacer 2x el límite en el
    cruce de dos ventanas. Para rate limiting de chatbot académico es aceptable.

    El bucket se calcula como `int(time.time()) // window_sec`, que para
    window_sec=86400 (límite diario) alinea naturalmente con medianoche UTC
    (epoch 0 es 1970-01-01 00:00:00 UTC), no con medianoche en horario local
    (ej. Argentina, UTC-3). Se decidió NO ajustar esto a medianoche local
    porque: (1) no existe hoy una config de timezone en `Settings`, agregarla
    solo para esto sería alcance fuera de lo pedido; (2) es un límite de uso
    razonable, no una cuota facturable — un corrimiento de a lo sumo unas
    horas en el reset es aceptable. Si en el futuro se necesita precisión de
    medianoche local, ajustar acá el cálculo de `bucket` usando la timezone
    de `Settings`.

    Cada combinación (window_sec, user_id) usa su propia key en Redis, así que
    límites con distinto window_sec (ej. por-minuto vs diario) son independientes
    entre sí: agotar uno no cuenta contra el otro.

    Args:
        user_id: ID del usuario de Moodle (del payload, no de la sesión HTTP).
        redis: Cliente Redis async (inyectado como FastAPI Dependency).
        limit: Máximo de requests permitidos en la ventana.
        window_sec: Tamaño de la ventana en segundos (default 60 = 1 min).
        scope: Identificador lógico del límite ("minute", "daily", ...) usado
            para armar un mensaje de error distinguible por el frontend. No
            afecta el cálculo de la key — eso ya lo hace `window_sec`.
    """
    bucket = int(time.time()) // window_sec
    key = f"nexusai:ratelimit:{window_sec}:{user_id}:{bucket}"

    # INCR + EXPIRE en pipeline: casi atómico y O(1).
    # TTL con buffer (+10s) para que la key no expire antes de que termine la ventana.
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_sec + 10)
    try:
        results = await pipe.execute()
    except Exception:
        # Fail-open a propósito: si Redis está caído/timeout, el rate limit
        # es un guardrail de abuso, no un control de seguridad crítico —
        # bloquear con 500 a TODOS los alumnos (no solo a los que superaron
        # el límite) por la caída de una dependencia auxiliar es peor
        # experiencia que el riesgo residual de dejar pasar de más durante
        # la ventana en que Redis no responde. Mismo criterio que
        # moderation.py (MODERATION_FAIL_OPEN) y alerting.py para
        # dependencias auxiliares no críticas.
        logger.error(
            "Rate limit check falló (Redis no disponible), dejando pasar: scope=%s user_id=%s",
            scope,
            user_id,
        )
        return
    count = int(results[0])

    if count > limit:
        retry_after = window_sec - (int(time.time()) % window_sec)
        message_template = _MESSAGES.get(scope, _MESSAGES["minute"])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "scope": scope,
                "message": message_template.format(limit=limit),
                "limit": limit,
                "window_sec": window_sec,
            },
            headers={"Retry-After": str(retry_after)},
        )
