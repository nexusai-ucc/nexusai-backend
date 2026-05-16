"""Rate limiting por user_id usando ventanas fijas en Redis."""

from __future__ import annotations

import time

import redis.asyncio as redis_async
from fastapi import HTTPException, status


async def check_rate_limit(
    user_id: int,
    redis: redis_async.Redis,
    limit: int,
    window_sec: int = 60,
) -> None:
    """Lanza HTTP 429 si user_id superó `limit` llamadas en `window_sec` segundos.

    Implementación: ventana fija (fixed window). El bucket cambia cada `window_sec`
    segundos, así que en el peor caso un usuario puede hacer 2x el límite en el
    cruce de dos ventanas. Para rate limiting de chatbot académico es aceptable.

    Args:
        user_id: ID del usuario de Moodle (del payload, no de la sesión HTTP).
        redis: Cliente Redis async (inyectado como FastAPI Dependency).
        limit: Máximo de requests permitidos en la ventana.
        window_sec: Tamaño de la ventana en segundos (default 60 = 1 min).
    """
    bucket = int(time.time()) // window_sec
    key = f"nexusai:ratelimit:{window_sec}:{user_id}:{bucket}"

    # INCR + EXPIRE en pipeline: casi atómico y O(1).
    # TTL con buffer (+10s) para que la key no expire antes de que termine la ventana.
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_sec + 10)
    results = await pipe.execute()
    count = int(results[0])

    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit excedido: máximo {limit} requests por {window_sec}s",
            headers={"Retry-After": str(window_sec)},
        )
