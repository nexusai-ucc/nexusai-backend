"""
Cliente Redis async compartido.

Lo usan:
  - app.auth.hmac → store de nonces para anti-replay
  - app.shared.rate_limit (futuro) → contador por usuario/día
  - cualquier feature futura que necesite cache/locks distribuidos

Patrón:
  - Una sola conexión pool global (creada lazy on first use).
  - Cerrada en el lifespan shutdown de FastAPI (ver app/main.py).
  - Inyectada como FastAPI Dependency con `Depends(get_redis)`.
"""

from __future__ import annotations

from typing import Optional

import redis.asyncio as redis_async

from app.shared.config import get_settings

# Singleton del client. Lazy: se crea en la primera llamada a get_redis().
_redis_client: Optional[redis_async.Redis] = None


async def get_redis() -> redis_async.Redis:
    """
    Devuelve el cliente Redis async global.

    Uso:
        from fastapi import Depends
        from app.infrastructure.redis_client import get_redis

        @app.get("/...")
        async def endpoint(redis: redis_async.Redis = Depends(get_redis)):
            await redis.set("key", "value")
    """
    global _redis_client

    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis_async.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            # Pool razonable para 1 worker uvicorn. En prod con N workers, cada
            # worker tiene su propio pool — el total de conexiones será N * 10.
            max_connections=10,
            # Timeouts agresivos: si Redis no responde rápido, fallamos rápido.
            # No queremos requests del API colgadas por una caída de Redis.
            socket_timeout=2,
            socket_connect_timeout=2,
        )

    return _redis_client


async def close_redis() -> None:
    """
    Cierra el cliente Redis. Llamado desde el lifespan shutdown de FastAPI
    para liberar las conexiones al apagar el container limpiamente.
    """
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
