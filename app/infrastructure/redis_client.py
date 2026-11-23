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
        # Pool que ESPERA una conexión libre en vez de fallar. El pool común de
        # redis-py tira ConnectionError("Too many connections") apenas se agota:
        # con 60 chats a la vez (demo de la defensa) daba 500 en ~2,5 % de los
        # pedidos. Cada worker uvicorn tiene su propio pool (total N * 50).
        pool = redis_async.BlockingConnectionPool.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
            # Si no se libera ninguna conexión en 5 s, recién ahí falla.
            timeout=5,
            # Timeouts agresivos: si Redis no responde rápido, fallamos rápido.
            # No queremos requests del API colgadas por una caída de Redis.
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        _redis_client = redis_async.Redis(connection_pool=pool)

    return _redis_client


async def close_redis() -> None:
    """
    Cierra el cliente Redis. Llamado desde el lifespan shutdown de FastAPI
    para liberar las conexiones al apagar el container limpiamente.
    """
    global _redis_client

    if _redis_client is not None:
        # El pool se pasó explícito: hay que pedir que se cierre también.
        await _redis_client.aclose(close_connection_pool=True)
        _redis_client = None
