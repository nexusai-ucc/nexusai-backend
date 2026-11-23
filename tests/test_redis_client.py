"""El cliente Redis compartido espera una conexión libre en vez de fallar.

Con el pool común, redis-py tira ConnectionError("Too many connections") apenas
se agota: en la prueba de carga de la demo (60 chats a la vez pasando por
Moodle) eso daba 500 en ~2,5 % de los pedidos.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import redis.asyncio as redis_async

from app.infrastructure import redis_client


async def test_el_pool_espera_una_conexion_libre_y_tiene_margen():
    settings = SimpleNamespace(redis_url="redis://localhost:6379/0")
    with (
        patch.object(redis_client, "_redis_client", None),
        patch.object(redis_client, "get_settings", return_value=settings),
    ):
        client = await redis_client.get_redis()
        pool = client.connection_pool

        assert isinstance(pool, redis_async.BlockingConnectionPool)
        assert pool.max_connections >= 50
        assert pool.timeout > 0
        # La misma instancia en el segundo pedido (singleton).
        assert await redis_client.get_redis() is client
        await redis_client.close_redis()
        assert redis_client._redis_client is None
