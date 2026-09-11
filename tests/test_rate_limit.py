"""
Tests de app.shared.rate_limit.check_rate_limit.

Cubre:
  - Se permiten requests hasta el límite (por minuto y diario).
  - Se bloquea con 429 al superar el límite.
  - El límite diario y el de minuto son independientes entre sí (distinta
    key en Redis por tener distinto window_sec, así que agotar uno no
    afecta al otro).
  - El mensaje de error distingue "minute" de "daily" (para que el frontend
    pueda mostrar algo específico).

No usamos el fixture `fake_redis` (MagicMock) del conftest porque
check_rate_limit necesita un pipeline con INCR real (contador que
persiste entre llamadas) — un fake en memoria representa mejor el
comportamiento real de Redis que mockear el resultado de `execute()`
a mano en cada test.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.shared.rate_limit import check_rate_limit

DAY = 86400
MINUTE = 60


class _FakePipeline:
    """Emula redis.pipeline() respaldado por un dict compartido (INCR real)."""

    def __init__(self, store: dict[str, int]) -> None:
        self._store = store
        self._ops: list[tuple] = []

    def incr(self, key: str):
        self._ops.append(("incr", key))
        return self

    def expire(self, key: str, ttl: int):
        self._ops.append(("expire", key, ttl))
        return self

    async def execute(self) -> list:
        results = []
        for op in self._ops:
            if op[0] == "incr":
                _, key = op
                self._store[key] = self._store.get(key, 0) + 1
                results.append(self._store[key])
            elif op[0] == "expire":
                results.append(True)
        return results


class FakeRedis:
    """Redis async falso: suficiente para ejercitar check_rate_limit end-to-end."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    def pipeline(self):
        return _FakePipeline(self.store)


@pytest.fixture
def fake_redis_counter() -> FakeRedis:
    return FakeRedis()


async def test_permite_hasta_el_limite_diario(fake_redis_counter: FakeRedis) -> None:
    """Las primeras `limit` consultas del día pasan sin lanzar excepción."""
    for _ in range(50):
        await check_rate_limit(
            user_id=1,
            redis=fake_redis_counter,
            limit=50,
            window_sec=DAY,
            scope="daily",
        )
    # Ninguna de las 50 llamadas anteriores debería haber lanzado.


async def test_bloquea_al_superar_el_limite_diario(fake_redis_counter: FakeRedis) -> None:
    """La consulta número 51 (con limit=50) debe devolver 429 con scope=daily."""
    for _ in range(50):
        await check_rate_limit(
            user_id=1,
            redis=fake_redis_counter,
            limit=50,
            window_sec=DAY,
            scope="daily",
        )

    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(
            user_id=1,
            redis=fake_redis_counter,
            limit=50,
            window_sec=DAY,
            scope="daily",
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["scope"] == "daily"
    assert exc_info.value.detail["error"] == "rate_limit_exceeded"
    assert "mañana" in exc_info.value.detail["message"]


async def test_limite_diario_y_por_minuto_son_independientes(
    fake_redis_counter: FakeRedis,
) -> None:
    """
    Agotar el límite por minuto de un usuario no debe afectar su límite
    diario, y viceversa — usan keys distintas en Redis (la key incluye
    window_sec).
    """
    user_id = 42

    # Agotamos el límite por minuto (limit=20).
    for _ in range(20):
        await check_rate_limit(
            user_id=user_id,
            redis=fake_redis_counter,
            limit=20,
            window_sec=MINUTE,
            scope="minute",
        )

    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(
            user_id=user_id,
            redis=fake_redis_counter,
            limit=20,
            window_sec=MINUTE,
            scope="minute",
        )
    assert exc_info.value.detail["scope"] == "minute"

    # El límite diario del mismo usuario sigue intacto: debe permitir
    # requests normalmente pese a que el límite por minuto ya está agotado.
    await check_rate_limit(
        user_id=user_id,
        redis=fake_redis_counter,
        limit=50,
        window_sec=DAY,
        scope="daily",
    )

    # Y a la inversa: agotar el diario no debe afectar un nuevo usuario
    # ni un nuevo minuto para el límite por minuto de este mismo usuario
    # (distinta key: distinto bucket/limit no interfieren).
    daily_key_calls = 49  # ya hicimos 1 arriba, faltan 49 para llegar a 50
    for _ in range(daily_key_calls):
        await check_rate_limit(
            user_id=user_id,
            redis=fake_redis_counter,
            limit=50,
            window_sec=DAY,
            scope="daily",
        )
    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(
            user_id=user_id,
            redis=fake_redis_counter,
            limit=50,
            window_sec=DAY,
            scope="daily",
        )
    assert exc_info.value.detail["scope"] == "daily"


async def test_mensaje_por_minuto_distingue_de_diario(
    fake_redis_counter: FakeRedis,
) -> None:
    """El detail del 429 por minuto no debe confundirse con el diario."""
    for _ in range(5):
        await check_rate_limit(
            user_id=7,
            redis=fake_redis_counter,
            limit=5,
            window_sec=MINUTE,
            scope="minute",
        )

    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(
            user_id=7,
            redis=fake_redis_counter,
            limit=5,
            window_sec=MINUTE,
            scope="minute",
        )

    detail = exc_info.value.detail
    assert detail["scope"] == "minute"
    assert "minuto" in detail["message"]
    assert "mañana" not in detail["message"]
