"""
Tests de app.shared.token_budget (check_token_budget + record_token_usage).

Cubre:
  - Mientras el acumulado sea menor al límite, check_token_budget no bloquea.
  - Al llegar (o superar) el límite, check_token_budget corta con 429.
  - record_token_usage suma tokens al acumulado de la ventana correspondiente.
  - Las ventanas horaria y diaria son independientes entre sí (distinta key
    en Redis por tener distinto window_sec).
  - check_token_budget NO incrementa nada por sí sola (solo lee) — el
    acumulado solo cambia vía record_token_usage.
  - Fail-open si Redis no responde, en ambas funciones.
  - El mensaje de error respeta el idioma de la pregunta (en/es).

Mismo criterio que test_rate_limit.py: un FakeRedis respaldado por un dict
compartido en vez del fixture `fake_redis` (MagicMock) del conftest, porque
acá también hace falta un GET/INCRBY real que persista entre llamadas.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.shared.token_budget import check_token_budget, record_token_usage

DAY = 86400
HOUR = 3600


class _FakePipeline:
    """Emula redis.pipeline() respaldado por un dict compartido (INCRBY real)."""

    def __init__(self, store: dict[str, int]) -> None:
        self._store = store
        self._ops: list[tuple] = []

    def incrby(self, key: str, amount: int):
        self._ops.append(("incrby", key, amount))
        return self

    def expire(self, key: str, ttl: int):
        self._ops.append(("expire", key, ttl))
        return self

    async def execute(self) -> list:
        results = []
        for op in self._ops:
            if op[0] == "incrby":
                _, key, amount = op
                self._store[key] = self._store.get(key, 0) + amount
                results.append(self._store[key])
            elif op[0] == "expire":
                results.append(True)
        return results


class FakeRedis:
    """Redis async falso: suficiente para ejercitar check/record end-to-end."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    async def get(self, key: str):
        value = self.store.get(key)
        return None if value is None else str(value)

    def pipeline(self):
        return _FakePipeline(self.store)


@pytest.fixture
def fake_redis_budget() -> FakeRedis:
    return FakeRedis()


async def test_permite_mientras_no_se_alcance_el_limite(
    fake_redis_budget: FakeRedis,
) -> None:
    """Sin uso registrado todavía, check_token_budget no debe lanzar."""
    await check_token_budget(
        user_id=1, redis=fake_redis_budget, limit=1000, window_sec=DAY, scope="daily"
    )

    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=500, window_sec=DAY
    )

    # 500 usados de 1000 — sigue permitiendo.
    await check_token_budget(
        user_id=1, redis=fake_redis_budget, limit=1000, window_sec=DAY, scope="daily"
    )


async def test_bloquea_al_alcanzar_el_limite(fake_redis_budget: FakeRedis) -> None:
    """Al llegar exactamente al límite, la siguiente consulta se corta con 429."""
    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=1000, window_sec=DAY
    )

    with pytest.raises(HTTPException) as exc_info:
        await check_token_budget(
            user_id=1, redis=fake_redis_budget, limit=1000, window_sec=DAY, scope="daily"
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "token_budget_exceeded"
    assert exc_info.value.detail["scope"] == "daily"
    assert exc_info.value.detail["used"] == 1000
    assert "mañana" in exc_info.value.detail["message"]


async def test_record_token_usage_acumula_entre_llamadas(
    fake_redis_budget: FakeRedis,
) -> None:
    """Varias respuestas del LLM suman al mismo acumulado hasta cortar."""
    await record_token_usage(
        user_id=5, redis=fake_redis_budget, tokens=300, window_sec=HOUR
    )
    await record_token_usage(
        user_id=5, redis=fake_redis_budget, tokens=300, window_sec=HOUR
    )
    await record_token_usage(
        user_id=5, redis=fake_redis_budget, tokens=300, window_sec=HOUR
    )
    # 900 acumulados, límite 1000 — todavía pasa.
    await check_token_budget(
        user_id=5, redis=fake_redis_budget, limit=1000, window_sec=HOUR, scope="hourly"
    )

    await record_token_usage(
        user_id=5, redis=fake_redis_budget, tokens=200, window_sec=HOUR
    )
    # 1100 acumulados, límite 1000 — ahora corta.
    with pytest.raises(HTTPException) as exc_info:
        await check_token_budget(
            user_id=5,
            redis=fake_redis_budget,
            limit=1000,
            window_sec=HOUR,
            scope="hourly",
        )
    assert exc_info.value.detail["used"] == 1100


async def test_check_token_budget_no_incrementa_por_si_sola(
    fake_redis_budget: FakeRedis,
) -> None:
    """A diferencia de check_rate_limit, check_token_budget solo LEE."""
    for _ in range(10):
        await check_token_budget(
            user_id=9, redis=fake_redis_budget, limit=1, window_sec=DAY, scope="daily"
        )
    # El acumulado sigue en 0 — ninguna llamada a check_token_budget lo tocó.
    assert fake_redis_budget.store == {}


async def test_ventana_horaria_y_diaria_son_independientes(
    fake_redis_budget: FakeRedis,
) -> None:
    """Agotar la ventana horaria de un usuario no afecta su ventana diaria."""
    user_id = 42

    await record_token_usage(
        user_id=user_id, redis=fake_redis_budget, tokens=1000, window_sec=HOUR
    )
    with pytest.raises(HTTPException) as exc_info:
        await check_token_budget(
            user_id=user_id,
            redis=fake_redis_budget,
            limit=1000,
            window_sec=HOUR,
            scope="hourly",
        )
    assert exc_info.value.detail["scope"] == "hourly"

    # El límite diario del mismo usuario sigue intacto.
    await check_token_budget(
        user_id=user_id,
        redis=fake_redis_budget,
        limit=5000,
        window_sec=DAY,
        scope="daily",
    )


async def test_record_token_usage_ignora_valores_no_positivos(
    fake_redis_budget: FakeRedis,
) -> None:
    """tokens<=0 no debe tocar Redis (evita keys vacías por respuestas de 0 tokens)."""
    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=0, window_sec=DAY
    )
    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=-5, window_sec=DAY
    )
    assert fake_redis_budget.store == {}


class _BrokenRedis:
    """Simula Redis caído/timeout: get() y pipeline().execute() revientan."""

    class _BrokenPipeline:
        def incrby(self, key: str, amount: int):
            return self

        def expire(self, key: str, ttl: int):
            return self

        async def execute(self):
            raise ConnectionError("simulated redis outage")

    async def get(self, key: str):
        raise ConnectionError("simulated redis outage")

    def pipeline(self):
        return self._BrokenPipeline()


async def test_check_token_budget_falla_abierto_cuando_redis_no_responde() -> None:
    """Si Redis está caído, no debe tumbar el request con 500 para todos."""
    await check_token_budget(
        user_id=1, redis=_BrokenRedis(), limit=1000, window_sec=DAY, scope="daily"
    )


async def test_record_token_usage_falla_abierto_cuando_redis_no_responde() -> None:
    """Perder el conteo de una request por Redis caído no debe romper la respuesta."""
    await record_token_usage(
        user_id=1, redis=_BrokenRedis(), tokens=500, window_sec=DAY
    )


@pytest.mark.parametrize("scope,window", [("hourly", HOUR), ("daily", DAY)])
async def test_mensaje_es_ingles_cuando_la_pregunta_es_ingles(
    fake_redis_budget: FakeRedis, scope: str, window: int
) -> None:
    """El alumno ve este texto tal cual, así que sigue el idioma de la pregunta."""
    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=1, window_sec=window
    )

    with pytest.raises(HTTPException) as exc_info:
        await check_token_budget(
            user_id=1,
            redis=fake_redis_budget,
            limit=1,
            window_sec=window,
            scope=scope,
            language="en",
        )

    message = exc_info.value.detail["message"]
    assert "limit of 1 tokens" in message
    assert "Alcanzaste" not in message


async def test_mensaje_queda_en_espanol_por_default(
    fake_redis_budget: FakeRedis,
) -> None:
    await record_token_usage(
        user_id=1, redis=fake_redis_budget, tokens=1, window_sec=DAY
    )

    with pytest.raises(HTTPException) as exc_info:
        await check_token_budget(
            user_id=1, redis=fake_redis_budget, limit=1, window_sec=DAY, scope="daily"
        )

    assert "Alcanzaste tu límite de 1 tokens" in exc_info.value.detail["message"]
