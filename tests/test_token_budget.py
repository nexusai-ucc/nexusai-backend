"""
Tests de app.shared.token_budget (reserve_token_budget + finalize_token_usage
+ estimate_tokens/estimate_tokens_for_messages).

Cubre:
  - reserve_token_budget reserva atómicamente (INCRBY) y permite mientras el
    acumulado quede por debajo del límite.
  - Al llegar (o superar) el límite, reserve_token_budget revierte la
    reserva y corta con 429.
  - finalize_token_usage ajusta el acumulado con la diferencia entre lo
    reservado y el costo real (positiva o negativa).
  - Las ventanas horaria y diaria son independientes entre sí.
  - Los buckets de alumno y docente (mismo user_id, distinto is_teacher) son
    independientes entre sí — ver el docstring del módulo sobre el trade-off.
  - Requests CONCURRENTES del mismo usuario no pueden todas leer "0 usado" y
    pasar juntas — a diferencia de un simple GET, cada reserve_token_budget
    incrementa de inmediato, así que la carrera queda acotada (test que
    reproduce el mecanismo de falla que encontró /audit sobre la versión
    anterior de este módulo).
  - Fail-open si Redis no responde, en reserve y en finalize.
  - El mensaje de error respeta el idioma de la pregunta (en/es).

Mismo criterio que test_rate_limit.py: un FakeRedis respaldado por un dict
compartido en vez del fixture `fake_redis` (MagicMock) del conftest, porque
acá hace falta un INCRBY/DECRBY real que persista entre llamadas.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.shared.token_budget import (
    estimate_tokens,
    estimate_tokens_for_messages,
    finalize_token_usage,
    reserve_token_budget,
)

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
    """Redis async falso: suficiente para ejercitar reserve/finalize end-to-end."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    async def get(self, key: str):
        value = self.store.get(key)
        return None if value is None else str(value)

    async def decrby(self, key: str, amount: int):
        self.store[key] = self.store.get(key, 0) - amount
        return self.store[key]

    def pipeline(self):
        return _FakePipeline(self.store)


class _LockstepFakeRedis(FakeRedis):
    """Como FakeRedis, pero incrby() cede el control (`asyncio.sleep(0)`)
    ANTES de aplicar el incremento — simula que varias corrutinas puedan
    quedar "en vuelo" en el mismo punto al mismo tiempo, como pasaría con
    requests HTTP concurrentes reales (I/O de red de por medio). Sin esto,
    un test con asyncio.gather corre las corrutinas una atrás de la otra sin
    intercalarse de verdad y no reproduce la carrera."""

    def pipeline(self):
        return _LockstepFakePipeline(self.store)


class _LockstepFakePipeline(_FakePipeline):
    async def execute(self) -> list:
        await asyncio.sleep(0)
        return await super().execute()


@pytest.fixture
def fake_redis_budget() -> FakeRedis:
    return FakeRedis()


# ---------------------------------------------------------------------------
# estimate_tokens / estimate_tokens_for_messages
# ---------------------------------------------------------------------------


def test_estimate_tokens_vacio_es_cero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_texto_no_vacio_es_positivo() -> None:
    assert estimate_tokens("hola, ¿cómo estás?") > 0


def test_estimate_tokens_for_messages_suma_todos_los_content() -> None:
    messages = [
        {"role": "system", "content": "sos un asistente"},
        {"role": "user", "content": "hola"},
    ]
    total = estimate_tokens_for_messages(messages)
    assert total == estimate_tokens("sos un asistente") + estimate_tokens("hola")


# ---------------------------------------------------------------------------
# reserve_token_budget / finalize_token_usage — camino secuencial
# ---------------------------------------------------------------------------


async def test_reserva_permite_mientras_no_se_alcance_el_limite(
    fake_redis_budget: FakeRedis,
) -> None:
    reserved = await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=1000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=200,
    )
    assert reserved == 200
    assert fake_redis_budget.store[f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"] == 200


async def test_reserva_bloquea_al_alcanzar_el_limite_y_revierte(
    fake_redis_budget: FakeRedis,
) -> None:
    """La reserva que hace que el acumulado llegue al límite corta con 429
    y NO deja el incremento aplicado (se revierte)."""
    await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=1000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=1000,
    )

    with pytest.raises(HTTPException) as exc_info:
        await reserve_token_budget(
            user_id=1,
            is_teacher=False,
            redis=fake_redis_budget,
            limit=1000,
            window_sec=DAY,
            scope="daily",
            estimated_tokens=50,
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "token_budget_exceeded"
    assert exc_info.value.detail["scope"] == "daily"
    assert exc_info.value.detail["used"] == 1000
    assert "mañana" in exc_info.value.detail["message"]
    # La reserva de 50 que disparó el 429 no debe quedar aplicada.
    key = f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"
    assert fake_redis_budget.store[key] == 1000


async def test_finalize_ajusta_con_costo_real_mayor_a_lo_reservado(
    fake_redis_budget: FakeRedis,
) -> None:
    reserved = await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=10000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=100,
    )
    await finalize_token_usage(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        window_sec=DAY,
        reserved_tokens=reserved,
        actual_tokens=900,
    )
    key = f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"
    assert fake_redis_budget.store[key] == 900


async def test_finalize_ajusta_con_costo_real_menor_a_lo_reservado(
    fake_redis_budget: FakeRedis,
) -> None:
    """tiktoken puede sobreestimar — el ajuste puede ser negativo."""
    reserved = await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=10000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=500,
    )
    await finalize_token_usage(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        window_sec=DAY,
        reserved_tokens=reserved,
        actual_tokens=120,
    )
    key = f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"
    assert fake_redis_budget.store[key] == 120


async def test_finalize_con_reserved_cero_suma_directo(
    fake_redis_budget: FakeRedis,
) -> None:
    """Caso moderación: nada reservado (reserved_tokens=0) — se suma directo."""
    await finalize_token_usage(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        window_sec=DAY,
        reserved_tokens=0,
        actual_tokens=42,
    )
    key = f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"
    assert fake_redis_budget.store[key] == 42


async def test_ventana_horaria_y_diaria_son_independientes(
    fake_redis_budget: FakeRedis,
) -> None:
    user_id = 42
    await reserve_token_budget(
        user_id=user_id,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=1000,
        window_sec=HOUR,
        scope="hourly",
        estimated_tokens=1000,
    )
    with pytest.raises(HTTPException) as exc_info:
        await reserve_token_budget(
            user_id=user_id,
            is_teacher=False,
            redis=fake_redis_budget,
            limit=1000,
            window_sec=HOUR,
            scope="hourly",
            estimated_tokens=1,
        )
    assert exc_info.value.detail["scope"] == "hourly"

    # El límite diario del mismo usuario sigue intacto.
    await reserve_token_budget(
        user_id=user_id,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=5000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=1,
    )


async def test_bucket_de_alumno_y_docente_son_independientes(
    fake_redis_budget: FakeRedis,
) -> None:
    """Mismo user_id, pero is_teacher distinto en cada request (ej. ayudante
    de cátedra que en un curso es editingteacher y en otro alumno) — no
    deben compartir contador. Ver el docstring del módulo (hallazgo de
    /audit sobre PR #516: antes de esto compartían un único bucket global,
    con bloqueos inconsistentes según en qué curso preguntara)."""
    user_id = 7
    await reserve_token_budget(
        user_id=user_id,
        is_teacher=True,
        redis=fake_redis_budget,
        limit=20000,
        window_sec=HOUR,
        scope="hourly",
        estimated_tokens=15000,
    )
    # El alumno (mismo user_id) sigue con su cupo intacto pese al gasto
    # grande hecho como docente.
    reserved_student = await reserve_token_budget(
        user_id=user_id,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=8000,
        window_sec=HOUR,
        scope="hourly",
        estimated_tokens=500,
    )
    assert reserved_student == 500


# ---------------------------------------------------------------------------
# Concurrencia — el hallazgo central de /audit sobre la versión anterior
# ---------------------------------------------------------------------------


def _bucket(window_sec: int) -> int:
    import time

    return int(time.time()) // window_sec


async def test_reservas_concurrentes_no_pueden_pasar_todas_con_acumulado_viejo(
) -> None:
    """Antes de reserve_token_budget, check_token_budget solo hacía GET: N
    requests concurrentes del mismo usuario podían leer el mismo acumulado
    "viejo" (0) y pasar todas, porque nada se sumaba hasta después de que el
    LLM respondiera — ver /audit sobre PR #516. Con INCRBY atómico dentro
    de la propia reserva, un burst de requests concurrentes que juntas
    superan el límite deja bloqueadas a las que exceden el cupo, no a
    todas-pasan-de-largo.
    """
    redis = _LockstepFakeRedis()
    limit = 1000
    # 5 requests concurrentes, cada una reserva 300 — 5*300=1500 > 1000.
    # Con la implementación vieja (solo GET antes de llamar al LLM), las 5
    # hubieran leído "0 usado" y pasado todas — el acumulado recién se
    # actualizaba minutos después, cuando cada LLM terminaba de responder.
    # Acá, cada reserva se aplica de inmediato (INCRBY atómico): la 1ra deja
    # el acumulado en 300 (0 < 1000, pasa), la 2da en 600, la 3ra en 900, la
    # 4ta en 1200 (pasa: el acumulado ANTES de ella era 900 < 1000 — puede
    # cruzar el límite una vez, mismo criterio que rate_limit.py). La 5ta ya
    # ve un acumulado previo de 1200 >= 1000 y se revierte con 429. Como
    # mucho ceil(1000/300)=4 pueden entrar.

    async def attempt():
        try:
            await reserve_token_budget(
                user_id=1,
                is_teacher=False,
                redis=redis,
                limit=limit,
                window_sec=DAY,
                scope="daily",
                estimated_tokens=300,
            )
            return "allowed"
        except HTTPException:
            return "blocked"

    results = await asyncio.gather(*(attempt() for _ in range(5)))

    allowed = results.count("allowed")
    blocked = results.count("blocked")
    assert allowed + blocked == 5
    # Como mucho ceil(1000/300)=4 pueden entrar sin que el acumulado llegue
    # al límite — nunca las 5 (que es lo que pasaba con el diseño anterior).
    assert allowed <= 4
    assert blocked >= 1
    # El acumulado final nunca queda por encima del límite (cada reserva que
    # lo hubiera superado se revirtió).
    key = f"nexusai:tokenbudget:{DAY}:student:1:{_bucket(DAY)}"
    assert redis.store[key] < limit + 300  # nunca queda una reserva de más sin revertir
    assert redis.store[key] == allowed * 300


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


class _BrokenRedis:
    """Simula Redis caído/timeout: toda operación revienta."""

    class _BrokenPipeline:
        def incrby(self, key: str, amount: int):
            return self

        def expire(self, key: str, ttl: int):
            return self

        async def execute(self):
            raise ConnectionError("simulated redis outage")

    async def get(self, key: str):
        raise ConnectionError("simulated redis outage")

    async def decrby(self, key: str, amount: int):
        raise ConnectionError("simulated redis outage")

    def pipeline(self):
        return self._BrokenPipeline()


async def test_reserve_falla_abierto_cuando_redis_no_responde() -> None:
    """Si Redis está caído, no debe tumbar el request con 500 para todos —
    y debe devolver 0 (nada quedó reservado de verdad)."""
    reserved = await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=_BrokenRedis(),
        limit=1000,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=300,
    )
    assert reserved == 0


async def test_finalize_falla_abierto_cuando_redis_no_responde() -> None:
    await finalize_token_usage(
        user_id=1,
        is_teacher=False,
        redis=_BrokenRedis(),
        window_sec=DAY,
        reserved_tokens=0,
        actual_tokens=500,
    )


# ---------------------------------------------------------------------------
# Mensajes de error (idioma)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope,window", [("hourly", HOUR), ("daily", DAY)])
async def test_mensaje_es_ingles_cuando_la_pregunta_es_ingles(
    fake_redis_budget: FakeRedis, scope: str, window: int
) -> None:
    await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=1,
        window_sec=window,
        scope=scope,
        estimated_tokens=1,
    )

    with pytest.raises(HTTPException) as exc_info:
        await reserve_token_budget(
            user_id=1,
            is_teacher=False,
            redis=fake_redis_budget,
            limit=1,
            window_sec=window,
            scope=scope,
            estimated_tokens=1,
            language="en",
        )

    message = exc_info.value.detail["message"]
    assert "limit of 1 tokens" in message
    assert "Alcanzaste" not in message


async def test_mensaje_queda_en_espanol_por_default(
    fake_redis_budget: FakeRedis,
) -> None:
    await reserve_token_budget(
        user_id=1,
        is_teacher=False,
        redis=fake_redis_budget,
        limit=1,
        window_sec=DAY,
        scope="daily",
        estimated_tokens=1,
    )

    with pytest.raises(HTTPException) as exc_info:
        await reserve_token_budget(
            user_id=1,
            is_teacher=False,
            redis=fake_redis_budget,
            limit=1,
            window_sec=DAY,
            scope="daily",
            estimated_tokens=1,
        )

    assert "Alcanzaste tu límite de 1 tokens" in exc_info.value.detail["message"]
