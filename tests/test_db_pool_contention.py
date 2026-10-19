"""
DEF-09: el registro de consumo no puede quedarse esperando el pool de los
requests (TPC-09, #581).

El stress mostró que con ~40 alumnos el chat colapsaba. Cada request retiene
su conexión ("idle in transaction") mientras llama al LLM, y la capa de
proveedores registra el consumo *dentro* de ese mismo request
(`await record_usage(...)`). Si el registro usa el mismo pool, con el pool
lleno cada request espera una segunda conexión que nadie suelta: retener y
esperar. Termina por timeout a los 30 s, se pierde el registro y el request
suma 30 s.

Este test lo reproduce contra un Postgres REAL (el que levanta backend-ci):
ocupa todas las conexiones del pool de los requests con una transacción
abierta y pide registrar un consumo. Tiene que terminar enseguida y dejar la
fila escrita. Sin Postgres (en local), se saltea.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sqlalchemy import select, text

from app.db import session as db_session
from app.db.models import LlmUsage
from app.shared import usage_ledger
from app.shared.config import get_settings
from app.shared.usage_ledger import UsageRecord, record_usage

REQUEST_POOL_SIZE = 5 + 10  # pool_size + max_overflow de get_engine()


async def _postgres_available() -> bool:
    try:
        engine = db_session.get_engine()
        async with engine.connect() as conn:
            await asyncio.wait_for(conn.execute(text("select 1")), timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture
async def real_db(monkeypatch):
    """Engines reales, tabla llm_usage creada y todo limpio al terminar."""
    monkeypatch.setattr(get_settings(), "usage_ledger_enabled", True)
    if not await _postgres_available():
        try:
            await db_session.dispose_engine()
        except Exception:
            pass  # sin base no hay nada que cerrar
        pytest.skip("sin Postgres alcanzable (en CI lo provee backend-ci)")
    async with db_session.get_engine().begin() as conn:
        await conn.run_sync(LlmUsage.__table__.create, checkfirst=True)
    yield
    async with db_session.get_engine().begin() as conn:
        await conn.run_sync(LlmUsage.__table__.drop, checkfirst=True)
    await db_session.dispose_engine()


async def test_usage_is_recorded_while_the_request_pool_is_exhausted(real_db):
    """Criterio: con las 15 conexiones de los requests retenidas en una
    transacción abierta (lo que pasa con 15 chats esperando al LLM), registrar
    un consumo termina en menos de 5 s y la fila queda escrita."""
    # Arrange: ocupar todo el pool de los requests, como 15 chats en vuelo.
    factory = db_session.get_session_factory()
    held = [factory() for _ in range(REQUEST_POOL_SIZE)]
    for session in held:
        # Abre la transacción y retiene la conexión.
        await session.execute(text("select 1"))
    record = UsageRecord(kind="llm", feature="test.def09")

    try:
        # Act.
        start = time.perf_counter()
        await asyncio.wait_for(record_usage(record), timeout=8)
        elapsed = time.perf_counter() - start
    finally:
        for session in held:
            await session.close()

    # Assert.
    assert elapsed < 5, f"el registro esperó {elapsed:.1f} s por una conexión"
    async with factory() as session:
        rows = (
            await session.execute(
                select(LlmUsage).where(LlmUsage.feature == "test.def09")
            )
        ).all()
    assert len(rows) == 1


def test_the_ledger_does_not_share_the_request_pool():
    """Criterio: el registro de consumo abre sus sesiones con un engine propio
    (si compartiera el de los requests, vuelve el interbloqueo)."""
    # Act.
    persist_source = usage_ledger._persist.__code__.co_names

    # Assert.
    assert "get_ledger_session_factory" in persist_source
    assert "get_session_factory" not in persist_source
