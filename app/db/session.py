"""
SQLAlchemy + asyncpg session factory.

Patrón estándar de FastAPI con SQLAlchemy 2.x async:
  - Un engine global (pool de conexiones), creado lazy.
  - Un async_sessionmaker factory.
  - Una FastAPI Dependency `get_db` que abre/cierra session por request.

Ver investigacion/04-chromadb/decision-pgvector.md y ADR-002 para el contexto:
todo (Moodle data + nuestro plugin + embeddings RAG) vive en la misma instancia
PostgreSQL. La extensión `vector` se habilita vía script en docker-compose.
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.shared.config import get_settings


class Base(DeclarativeBase):
    """
    Base de todos los modelos ORM de NexusAI.

    Los modelos viven cerca de su feature (ej: app/chat/models.py,
    app/documents/models.py) y heredan de esta Base. Alembic los descubre
    via importación en app/db/models.py (TODO: crear cuando agreguemos
    Alembic).
    """

    pass


# Singletons. Lazy: se crean en el primer get_engine().
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_ledger_engine: Optional[AsyncEngine] = None
_ledger_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _build_async_url(database_url: str) -> str:
    """
    Convierte la URL "postgresql://..." de .env al formato asyncpg que
    necesita SQLAlchemy: "postgresql+asyncpg://...".

    Aceptamos ambos formatos en .env para no obligar al equipo a memorizar
    el +asyncpg.
    """
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        # Heroku-style URL — también la convertimos.
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    raise ValueError(
        f"DATABASE_URL no reconocida: {database_url!r}. "
        "Esperaba postgresql:// o postgresql+asyncpg://"
    )


def get_engine() -> AsyncEngine:
    """
    Devuelve el engine async global.

    El engine mantiene un pool de conexiones — no abre una conexión por
    request, las recicla. Default de SQLAlchemy: 5 conexiones + 10 overflow.
    """
    global _engine

    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            _build_async_url(settings.database_url),
            # echo=True para ver SQL en logs durante dev. En prod = False
            # porque imprime cada query y satura los logs.
            echo=(settings.env == "development"),
            # pool_pre_ping previene "stale connection" después de que la
            # DB recicla conexiones inactivas (típico con pgbouncer o cuando
            # postgres restartea de noche).
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Devuelve la factory de sessions async."""
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            # expire_on_commit=False → después del commit, los objetos siguen
            # siendo accesibles sin re-fetch. Estándar para apps FastAPI.
            expire_on_commit=False,
            autoflush=False,
        )

    return _session_factory


def get_ledger_session_factory() -> async_sessionmaker[AsyncSession]:
    """Factory de sessions del registro de consumo, con su propio pool chico.

    DEF-09: el registro (app/shared/usage_ledger.py) se escribe DENTRO del
    request que llamó al proveedor, mientras ese request todavía tiene su
    conexión. Con un solo pool, bajo carga cada request retenía una conexión
    y esperaba otra que nadie soltaba (retener y esperar): 30 s de timeout
    por llamada y el registro perdido. Con un pool aparte el registro nunca
    compite con los requests, y si se satura espera a lo sumo
    USAGE_LEDGER_POOL_TIMEOUT_SEC.
    """
    global _ledger_engine, _ledger_session_factory

    if _ledger_session_factory is None:
        settings = get_settings()
        _ledger_engine = create_async_engine(
            _build_async_url(settings.database_url),
            pool_pre_ping=True,
            pool_size=settings.usage_ledger_pool_size,
            max_overflow=settings.usage_ledger_max_overflow,
            pool_timeout=settings.usage_ledger_pool_timeout_sec,
        )
        _ledger_session_factory = async_sessionmaker(
            bind=_ledger_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _ledger_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency. Abre una session, la cede al endpoint, y la cierra
    al final (incluso si el endpoint tira excepción).

    Uso:
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.db.session import get_db

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def dispose_engine() -> None:
    """
    Cierra el engine y libera todas las conexiones del pool.
    Llamado desde el lifespan shutdown de FastAPI (ver app/main.py).
    """
    global _engine, _session_factory, _ledger_engine, _ledger_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
    if _ledger_engine is not None:
        await _ledger_engine.dispose()
        _ledger_engine = None
        _ledger_session_factory = None
