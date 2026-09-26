"""
Almacén de resúmenes permanentes (COST-02, issue #521).

Guarda en `document_summaries` el resumen de cada documento y el de repaso
pre-examen, y los sirve mientras no cambie lo que los originó. Reemplaza a la
caché de Redis de 24 h de PERF-02.

Fail-open: si la base falla al leer o guardar, se loguea y el resumen se
genera igual (o se devuelve sin guardar). Las dos operaciones corren dentro de
un savepoint, así un error no deja la sesión del pedido inutilizable.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentSummary

logger = logging.getLogger("nexusai.documents.summary_store")

KIND_DOCUMENT = "document"
KIND_PRE_EXAM = "pre_exam"


@dataclass(frozen=True)
class StoredSummary:
    """Un resumen guardado y lo que costó generarlo."""

    payload: dict
    model: Optional[str]
    provider: Optional[str]
    prompt_tokens: int
    completion_tokens: int

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def make_key(*parts: str) -> str:
    """Hash estable de las partes que definen un resumen."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


async def lookup(db: AsyncSession, cache_key: str) -> Optional[StoredSummary]:
    """Trae el resumen guardado para esa clave y cuenta el acierto. None si no
    hay, o si la base falla."""
    try:
        async with db.begin_nested():
            row = (
                await db.execute(
                    update(DocumentSummary)
                    .where(DocumentSummary.cache_key == cache_key)
                    .values(
                        hits=DocumentSummary.hits + 1,
                        last_hit_at=datetime.now(timezone.utc),
                    )
                    .returning(
                        DocumentSummary.payload,
                        DocumentSummary.model,
                        DocumentSummary.provider,
                        DocumentSummary.prompt_tokens,
                        DocumentSummary.completion_tokens,
                    )
                )
            ).first()
        await db.commit()
    except Exception as exc:
        logger.warning(
            "Resúmenes guardados no disponibles en lectura: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None
    if row is None or not isinstance(row.payload, dict):
        return None
    return StoredSummary(
        payload=row.payload,
        model=row.model,
        provider=row.provider,
        prompt_tokens=row.prompt_tokens or 0,
        completion_tokens=row.completion_tokens or 0,
    )


async def save(
    db: AsyncSession,
    *,
    cache_key: str,
    kind: str,
    course_id: int,
    document_id: Optional[UUID],
    prompt_version: str,
    payload: dict,
    model: Optional[str],
    provider: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Guarda un resumen recién generado. Si otro pedido ya lo guardó (dos
    alumnos pidiéndolo a la vez), se queda con el primero. Nunca propaga."""
    try:
        async with db.begin_nested():
            await db.execute(
                pg_insert(DocumentSummary)
                .values(
                    cache_key=cache_key,
                    kind=kind,
                    course_id=course_id,
                    document_id=document_id,
                    prompt_version=prompt_version,
                    payload=payload,
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                .on_conflict_do_nothing(index_elements=["cache_key"])
            )
        await db.commit()
    except Exception as exc:
        logger.warning(
            "Resúmenes guardados no disponibles en escritura: %s: %s",
            type(exc).__name__,
            exc,
        )
