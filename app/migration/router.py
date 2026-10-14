"""
Export temporal de los datos del alumno para migrarlos a Moodle (DATA-04, #524).

En la opción C (ADR-014) los datos del alumno pasan a tablas nativas de Moodle.
El plugin los trae con `cli/migrate_from_backend.php` (DATA-06, #526) a través
de estos dos endpoints, firmados con HMAC igual que el resto:

- `GET /export?table=...&after=...&limit=...`: filas de una tabla, en orden de
  creación, paginadas por cursor. Conserva los ids (UUID) para que Moodle los
  guarde en su columna `uuid` y reconstruya las relaciones.
  Las flashcards traen además `source_cmid`, la actividad de su documento, para
  que Moodle aplique la visibilidad del material (VIS-05).
- `GET /export/summary`: conteos por tabla y sumas de tokens por curso y mes,
  para comparar antes y después de migrar.

Está apagado por defecto (MIGRATION_EXPORT_ENABLED): se prende solo durante el
corte, porque entrega en bloque los datos de todos los alumnos. Se borra en la
limpieza del backend (DATA-08, #530).
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import (
    CalendarAlert,
    ChatSession,
    Document,
    Flashcard,
    FlashcardReview,
    ForumWebhookConfig,
    InteractionLog,
    Message,
    MessageFeedback,
    QuizAttempt,
    QuizError,
    UnansweredQuestion,
)
from app.db.session import get_db
from app.shared.config import get_settings

router = APIRouter()

# Nombre público de cada tabla (el de la base) y su modelo, en el orden en que
# hay que importarlas para que las relaciones ya existan en Moodle.
EXPORT_TABLES: dict[str, Any] = {
    "chat_sessions": ChatSession,
    "messages": Message,
    "interaction_logs": InteractionLog,
    "message_feedback": MessageFeedback,
    "unanswered_questions": UnansweredQuestion,
    "quiz_attempts": QuizAttempt,
    "quiz_errors": QuizError,
    "flashcards": Flashcard,
    "flashcard_reviews": FlashcardReview,
    "calendar_alerts": CalendarAlert,
    "forum_webhook_configs": ForumWebhookConfig,
}

_MAX_LIMIT = 1000


def _require_enabled() -> None:
    if not get_settings().migration_export_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="El export para la migración está apagado (MIGRATION_EXPORT_ENABLED).",
        )


def _value(value: Any) -> Any:
    """Valor de una columna listo para JSON."""
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if value is not None and hasattr(value, "tolist"):
        # Vector de pgvector (numpy): lista de floats.
        return [float(x) for x in value.tolist()]
    return value


def _row(model: Any, obj: Any) -> dict[str, Any]:
    return {c.key: _value(getattr(obj, c.key)) for c in model.__table__.columns}


def _uuid_key(value: Any) -> Optional[str]:
    """Un id de documento guardado como texto, en la forma canónica de UUID."""
    try:
        return str(uuid.UUID(str(value))) if value else None
    except ValueError:
        return None


async def _source_cmids(db: AsyncSession, rows: list[Any]) -> dict[str, int]:
    """Actividad (cmid) del documento de origen de cada flashcard, por id de documento."""
    ids = {
        uuid.UUID(key)
        for key in (_uuid_key(row.source_document_id) for row in rows)
        if key
    }
    if not ids:
        return {}
    result = await db.execute(
        select(Document.id, Document.cmid).where(
            Document.id.in_(ids), Document.cmid.is_not(None)
        )
    )
    return {str(doc_id): int(cmid) for doc_id, cmid in result.all()}


def _encode_cursor(created_at: datetime, row_id: Any) -> str:
    raw = f"{created_at.isoformat()}|{row_id}".encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        created, row_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        return datetime.fromisoformat(created), row_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cursor inválido"
        ) from exc


@router.get("/export")
async def export_table(
    _body: Annotated[bytes, Depends(verify_hmac)],
    table: str,
    after: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=_MAX_LIMIT),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Una página de filas de `table`, en orden de creación (y de id si empatan).

    `next` es el cursor de la página siguiente, o null en la última.
    """
    _require_enabled()
    model = EXPORT_TABLES.get(table)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tabla desconocida. Válidas: {', '.join(EXPORT_TABLES)}",
        )

    stmt = select(model).order_by(model.created_at, model.id).limit(limit + 1)
    if after:
        created_at, row_id = _decode_cursor(after)
        stmt = stmt.where(
            or_(
                model.created_at > created_at,
                and_(model.created_at == created_at, model.id > uuid.UUID(row_id)),
            )
        )
    rows = list((await db.execute(stmt)).scalars().all())
    has_more = len(rows) > limit
    rows = rows[:limit]
    out = [_row(model, r) for r in rows]
    if model is Flashcard:
        cmids = await _source_cmids(
            db, [r for r in rows if r.source_document_id is not None]
        )
        for item in out:
            key = _uuid_key(item.get("source_document_id"))
            item["source_cmid"] = cmids.get(key) if key else None
    return {
        "table": table,
        "rows": out,
        "next": _encode_cursor(rows[-1].created_at, rows[-1].id)
        if has_more and rows
        else None,
    }


@router.get("/export/summary")
async def export_summary(
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Conteos por tabla y tokens por curso y mes, para verificar la migración.

    Los tokens salen de los mensajes del asistente (por el curso de su
    conversación) y de las interacciones; son las dos fuentes que Moodle va a
    tener después de migrar, así que se comparan una contra otra.
    """
    _require_enabled()
    counts = {}
    for name, model in EXPORT_TABLES.items():
        counts[name] = int(
            (await db.execute(select(func.count()).select_from(model))).scalar() or 0
        )

    message_tokens = await db.execute(
        text(
            """
            SELECT s.course_id,
                   to_char(date_trunc('month', m.created_at AT TIME ZONE 'UTC'), 'YYYY-MM') AS month,
                   COUNT(*) AS messages,
                   COALESCE(SUM(m.token_count_prompt), 0) AS prompt_tokens,
                   COALESCE(SUM(m.token_count_completion), 0) AS completion_tokens
              FROM messages m
              JOIN chat_sessions s ON s.id = m.session_id
             GROUP BY 1, 2
             ORDER BY 1, 2
            """
        )
    )
    interaction_tokens = await db.execute(
        text(
            """
            SELECT course_id,
                   to_char(date_trunc('month', created_at AT TIME ZONE 'UTC'), 'YYYY-MM') AS month,
                   COUNT(*) AS interactions,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens
              FROM interaction_logs
             GROUP BY 1, 2
             ORDER BY 1, 2
            """
        )
    )
    return {
        "counts": counts,
        "message_tokens": [dict(r._mapping) for r in message_tokens],
        "interaction_tokens": [dict(r._mapping) for r in interaction_tokens],
    }
