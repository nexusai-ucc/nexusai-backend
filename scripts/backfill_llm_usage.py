"""
Carga en `llm_usage` los tokens históricos del chat (COST-01, issue #519).

Hasta el registro de consumo, solo el chat guardaba tokens: en cada mensaje
del asistente de `messages`. Este script pasa esos tokens a `llm_usage` para
que el registro arranque completo.

- Toma los mensajes del asistente con tokens, con el curso y el usuario de su
  sesión, anteriores a la primera fila real del chat en llm_usage (así no se
  cuenta dos veces lo que la API ya registró).
- Quedan con feature "chat.backfill", rol "unknown" y sin modelo ni costo:
  antes no se guardaba qué modelo respondió.
- Es idempotente: cada mensaje se identifica por request_id = id del mensaje
  y no se vuelve a cargar.
- Al final compara la suma de tokens de esos mensajes con la cargada.

Uso, dentro del contenedor de la API:

    python scripts/backfill_llm_usage.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import get_session_factory  # noqa: E402

_CUTOFF_SQL = """
SELECT COALESCE(MIN(created_at), now())
FROM llm_usage
WHERE feature IN ('chat.messages', 'chat.stream')
"""

_SOURCE_SQL = """
FROM messages m
JOIN chat_sessions s ON s.id = m.session_id
WHERE m.role = 'assistant'
  AND COALESCE(m.token_count_prompt, 0) + COALESCE(m.token_count_completion, 0) > 0
  AND m.created_at < :cutoff
"""

_PENDING_SQL = (
    "SELECT COUNT(*) "
    + _SOURCE_SQL
    + """
  AND NOT EXISTS (
      SELECT 1 FROM llm_usage u
      WHERE u.feature = 'chat.backfill' AND u.request_id = m.id::text
  )
"""
)

_INSERT_SQL = (
    """
INSERT INTO llm_usage (
    id, created_at, kind, feature, status, fallback,
    prompt_tokens, completion_tokens, cached_prompt_tokens, embedding_tokens,
    cost_usd, cache_hit, saved_tokens, course_id, user_id, role, request_id
)
SELECT
    gen_random_uuid(), m.created_at, 'llm', 'chat.backfill', 'ok', false,
    COALESCE(m.token_count_prompt, 0), COALESCE(m.token_count_completion, 0), 0, 0,
    NULL, false, 0, s.course_id, s.user_id, 'unknown', m.id::text
"""
    + _SOURCE_SQL
    + """
  AND NOT EXISTS (
      SELECT 1 FROM llm_usage u
      WHERE u.feature = 'chat.backfill' AND u.request_id = m.id::text
  )
"""
)

_SOURCE_TOTAL_SQL = (
    "SELECT COALESCE(SUM(COALESCE(m.token_count_prompt, 0) + COALESCE(m.token_count_completion, 0)), 0) "
    + _SOURCE_SQL
)

_LOADED_TOTAL_SQL = """
SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
FROM llm_usage
WHERE feature = 'chat.backfill'
"""


async def run(dry_run: bool) -> int:
    async with get_session_factory()() as session:
        cutoff = (await session.execute(text(_CUTOFF_SQL))).scalar_one()
        pending = (
            await session.execute(text(_PENDING_SQL), {"cutoff": cutoff})
        ).scalar_one()
        print(
            f"Mensajes anteriores a {cutoff.isoformat()} pendientes de cargar: {pending}"
        )
        if dry_run:
            return 0

        await session.execute(text(_INSERT_SQL), {"cutoff": cutoff})
        await session.commit()

        source = (
            await session.execute(text(_SOURCE_TOTAL_SQL), {"cutoff": cutoff})
        ).scalar_one()
        loaded = (await session.execute(text(_LOADED_TOTAL_SQL))).scalar_one()
    print(f"Tokens en messages: {source} · cargados en llm_usage: {loaded}")
    if source != loaded:
        print("ATENCIÓN: las sumas no coinciden.")
        return 1
    print("Las sumas coinciden.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carga los tokens históricos del chat en llm_usage"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Solo cuenta lo pendiente, sin escribir"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.dry_run)))


if __name__ == "__main__":
    main()
