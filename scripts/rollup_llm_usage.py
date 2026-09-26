"""
Resume por día el detalle viejo de `llm_usage` y lo borra (COST-01, issue #519).

Pasa a `llm_usage_daily` todo lo que tenga más de USAGE_LEDGER_DETAIL_DAYS
días (396 por defecto, 13 meses) y lo borra de `llm_usage`, en una sola
transacción. Es idempotente: si se vuelve a correr, suma sobre el resumen
existente y la segunda pasada no encuentra filas viejas.

Pensado para cron, una vez por día (ver docs/DEPLOY_ORACLE.md).

Uso, dentro del contenedor de la API:

    python scripts/rollup_llm_usage.py [--days N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import get_session_factory  # noqa: E402
from app.shared.config import get_settings  # noqa: E402

_COUNT_SQL = "SELECT COUNT(*) FROM llm_usage WHERE created_at < :cutoff"

_ROLLUP_SQL = """
INSERT INTO llm_usage_daily (
    id, day, client_id, course_id, role, feature, kind, provider, model, status,
    calls, prompt_tokens, completion_tokens, cached_prompt_tokens, embedding_tokens,
    audio_seconds, cost_usd, cache_hits, saved_tokens
)
SELECT
    gen_random_uuid(),
    (created_at AT TIME ZONE 'UTC')::date,
    COALESCE(client_id, ''),
    COALESCE(course_id, 0),
    role,
    feature,
    kind,
    COALESCE(provider, ''),
    COALESCE(model, ''),
    status,
    COUNT(*),
    SUM(prompt_tokens),
    SUM(completion_tokens),
    SUM(cached_prompt_tokens),
    SUM(embedding_tokens),
    COALESCE(SUM(audio_seconds), 0),
    COALESCE(SUM(cost_usd), 0),
    COUNT(*) FILTER (WHERE cache_hit),
    SUM(saved_tokens)
FROM llm_usage
WHERE created_at < :cutoff
GROUP BY 2, 3, 4, 5, 6, 7, 8, 9, 10
ON CONFLICT ON CONSTRAINT uq_llm_usage_daily_key DO UPDATE SET
    calls = llm_usage_daily.calls + EXCLUDED.calls,
    prompt_tokens = llm_usage_daily.prompt_tokens + EXCLUDED.prompt_tokens,
    completion_tokens = llm_usage_daily.completion_tokens + EXCLUDED.completion_tokens,
    cached_prompt_tokens = llm_usage_daily.cached_prompt_tokens + EXCLUDED.cached_prompt_tokens,
    embedding_tokens = llm_usage_daily.embedding_tokens + EXCLUDED.embedding_tokens,
    audio_seconds = llm_usage_daily.audio_seconds + EXCLUDED.audio_seconds,
    cost_usd = llm_usage_daily.cost_usd + EXCLUDED.cost_usd,
    cache_hits = llm_usage_daily.cache_hits + EXCLUDED.cache_hits,
    saved_tokens = llm_usage_daily.saved_tokens + EXCLUDED.saved_tokens
"""

_DELETE_SQL = "DELETE FROM llm_usage WHERE created_at < :cutoff"


async def run(days: int, dry_run: bool) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with get_session_factory()() as session:
        pending = (
            await session.execute(text(_COUNT_SQL), {"cutoff": cutoff})
        ).scalar_one()
        print(f"Filas de llm_usage anteriores a {cutoff.isoformat()}: {pending}")
        if dry_run or pending == 0:
            return
        await session.execute(text(_ROLLUP_SQL), {"cutoff": cutoff})
        await session.execute(text(_DELETE_SQL), {"cutoff": cutoff})
        await session.commit()
    print(f"Resumidas por día y borradas del detalle: {pending}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume por día el detalle viejo de llm_usage"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=get_settings().usage_ledger_detail_days,
        help="Días de detalle que se conservan (por defecto USAGE_LEDGER_DETAIL_DAYS)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Solo cuenta, sin escribir"
    )
    args = parser.parse_args()
    asyncio.run(run(args.days, args.dry_run))


if __name__ == "__main__":
    main()
