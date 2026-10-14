"""
Archiva las tablas de datos del alumno después del corte a Moodle (DATA-06, #526).

Con la opción C (ADR-014) los datos del alumno viven en Moodle. Una vez migrados
y verificados con `cli/migrate_from_backend.php`, las tablas viejas del backend
pasan al esquema `archive`: nadie las vuelve a escribir, pero siguen en la base
(y en los backups) hasta la limpieza del backend (DATA-08, #530).

- Sin opciones solo muestra cuántas filas tiene cada tabla y dónde está.
- `--apply` mueve las tablas a `archive` en una sola transacción.
- `--restore` las devuelve a `public`, por si hay que volver atrás el corte.

Mover de esquema conserva filas, índices, claves y secuencias. Las rutas viejas
del backend que todavía leen estas tablas (chat con estado, panel de admin)
dejan de funcionar: el plugin desde DATA-05 ya no las usa.

Uso, dentro del contenedor de la API:

    python scripts/archive_student_tables.py [--apply | --restore]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import get_session_factory  # noqa: E402

ARCHIVE_SCHEMA = "archive"

# Las mismas tablas que entrega el export de la migración (app/migration/router.py).
STUDENT_TABLES = (
    "chat_sessions",
    "messages",
    "interaction_logs",
    "message_feedback",
    "unanswered_questions",
    "quiz_attempts",
    "quiz_errors",
    "flashcards",
    "flashcard_reviews",
    "calendar_alerts",
    "forum_webhook_configs",
)

_WHERE_SQL = """
SELECT table_schema
FROM information_schema.tables
WHERE table_name = :name AND table_schema IN ('public', :archive)
"""


def move_statements(tables: tuple[str, ...], source: str, target: str) -> list[str]:
    """Sentencias para mover cada tabla de `source` a `target`."""
    return [f'ALTER TABLE "{source}"."{name}" SET SCHEMA "{target}"' for name in tables]


async def _locations(session) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name in STUDENT_TABLES:
        rows = await session.execute(
            text(_WHERE_SQL), {"name": name, "archive": ARCHIVE_SCHEMA}
        )
        found[name] = sorted(r[0] for r in rows)
    return found


async def _count(session, schema: str, name: str) -> int:
    return int(
        (
            await session.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{name}"'))
        ).scalar()
        or 0
    )


async def run(apply: bool, restore: bool) -> int:
    source, target = ("public", ARCHIVE_SCHEMA)
    if restore:
        source, target = (ARCHIVE_SCHEMA, "public")

    async with get_session_factory()() as session:
        locations = await _locations(session)
        for name, schemas in locations.items():
            for schema in schemas:
                rows = await _count(session, schema, name)
                print(f"{schema}.{name}: {rows} filas")

        if not (apply or restore):
            return 0

        both = [n for n, s in locations.items() if len(s) > 1]
        if both:
            print(f"ERROR: están en los dos esquemas: {', '.join(both)}")
            return 1
        pending = tuple(n for n, s in locations.items() if s == [source])
        if not pending:
            print(f"Nada para mover: ninguna tabla está en {source}.")
            return 0

        await session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{target}"'))
        for statement in move_statements(pending, source, target):
            await session.execute(text(statement))
        await session.commit()
    print(f"Movidas a {target}: {', '.join(pending)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archiva (o restaura) las tablas de datos del alumno"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--apply", action="store_true", help="Mover las tablas al esquema archive"
    )
    group.add_argument(
        "--restore", action="store_true", help="Devolver las tablas a public"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.apply, args.restore)))


if __name__ == "__main__":
    main()
