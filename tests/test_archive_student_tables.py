"""
Script que archiva las tablas del alumno después del corte (DATA-06, #526).

Se carga desde el path, como test_health_check_alert.py. El movimiento real
(`ALTER TABLE ... SET SCHEMA`) se probó contra PostgreSQL.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from app.migration.router import EXPORT_TABLES

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "archive_student_tables.py"
)
_spec = importlib.util.spec_from_file_location("archive_student_tables", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archive)


def test_archives_the_same_tables_the_export_hands_to_moodle():
    assert archive.STUDENT_TABLES == tuple(EXPORT_TABLES)


def test_move_statements_go_both_ways():
    assert archive.move_statements(("messages",), "public", "archive") == [
        'ALTER TABLE "public"."messages" SET SCHEMA "archive"'
    ]
    assert archive.move_statements(("messages",), "archive", "public") == [
        'ALTER TABLE "archive"."messages" SET SCHEMA "public"'
    ]
