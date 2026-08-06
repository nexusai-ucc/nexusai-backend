"""Add deleted_at + nullable user_id a quiz_attempts — PRIV-01 (issue #310)

Revision ID: 016_quiz_attempts_soft_delete
Revises: 015_quiz_attempts
Create Date: 2026-08-06 00:00:00.000000

PRIV-01: el alumno puede pedir el borrado de sus datos personales. Para
quiz_attempts no se puede hacer DELETE crudo porque su columna `score`
alimenta en vivo el histograma de Analytics del docente
(_get_quiz_score_distribution en app/admin/router.py, que filtra por
course_id/created_at y nunca por user_id). En vez de eso, se anonimiza
in-place: se limpia `user_id` y se marca `deleted_at`, la fila (y su score)
sigue existiendo para el agregado del curso.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "016_quiz_attempts_soft_delete"
down_revision = "015_quiz_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("quiz_attempts")}

    if "deleted_at" not in columns:
        op.add_column(
            "quiz_attempts",
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )

    op.alter_column(
        "quiz_attempts",
        "user_id",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    # No se puede volver a NOT NULL sin decidir qué hacer con las filas ya
    # anonimizadas (user_id NULL) — bloquear el downgrade a propósito en vez
    # de perder datos o inventar un valor sentinel silenciosamente.
    conn = op.get_bind()
    anonymized_count = conn.execute(
        sa.text("SELECT COUNT(*) FROM quiz_attempts WHERE user_id IS NULL")
    ).scalar_one()
    if anonymized_count:
        raise RuntimeError(
            f"No se puede hacer downgrade: hay {anonymized_count} filas de "
            "quiz_attempts con user_id NULL (anonimizadas por PRIV-01). "
            "Decidir manualmente qué hacer con ellas antes de bajar la migración."
        )

    op.alter_column(
        "quiz_attempts",
        "user_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.drop_column("quiz_attempts", "deleted_at")
