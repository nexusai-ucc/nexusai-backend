"""Add quiz_attempts table — ANALYTICS-01 (histograma de quiz scores por curso)

Revision ID: 014_quiz_attempts
Revises: 013_document_section
Create Date: 2026-07-29 00:00:00.000000

quiz_errors solo registra respuestas incorrectas, así que no alcanza para un
histograma representativo de puntajes (queda sesgado hacia errores). Esta
tabla registra el resultado de CADA intento de quiz completo (score total),
alimentando GET /api/v1/admin/analytics.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "014_quiz_attempts"
down_revision = "013_document_section"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "quiz_attempts" not in existing_tables:
        op.create_table(
            "quiz_attempts",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("course_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("total_questions", sa.Integer(), nullable=False),
            sa.Column("correct_answers", sa.Integer(), nullable=False),
            sa.Column("score", sa.Float(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("quiz_attempts")}
    if "ix_quiz_attempts_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_quiz_attempts_course_id_created_at",
            "quiz_attempts",
            ["course_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_quiz_attempts_course_id_created_at", "quiz_attempts")
    op.drop_table("quiz_attempts")
