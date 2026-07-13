"""Add quiz_errors table — SP-10 (sugerencias de repaso basadas en errores)

Revision ID: 010_quiz_errors
Revises: 009_forum_post_embeddings
Create Date: 2026-07-09 00:00:00.000000

Persiste las preguntas de quiz que el alumno respondió mal. Antes vivían
solo en localStorage del navegador (efímero, no cross-device). Esta tabla
es la base para:
  - ReviewPanel: historial de errores por alumno+curso, ahora server-side.
  - /api/v1/quiz/review-suggestions: agrega errores por archivo fuente y le
    pide al LLM sugerencias de qué repasar.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "010_quiz_errors"
down_revision = "009_forum_post_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "quiz_errors" not in existing_tables:
        op.create_table(
            "quiz_errors",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("course_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("question_type", sa.String(length=20), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("source_filename", sa.String(length=255), nullable=True),
            # No FK a documents.id: el enriquecimiento de source_document_id en
            # quiz/generate ya viene con un bug de tipos aguas arriba (UUID
            # tratado como int), así que este campo es best-effort / no confiable
            # para joins — se guarda como texto suelto, nunca para lógica crítica.
            sa.Column("source_document_id", sa.String(length=64), nullable=True),
            sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("correct_index", sa.Integer(), nullable=False, server_default="-1"),
            sa.Column("user_selected_index", sa.Integer(), nullable=True),
            sa.Column("user_answer", sa.Text(), nullable=True),
            sa.Column("ai_feedback", sa.Text(), nullable=True),
            sa.Column("ai_score", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("quiz_errors")}
    if "ix_quiz_errors_user_id_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_quiz_errors_user_id_course_id_created_at",
            "quiz_errors",
            ["user_id", "course_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_quiz_errors_user_id_course_id_created_at", "quiz_errors")
    op.drop_table("quiz_errors")
