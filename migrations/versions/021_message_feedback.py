"""Tabla message_feedback — voto 👍/👎 del alumno sobre respuestas del chat (ASIST-01, issue #321)

Revision ID: 021_message_feedback
Revises: 020_flashcards_spaced_rep
Create Date: 2026-09-07 00:00:00.000000

Anónima por diseño (mismo criterio que interaction_logs, DOC-D01): no
guarda user_id, solo user_id_hash (SHA-256) para permitir upsert cuando el
alumno cambia de voto. `message_id` es SET NULL (PRIV-01 hard-deletea
messages) y `course_id` va denormalizado para que el agregado del curso
sobreviva a ese borrado.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "021_message_feedback"
down_revision = "020_flashcards_spaced_rep"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("is_helpful", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("user_id_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("message_id", "user_id_hash", name="uq_message_feedback_message_user"),
    )
    op.create_index(
        "ix_message_feedback_course_id_created_at",
        "message_feedback",
        ["course_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_message_feedback_course_id_created_at", table_name="message_feedback")
    op.drop_table("message_feedback")
