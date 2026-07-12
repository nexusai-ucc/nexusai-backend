"""Add interaction_logs table — DOC-D01 (logging de interacciones anonimizado)

Revision ID: 010_interaction_logs
Revises: 009_forum_post_embeddings
Create Date: 2026-07-12 00:00:00.000000

Tabla que registra cada interacción con el asistente de forma anonimizada.
No almacena texto de preguntas ni user_id directo. Usa SHA-256 del user_id
para poder contar usuarios únicos sin exponer identidad.

Alimenta el dashboard de analytics para docentes (DOC-D01 → ANALYTICS-01).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "010_interaction_logs"
down_revision = "009_forum_post_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "interaction_logs" not in existing_tables:
        op.create_table(
            "interaction_logs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("course_id", sa.Integer(), nullable=False),
            sa.Column("user_id_hash", sa.String(64), nullable=False),
            sa.Column("question_char_count", sa.Integer(), nullable=False),
            sa.Column("answer_char_count", sa.Integer(), nullable=False),
            sa.Column("chunks_retrieved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("has_relevant_context", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("is_multicourse", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("latency_ms", sa.Float(), nullable=False),
            sa.Column("endpoint", sa.String(10), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("interaction_logs")}
    if "ix_interaction_logs_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_interaction_logs_course_id_created_at",
            "interaction_logs",
            ["course_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_interaction_logs_course_id_created_at", "interaction_logs")
    op.drop_table("interaction_logs")
