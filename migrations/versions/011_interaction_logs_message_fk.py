"""Add user_message_id FK to interaction_logs — DOC-D01 gap fix

Revision ID: 011_interaction_logs_message_fk
Revises: 010_quiz_errors
Create Date: 2026-07-12 00:00:00.000000

Agrega FK opcional user_message_id → messages.id con ON DELETE SET NULL.
Permite a DOC-D02 joinear interaction_logs con messages para cruzar
contenido de la pregunta con métricas de RAG (grounding, chunks, latencia).

Nota: down_revision apunta a 010_quiz_errors (rama local) en lugar de
010_interaction_logs (rama santi) porque esa migración no fue mergeada a
main. El upgrade() tiene guards IF NOT EXISTS para idempotencia.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "011_interaction_logs_message_fk"
down_revision = "010_quiz_errors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # interaction_logs puede no existir si la migración de analytics (santi)
    # nunca se aplicó en este entorno — skip silencioso.
    existing_tables = inspector.get_table_names()
    if "interaction_logs" not in existing_tables:
        return

    existing_cols = {c["name"] for c in inspector.get_columns("interaction_logs")}

    if "user_message_id" not in existing_cols:
        op.add_column(
            "interaction_logs",
            sa.Column("user_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        op.create_foreign_key(
            "fk_interaction_logs_user_message_id",
            "interaction_logs",
            "messages",
            ["user_message_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    if "interaction_logs" not in existing_tables:
        return
    op.drop_constraint("fk_interaction_logs_user_message_id", "interaction_logs", type_="foreignkey")
    op.drop_column("interaction_logs", "user_message_id")
