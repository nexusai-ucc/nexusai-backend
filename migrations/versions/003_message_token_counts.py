"""Add token count columns to messages table — BACK-12

Revision ID: 003_message_token_counts
Revises: 002_hnsw_index
Create Date: 2026-05-16 00:00:00.000000

Agrega token_count_prompt y token_count_completion a la tabla messages.
Ambas columnas son nullable para compatibilidad con filas anteriores
(mensajes existentes tienen NULL en estas columnas).

El backend guarda estos valores solo en mensajes role='assistant'
(respuestas del LLM). Los mensajes role='user' los dejan en NULL.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003_message_token_counts"
down_revision = "002_hnsw_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("token_count_prompt", sa.Integer(), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("token_count_completion", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "token_count_completion")
    op.drop_column("messages", "token_count_prompt")
