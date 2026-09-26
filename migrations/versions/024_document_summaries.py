"""Tabla document_summaries: resúmenes permanentes por versión del documento (COST-02, issue #521)

Revision ID: 024_document_summaries
Revises: 023_llm_usage
Create Date: 2026-09-26 00:00:00.000000

Reemplaza a la caché de Redis de 24 h (PERF-02): el resumen de un documento,
y el de repaso pre-examen que combina varios, se guardan hasta que cambie el
archivo, el modelo o el prompt, en vez de vencer al día. Ver
app/documents/summary_store.py.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "024_document_summaries"
down_revision = "023_llm_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(12), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("prompt_version", sa.String(40), nullable=False),
        sa.Column("model", sa.String(120), nullable=True),
        sa.Column("provider", sa.String(40), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("cache_key", name="uq_document_summaries_cache_key"),
    )
    op.create_index(
        "ix_document_summaries_course_id", "document_summaries", ["course_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_summaries_course_id", table_name="document_summaries")
    op.drop_table("document_summaries")
