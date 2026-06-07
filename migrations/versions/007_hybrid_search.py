"""Add content_tsv tsvector column to chunks for hybrid search (Feature A+)

Revision ID: 007_hybrid_search
Revises: 006_unanswered_questions
Create Date: 2026-06-07 00:00:00.000000

Agrega content_tsv como columna generada (GENERATED ALWAYS AS STORED) a la
tabla chunks. PostgreSQL la mantiene sincronizada automáticamente con content.
El índice GIN acelera las búsquedas full-text con @@ plainto_tsquery.
"""

from __future__ import annotations

from alembic import op

revision = "007_hybrid_search"
down_revision = "006_unanswered_questions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.drop_column("chunks", "content_tsv")
