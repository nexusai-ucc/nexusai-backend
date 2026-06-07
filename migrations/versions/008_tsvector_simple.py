"""Switch content_tsv from 'spanish' to 'simple' text search config

Revision ID: 008_tsvector_simple
Revises: 007_hybrid_search
Create Date: 2026-06-07 00:00:00.000000

'spanish' stemming drops unrecognised tokens (NexusAI, RAG, Sprint, Python,
etc.). 'simple' lowercases and indexes every token as-is, so technical terms
and proper nouns are always searchable.
"""

from __future__ import annotations

from alembic import op

revision = "008_tsvector_simple"
down_revision = "007_hybrid_search"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
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
