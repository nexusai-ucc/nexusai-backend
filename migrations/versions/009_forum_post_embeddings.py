"""Add forum_post_embeddings table — Épica 06 (foros con IA)

Revision ID: 009_forum_post_embeddings
Revises: 008_tsvector_simple
Create Date: 2026-07-09 00:00:00.000000

Tabla que almacena el embedding de cada post de foro indexado.
Un post = un vector (sin chunking, los posts son texto corto).
Permite detección de duplicados por similitud coseno antes de publicar.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "009_forum_post_embeddings"
down_revision = "008_tsvector_simple"
branch_labels = None
depends_on = None

# Dimensiones del embedding (igual que chunks.embedding — gemini-embedding-001)
_EMBEDDING_DIM = 768


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "forum_post_embeddings" not in existing_tables:
        op.create_table(
            "forum_post_embeddings",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("forum_post_id", sa.Integer(), nullable=False),
            sa.Column("discussion_id", sa.Integer(), nullable=False),
            sa.Column("course_id", sa.Integer(), nullable=False),
            # SHA-256 del content para detectar ediciones y evitar re-embedear si no cambió
            sa.Column("content_hash", sa.String(64), nullable=False),
            # Texto del post (para devolver preview en la respuesta de duplicados)
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        # UNIQUE: un embedding por post (upsert por post_id)
        op.create_unique_constraint(
            "uq_forum_post_embeddings_post_id",
            "forum_post_embeddings",
            ["forum_post_id"],
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("forum_post_embeddings")}

    if "ix_forum_post_embeddings_course_id" not in existing_indexes:
        op.create_index(
            "ix_forum_post_embeddings_course_id",
            "forum_post_embeddings",
            ["course_id"],
        )

    if "ix_forum_post_embeddings_discussion_id" not in existing_indexes:
        op.create_index(
            "ix_forum_post_embeddings_discussion_id",
            "forum_post_embeddings",
            ["discussion_id"],
        )

    # Índice HNSW sobre embedding para búsqueda aproximada rápida por coseno
    # (igual que chunks — ver migración 002_hnsw_index)
    if "ix_forum_post_embeddings_embedding_hnsw" not in existing_indexes:
        op.execute(
            f"""
            CREATE INDEX ix_forum_post_embeddings_embedding_hnsw
            ON forum_post_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_forum_post_embeddings_embedding_hnsw")
    op.drop_index("ix_forum_post_embeddings_discussion_id", "forum_post_embeddings")
    op.drop_index("ix_forum_post_embeddings_course_id", "forum_post_embeddings")
    op.drop_table("forum_post_embeddings")
