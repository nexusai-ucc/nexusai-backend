"""HNSW index on chunks.embedding for fast cosine similarity retrieval

Revision ID: 002_hnsw_index
Revises: 001_initial_schema
Create Date: 2026-05-05 12:00:00.000000

Sin este índice, los queries `ORDER BY embedding <=> $1` hacen full table scan
sobre cada chunk → O(N) por query. Con HNSW, la búsqueda aproximada es ~O(log N)
y reduce la latencia a ms incluso con millones de chunks.

Referencia: https://github.com/pgvector/pgvector#indexing
"""

from __future__ import annotations

from alembic import op


revision = "002_hnsw_index"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # HNSW con distancia coseno (vector_cosine_ops).
    # Parámetros default de pgvector: m=16, ef_construction=64.
    # Más alto = mejor recall, más lento construir y más memoria.
    # Para nuestro caso (single-institution, <100k chunks típico) los defaults son OK.
    #
    # IMPORTANTE: el índice se construye sobre la columna `embedding` que es
    # nullable. Los rows con embedding=NULL (documents en estado 'pending'
    # antes de procesarse) simplemente no aparecen en los resultados — eso es
    # comportamiento deseado.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )

    # Índice compuesto en (course_id) sobre la tabla documents — acelera el
    # JOIN que hace el retriever para filtrar chunks por curso.
    # ix_documents_course_id ya existe desde 001 (definido en models.py via
    # __table_args__), pero por las dudas lo creamos IF NOT EXISTS por idempotencia.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documents_course_id
        ON documents (course_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    # No tocamos ix_documents_course_id en el downgrade: lo gestiona la migración 001.
