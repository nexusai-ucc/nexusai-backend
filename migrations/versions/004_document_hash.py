"""Add file_hash to documents for incremental indexing — CONT-04

Revision ID: 004_document_hash
Revises: 003_message_token_counts
Create Date: 2026-05-25 00:00:00.000000

Agrega file_hash (SHA-256 del contenido base64 recibido) a la tabla documents.
Permite detectar re-subidas del mismo archivo sin re-indexar.

La columna es NULLABLE para compatibilidad con documentos existentes.
El índice UNIQUE cubre (course_id, filename, file_hash): PostgreSQL trata
los NULL como distintos en índices únicos, por lo que los documentos
anteriores a esta migración (file_hash = NULL) no colisionan entre sí.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_document_hash"
down_revision = "003_message_token_counts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("file_hash", sa.String(length=64), nullable=True),
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_course_filename_hash
        ON documents (course_id, filename, file_hash)
        WHERE file_hash IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_course_filename_hash")
    op.drop_column("documents", "file_hash")
