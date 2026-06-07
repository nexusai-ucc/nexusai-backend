"""Add storage_path to documents for original-file download support

Revision ID: 005_document_storage_path
Revises: 004_document_hash
Create Date: 2026-06-06 00:00:00.000000

Agrega storage_path (String nullable) a la tabla documents.
Almacena el nombre del archivo en /app/uploads/ (e.g.
"<uuid>_<filename>") para que el endpoint de descarga pueda
servirlo. NULL en documentos indexados antes de esta migración.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "005_document_storage_path"
down_revision = "004_document_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("storage_path", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "storage_path")
