"""Add section column to documents — BUS-05 (filtro por unidad/sección)

Revision ID: 013_document_section
Revises: 012_interaction_logs_message_fk
Create Date: 2026-07-27 00:00:00.000000

Agrega columna opcional `section` (número de sección/unidad de Moodle) a
`documents`, más un índice compuesto (course_id, section) — la búsqueda
siempre filtra por curso, y opcionalmente también por sección.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "013_document_section"
down_revision = "012_interaction_logs_message_fk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("section", sa.Integer(), nullable=True))
    op.create_index(
        "ix_documents_course_id_section", "documents", ["course_id", "section"]
    )


def downgrade() -> None:
    op.drop_index("ix_documents_course_id_section", table_name="documents")
    op.drop_column("documents", "section")
