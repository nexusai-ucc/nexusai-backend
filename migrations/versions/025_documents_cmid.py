"""documents.cmid: actividad de Moodle de la que salió cada documento (VIS-01, issue #536)

Revision ID: 025_documents_cmid
Revises: 024_document_summaries
Create Date: 2026-09-27 00:00:00.000000

Cada documento guarda el id de la actividad de Moodle (`cmid`) de la que salió.
El plugin manda en cada pedido los `cmid` que el usuario puede ver y el
backend solo usa documentos cuyo `cmid` está en esa lista: así el material
oculto, restringido o borrado en Moodle no aparece en las respuestas. Ver
app/shared/visibility.py.

Los documentos existentes quedan con `cmid` NULL y dejan de entrar en las
respuestas hasta que se vuelvan a subir como actividad del aula.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "025_documents_cmid"
down_revision = "024_document_summaries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("cmid", sa.Integer(), nullable=True))
    op.create_index("ix_documents_cmid", "documents", ["cmid"])


def downgrade() -> None:
    op.drop_index("ix_documents_cmid", table_name="documents")
    op.drop_column("documents", "cmid")
