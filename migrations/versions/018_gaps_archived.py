"""Add archived_at a unanswered_questions — DOC-D08 (issue #383)

Revision ID: 018_gaps_archived
Revises: 017_gaps_embedding
Create Date: 2026-08-10 00:00:00.000000

DOC-D08: el docente archiva un gap desde el panel una vez que ya lo resolvió
(agregó material, o decidió que no es relevante), para que no se acumule
indefinidamente en la vista por default. NULL = activo. Si un alumno vuelve
a preguntar algo equivalente más adelante, se inserta una fila nueva con
archived_at=NULL — el gap reaparece solo, sin ninguna lógica extra en el
router (ver app/gaps/router.py).

Nombre de revisión acortado a propósito (mismo motivo que 017_gaps_embedding):
alembic_version.version_num es varchar(32) en Postgres.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018_gaps_archived"
down_revision = "017_gaps_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("unanswered_questions")}

    if "archived_at" not in columns:
        op.add_column(
            "unanswered_questions",
            sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("unanswered_questions", "archived_at")
