"""forum_post_embeddings.cmid y group_id: visibilidad de los posts de foro (VIS-06, issue #541)

Revision ID: 026_forum_post_visibility
Revises: 025_documents_cmid
Create Date: 2026-09-27 12:00:00.000000

Cada post indexado guarda el foro (cmid de la actividad) y el grupo de su
discusión. El plugin manda en cada pedido los foros que el usuario puede ver y
sus grupos, y solo se usan posts de foros visibles y de discusiones sin grupo o
de un grupo del usuario. Los posts que ya estaban indexados quedan sin cmid y
dejan de mostrarse hasta que se vuelvan a indexar.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "026_forum_post_visibility"
down_revision = "025_documents_cmid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "forum_post_embeddings", sa.Column("cmid", sa.Integer(), nullable=True)
    )
    op.add_column(
        "forum_post_embeddings", sa.Column("group_id", sa.Integer(), nullable=True)
    )
    op.create_index("ix_forum_post_embeddings_cmid", "forum_post_embeddings", ["cmid"])


def downgrade() -> None:
    op.drop_index("ix_forum_post_embeddings_cmid", table_name="forum_post_embeddings")
    op.drop_column("forum_post_embeddings", "group_id")
    op.drop_column("forum_post_embeddings", "cmid")
