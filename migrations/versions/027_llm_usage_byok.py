"""llm_usage.byok: llamadas hechas con la key propia del usuario

Revision ID: 027_llm_usage_byok
Revises: 026_forum_post_visibility
Create Date: 2026-11-18 12:00:00.000000

Un alumno o docente puede cargar la API key de su propia cuenta como último
eslabón de la cadena de modelos (ver app/providers/user_llm.py). Esas llamadas
quedan registradas con byok=true y costo 0: las paga el usuario, no la
facultad. Las filas existentes quedan en false.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "027_llm_usage_byok"
down_revision = "026_forum_post_visibility"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_usage",
        sa.Column("byok", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("llm_usage", "byok")
