"""Tabla forum_webhook_configs — URL de webhook por curso para el digest semanal del foro (FOR-07, issue #378)

Revision ID: 022_forum_webhook_configs
Revises: 021_message_feedback
Create Date: 2026-09-08 00:00:00.000000

El plugin Moodle no tiene ninguna tabla de configuración por-curso propia
(solo un placeholder en install.xml, nunca usado, y no existe
db/upgrade.php) — mismo criterio que CalendarAlert (CAL-02): la config
vive acá, en el backend, guardada/leída vía un endpoint FastAPI llamado
desde una external function PHP nueva.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "022_forum_webhook_configs"
down_revision = "021_message_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forum_webhook_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("webhook_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("course_id", name="uq_forum_webhook_configs_course"),
    )


def downgrade() -> None:
    op.drop_table("forum_webhook_configs")
