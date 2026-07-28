"""Crear tabla calendar_alerts — CAL-02

Revision ID: 013_calendar_alerts
Revises: 012_quiz_attempts
Create Date: 2026-07-24 00:00:00.000000

Alertas de calendario configurables por el alumno: un registro por (user_id, event_id)
con columna notified para que el cron PHP no duplique notificaciones.

Nota: down_revision apunta a 012_quiz_attempts (rama feature/sp04-sp09-fill-blank-historial)
porque esa migración llegó primero a development. La migración 012_interaction_logs_message_fk
quedó absorbida en esa cadena.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "013_calendar_alerts"
down_revision = "012_quiz_attempts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "calendar_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("event_name", sa.String(200), nullable=False),
        sa.Column("event_timestamp", sa.Integer(), nullable=False),
        sa.Column("days_before", sa.Integer(), nullable=False),
        sa.Column("notified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("user_id", "event_id", name="uq_calendar_alerts_user_event"),
    )
    op.create_index(
        "ix_calendar_alerts_user_course",
        "calendar_alerts",
        ["user_id", "course_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_calendar_alerts_user_course", table_name="calendar_alerts")
    op.drop_table("calendar_alerts")
