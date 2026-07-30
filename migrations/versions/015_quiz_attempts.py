"""Add quiz_attempts table — SP-09 + ANALYTICS-01 (historial y métricas de quizzes)

Revision ID: 015_quiz_attempts
Revises: 014_calendar_alerts
Create Date: 2026-07-30 00:00:00.000000

Unifica SP-09 (historial de práctica del alumno: question_type, difficulty, topic)
con ANALYTICS-01 (histograma de puntajes por curso: score). El score se
calcula server-side a partir de correct_answers/total_questions.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "015_quiz_attempts"
down_revision = "014_calendar_alerts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "quiz_attempts" not in existing_tables:
        op.create_table(
            "quiz_attempts",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("course_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("question_type", sa.String(length=20), nullable=True),
            sa.Column("difficulty", sa.String(length=10), nullable=False, server_default="medium"),
            sa.Column("topic", sa.String(length=200), nullable=True),
            sa.Column("total_questions", sa.Integer(), nullable=False),
            sa.Column("correct_answers", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("quiz_attempts")}
    if "ix_quiz_attempts_user_id_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_quiz_attempts_user_id_course_id_created_at",
            "quiz_attempts",
            ["user_id", "course_id", "created_at"],
        )
    if "ix_quiz_attempts_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_quiz_attempts_course_id_created_at",
            "quiz_attempts",
            ["course_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_quiz_attempts_course_id_created_at", "quiz_attempts")
    op.drop_index("ix_quiz_attempts_user_id_course_id_created_at", "quiz_attempts")
    op.drop_table("quiz_attempts")
