"""Add quiz_attempts table — SP-09 (historial de quizzes por alumno)

Revision ID: 013_quiz_attempts
Revises: 012_interaction_logs_message_fk
Create Date: 2026-07-15 00:00:00.000000

Registra cada sesión de quiz completada por un alumno: tipo de pregunta,
dificultad, puntaje y fecha. Permite mostrar el historial de práctica en
el QuizPanel y detectar tendencias de estudio a lo largo del tiempo.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "013_quiz_attempts"
down_revision = "012_interaction_logs_message_fk"
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
            sa.Column("question_type", sa.String(length=20), nullable=False),
            sa.Column("difficulty", sa.String(length=10), nullable=False, server_default="medium"),
            sa.Column("topic", sa.String(length=200), nullable=True),
            sa.Column("total_questions", sa.Integer(), nullable=False),
            sa.Column("correct_count", sa.Integer(), nullable=False, server_default="0"),
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


def downgrade() -> None:
    op.drop_index("ix_quiz_attempts_user_id_course_id_created_at", "quiz_attempts")
    op.drop_table("quiz_attempts")
