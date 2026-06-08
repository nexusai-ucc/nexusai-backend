"""Add unanswered_questions table — Feature G (gaps detection)

Revision ID: 006_unanswered_questions
Revises: 005_document_storage_path
Create Date: 2026-05-28 19:00:00.000000

Tabla que registra preguntas del alumno que el material del curso no pudo
responder bien (chunks recuperados = 0 o todos con similarity < umbral).

El docente consulta esta tabla para descubrir gaps en su material indexado.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "006_unanswered_questions"
down_revision = "005_document_storage_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    if "unanswered_questions" not in existing_tables:
        op.create_table(
            "unanswered_questions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("course_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("max_similarity", sa.Float(), nullable=True),
            sa.Column("chunks_retrieved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("unanswered_questions")}
    if "ix_unanswered_questions_course_id_created_at" not in existing_indexes:
        op.create_index(
            "ix_unanswered_questions_course_id_created_at",
            "unanswered_questions",
            ["course_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_unanswered_questions_course_id_created_at", "unanswered_questions")
    op.drop_table("unanswered_questions")
