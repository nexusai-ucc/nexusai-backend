"""Tablas flashcards + flashcard_reviews — repetición espaciada SM-2 (SP-11, issue #315)

Revision ID: 020_flashcards_spaced_rep
Revises: 019_study_plan_dismissed
Create Date: 2026-09-07 00:00:00.000000

Las flashcards (question_type='flashcard' en /quiz/generate) hoy son
100% efímeras: un lote nuevo por LLM en cada request, sin ID ni tabla
propia. `flashcards` les da identidad estable (dedup por course_id +
content_hash); `flashcard_reviews` guarda el estado SM-2 por alumno+
flashcard (ease_factor, interval_days, repetitions, next_review_at).

Nombre de revisión acortado a propósito (mismo motivo que 017/018/019):
alembic_version.version_num es varchar(32) en Postgres.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "020_flashcards_spaced_rep"
down_revision = "019_study_plan_dismissed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flashcards",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(200), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("source_filename", sa.String(255), nullable=True),
        sa.Column("source_document_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("course_id", "content_hash", name="uq_flashcards_course_content_hash"),
    )
    op.create_index("ix_flashcards_course_id", "flashcards", ["course_id"])

    op.create_table(
        "flashcard_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "flashcard_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("flashcards.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("ease_factor", sa.Float(), nullable=False, server_default="2.5"),
        sa.Column("interval_days", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("repetitions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_review_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("flashcard_id", "user_id", name="uq_flashcard_reviews_flashcard_user"),
    )
    op.create_index(
        "ix_flashcard_reviews_user_id_next_review_at",
        "flashcard_reviews",
        ["user_id", "next_review_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_flashcard_reviews_user_id_next_review_at", table_name="flashcard_reviews")
    op.drop_table("flashcard_reviews")
    op.drop_index("ix_flashcards_course_id", table_name="flashcards")
    op.drop_table("flashcards")
