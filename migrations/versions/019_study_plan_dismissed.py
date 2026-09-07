"""Add dismissed_at a quiz_errors y student_dismissed_at a unanswered_questions — SP-13 (issue #323)

Revision ID: 019_study_plan_dismissed
Revises: 018_gaps_archived
Create Date: 2026-09-07 00:00:00.000000

SP-13: el alumno puede descartar un tema puntual del Plan de estudio sin
borrar el historial subyacente. `study_plan()` combina dos tablas (QuizError
+ UnansweredQuestion), y el descarte NO debe afectar lo que ve el docente en
Gaps/Analytics — por eso son columnas nuevas, separadas de `archived_at`
(que ya existe en unanswered_questions pero es el archivado del DOCENTE,
DOC-D08). NULL = activo. Si el alumno vuelve a errar sobre el mismo tema
más adelante, se inserta una fila nueva sin dismiss — el tema reaparece
solo, sin lógica extra en el router (ver app/quiz/router.py::study_plan).

Nombre de revisión acortado a propósito (mismo motivo que 017/018):
alembic_version.version_num es varchar(32) en Postgres.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "019_study_plan_dismissed"
down_revision = "018_gaps_archived"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    quiz_error_columns = {c["name"] for c in inspector.get_columns("quiz_errors")}
    if "dismissed_at" not in quiz_error_columns:
        op.add_column(
            "quiz_errors",
            sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        )

    unanswered_columns = {c["name"] for c in inspector.get_columns("unanswered_questions")}
    if "student_dismissed_at" not in unanswered_columns:
        op.add_column(
            "unanswered_questions",
            sa.Column("student_dismissed_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("unanswered_questions", "student_dismissed_at")
    op.drop_column("quiz_errors", "dismissed_at")
