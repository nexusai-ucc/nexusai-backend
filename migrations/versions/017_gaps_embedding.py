"""Add embedding column a unanswered_questions — DOC-D06 (issue #313)

Revision ID: 017_gaps_embedding
Revises: 016_quiz_attempts_soft_delete
Create Date: 2026-08-08 00:00:00.000000

DOC-D06: el panel de Gaps del docente agrupaba preguntas sin respuesta por
texto normalizado (lower+trim), así que dos preguntas que dicen lo mismo
con palabras distintas aparecían como gaps separados. Esta columna guarda
el embedding de cada pregunta para que el router pueda agruparlas por
similitud coseno en vez de texto literal.

Nota sobre el nombre de la revisión: el ID original
"017_unanswered_questions_embedding" (34 caracteres) rompía la migración
al final — `alembic_version.version_num` es `varchar(32)` en Postgres, y
recién falla en el UPDATE final, después de ya haber aplicado el ALTER
TABLE. Nombre acortado para entrar en el límite.

A diferencia de forum_post_embeddings (migración 009), acá NO se agrega un
índice HNSW: el uso no es "buscar el top-K más parecido a una query nueva"
(para lo que HNSW sirve), es "clustering completo de todas las filas del
curso entre sí", que se hace en Python sobre el resultado ya traído — un
índice ANN no aporta nada ahí.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "017_gaps_embedding"
down_revision = "016_quiz_attempts_soft_delete"
branch_labels = None
depends_on = None

# Igual que chunks.embedding / forum_post_embeddings.embedding (gemini-embedding-001)
_EMBEDDING_DIM = 768


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("unanswered_questions")}

    if "embedding" not in columns:
        op.add_column(
            "unanswered_questions",
            sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("unanswered_questions", "embedding")
