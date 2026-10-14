"""Registro de consumo por llamada a proveedor: llm_usage, model_prices y llm_usage_daily (COST-01, issue #519)

Revision ID: 023_llm_usage
Revises: 022_forum_webhook_configs
Create Date: 2026-09-26 00:00:00.000000

Hasta acá solo el chat guardaba tokens (messages, interaction_logs). Estas
tablas registran todas las llamadas a proveedores de IA (LLM, embeddings,
voz), con el modelo que respondió de verdad y su costo, y resumen por día el
detalle viejo. Ver app/shared/usage_ledger.py.

Precios: solo se siembra gpt-4o-mini, el único documentado en
investigacion/03-openai/costos-rate-limits.md. El resto se carga con
scripts/model_prices.py; un modelo sin precio deja cost_usd en NULL y
dispara una alerta diaria.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "023_llm_usage"
down_revision = "022_forum_webhook_configs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("feature", sa.String(80), nullable=False),
        sa.Column("provider", sa.String(40), nullable=True),
        sa.Column("model", sa.String(120), nullable=True),
        sa.Column("fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "cached_prompt_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("embedding_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_seconds", sa.Float(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("saved_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("course_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(10), nullable=False, server_default="unknown"),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("client_id", sa.String(32), nullable=True),
    )
    op.create_index("ix_llm_usage_created_at", "llm_usage", ["created_at"])
    op.create_index(
        "ix_llm_usage_course_id_created_at", "llm_usage", ["course_id", "created_at"]
    )
    op.create_index(
        "ix_llm_usage_feature_created_at", "llm_usage", ["feature", "created_at"]
    )

    prices = op.create_table(
        "model_prices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("provider", sa.String(40), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column(
            "input_per_mtok", sa.Numeric(12, 6), nullable=False, server_default="0"
        ),
        sa.Column(
            "output_per_mtok", sa.Numeric(12, 6), nullable=False, server_default="0"
        ),
        sa.Column("cached_input_per_mtok", sa.Numeric(12, 6), nullable=True),
        sa.Column(
            "embedding_per_mtok", sa.Numeric(12, 6), nullable=False, server_default="0"
        ),
        sa.Column(
            "audio_per_minute", sa.Numeric(12, 6), nullable=False, server_default="0"
        ),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "provider",
            "model",
            "valid_from",
            name="uq_model_prices_provider_model_from",
        ),
    )
    op.bulk_insert(
        prices,
        [
            {
                "id": uuid.uuid4(),
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_per_mtok": 0.15,
                "output_per_mtok": 0.60,
                "cached_input_per_mtok": 0.075,
                "embedding_per_mtok": 0,
                "audio_per_minute": 0,
                "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ],
    )

    op.create_table(
        "llm_usage_daily",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("client_id", sa.String(32), nullable=False, server_default=""),
        sa.Column("course_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("role", sa.String(10), nullable=False, server_default="unknown"),
        sa.Column("feature", sa.String(80), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("provider", sa.String(40), nullable=False, server_default=""),
        sa.Column("model", sa.String(120), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "completion_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "cached_prompt_tokens", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("embedding_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=False, server_default="0"),
        sa.Column("cache_hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("saved_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint(
            "day",
            "client_id",
            "course_id",
            "role",
            "feature",
            "kind",
            "provider",
            "model",
            "status",
            name="uq_llm_usage_daily_key",
        ),
    )


def downgrade() -> None:
    op.drop_table("llm_usage_daily")
    op.drop_table("model_prices")
    op.drop_index("ix_llm_usage_feature_created_at", table_name="llm_usage")
    op.drop_index("ix_llm_usage_course_id_created_at", table_name="llm_usage")
    op.drop_index("ix_llm_usage_created_at", table_name="llm_usage")
    op.drop_table("llm_usage")
