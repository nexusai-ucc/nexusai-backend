"""Tests de app.shared.error_monitoring (ADR-012): qué eventos disparan qué
alerta y con qué umbral. `record_event_and_maybe_alert` (el mecanismo de
ventana fija) ya se testea en test_alerting.py — acá se testea el ruteo:
status code / tipo de excepción / latencia → el key_prefix y umbral correctos.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.shared import error_monitoring


def _rate_limit_error() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return openai.RateLimitError("quota exhausted", response=response, body=None)


def _server_error() -> openai.InternalServerError:
    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    return openai.InternalServerError("service unavailable", response=response, body=None)


# ============================================================
# record_5xx_and_maybe_alert
# ============================================================

async def test_5xx_below_500_is_ignored():
    redis = MagicMock()

    with patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock()) as mock_record:
        result = await error_monitoring.record_5xx_and_maybe_alert(
            redis, status_code=404, path="/api/v1/chat/messages"
        )

    assert result is False
    mock_record.assert_not_awaited()


async def test_5xx_delegates_to_generic_threshold_tracker():
    redis = MagicMock()

    with patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock(return_value=True)) as mock_record:
        result = await error_monitoring.record_5xx_and_maybe_alert(
            redis, status_code=500, path="/api/v1/chat/messages"
        )

    assert result is True
    mock_record.assert_awaited_once()
    _, kwargs = mock_record.call_args
    assert kwargs["key_prefix"].startswith("nexusai:alerts:5xx:")
    assert "/api/v1/chat/messages" in kwargs["message"]


# ============================================================
# record_llm_failure_and_maybe_alert
# ============================================================

async def test_llm_rate_limit_error_uses_quota_alert():
    """Un RateLimitError (cuota agotada en toda la cadena) usa el prefijo y
    umbral de cuota, NO el de fallas genéricas del LLM."""
    redis = MagicMock()

    with patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock(return_value=True)) as mock_record:
        result = await error_monitoring.record_llm_failure_and_maybe_alert(
            redis, endpoint="messages", error=_rate_limit_error()
        )

    assert result is True
    mock_record.assert_awaited_once()
    _, kwargs = mock_record.call_args
    assert kwargs["key_prefix"].startswith("nexusai:alerts:llm_quota:")
    assert "cuota" in kwargs["title"].lower()


async def test_llm_generic_failure_uses_failure_alert():
    """Un error que no es RateLimitError (ej. servidor caído) usa el
    prefijo genérico de fallas del LLM."""
    redis = MagicMock()

    with patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock(return_value=True)) as mock_record:
        result = await error_monitoring.record_llm_failure_and_maybe_alert(
            redis, endpoint="stream", error=_server_error()
        )

    assert result is True
    mock_record.assert_awaited_once()
    _, kwargs = mock_record.call_args
    assert kwargs["key_prefix"].startswith("nexusai:alerts:llm_failure:")
    assert "InternalServerError" in kwargs["message"]


# ============================================================
# record_llm_slow_and_maybe_alert
# ============================================================

async def test_llm_latency_below_threshold_is_ignored():
    redis = MagicMock()

    with patch("app.shared.error_monitoring.get_settings") as mock_settings, \
         patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock()) as mock_record:
        mock_settings.return_value.llm_slow_threshold_ms = 15_000
        result = await error_monitoring.record_llm_slow_and_maybe_alert(
            redis, endpoint="messages", latency_ms=500.0
        )

    assert result is False
    mock_record.assert_not_awaited()


async def test_llm_latency_above_threshold_is_counted():
    redis = MagicMock()

    with patch("app.shared.error_monitoring.get_settings") as mock_settings, \
         patch("app.shared.error_monitoring.record_event_and_maybe_alert", new=AsyncMock(return_value=True)) as mock_record:
        mock_settings.return_value.llm_slow_threshold_ms = 15_000
        mock_settings.return_value.llm_slow_window_sec = 300
        mock_settings.return_value.llm_slow_threshold_count = 5
        result = await error_monitoring.record_llm_slow_and_maybe_alert(
            redis, endpoint="messages", latency_ms=20_000.0
        )

    assert result is True
    mock_record.assert_awaited_once()
    _, kwargs = mock_record.call_args
    assert kwargs["key_prefix"].startswith("nexusai:alerts:llm_slow:")
