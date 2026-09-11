"""Tests de la lógica de detección de umbral de app.shared.alerting (ADR-012).

No se testea el envío real de email — se mockea `smtplib.SMTP_SSL` (vía
`_send_email_sync`, que corre en un thread aparte con `asyncio.to_thread`).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.shared import alerting


def _fake_redis(incr_result: int, set_result: bool = True) -> MagicMock:
    """Mock de Redis: la pipeline de INCR+EXPIRE devuelve `incr_result` como
    el conteo, y `redis.set(..., nx=True)` devuelve `set_result`."""
    redis_mock = MagicMock()

    pipe = MagicMock()
    pipe.incr = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[incr_result, True])
    redis_mock.pipeline = MagicMock(return_value=pipe)

    redis_mock.set = AsyncMock(return_value=set_result)
    return redis_mock


def _configured_settings() -> MagicMock:
    settings = MagicMock()
    settings.alert_smtp_host = "smtp.gmail.com"
    settings.alert_smtp_port = 465
    settings.alert_smtp_user = "nexusai.alertas@gmail.com"
    settings.alert_smtp_password = "app-password-falsa"
    settings.alert_email_to = "santiagotricherri@gmail.com"
    return settings


# ============================================================
# send_alert
# ============================================================

async def test_send_alert_sends_email_when_smtp_configured():
    with patch("app.shared.alerting.get_settings", return_value=_configured_settings()), \
         patch("app.shared.alerting._send_email_sync") as mock_send_sync:
        await alerting.send_alert("Título", "Mensaje")

    mock_send_sync.assert_called_once_with(
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        smtp_user="nexusai.alertas@gmail.com",
        smtp_password="app-password-falsa",
        email_to="santiagotricherri@gmail.com",
        subject="Título",
        body="Mensaje",
    )


async def test_send_alert_without_smtp_config_only_logs():
    settings = MagicMock()
    settings.alert_smtp_user = None
    settings.alert_smtp_password = None
    settings.alert_email_to = None

    with patch("app.shared.alerting.get_settings", return_value=settings), \
         patch("app.shared.alerting._send_email_sync") as mock_send_sync:
        await alerting.send_alert("Título", "Mensaje")

    mock_send_sync.assert_not_called()


async def test_send_alert_smtp_failure_does_not_propagate():
    with patch("app.shared.alerting.get_settings", return_value=_configured_settings()), \
         patch("app.shared.alerting._send_email_sync", side_effect=Exception("smtp unreachable")):
        # No debe lanzar.
        await alerting.send_alert("Título", "Mensaje")


# ============================================================
# record_event_and_maybe_alert — lógica de umbral
# ============================================================

async def test_alert_not_triggered_below_threshold():
    redis = _fake_redis(incr_result=2)

    with patch("app.shared.alerting.send_alert", new=AsyncMock()) as mock_send:
        triggered = await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=5,
            title="t",
            message="m",
        )

    assert triggered is False
    mock_send.assert_not_awaited()


async def test_alert_triggered_at_threshold():
    redis = _fake_redis(incr_result=5, set_result=True)

    with patch("app.shared.alerting.send_alert", new=AsyncMock()) as mock_send:
        triggered = await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=5,
            title="t",
            message="m",
        )

    assert triggered is True
    mock_send.assert_awaited_once_with("t", "m")


async def test_alert_triggered_only_once_per_window():
    """Superado el umbral, si la bandera de dedupe ya está seteada
    (`redis.set(nx=True)` devuelve False), no se vuelve a alertar aunque el
    conteo siga por encima del umbral."""
    redis = _fake_redis(incr_result=9, set_result=False)

    with patch("app.shared.alerting.send_alert", new=AsyncMock()) as mock_send:
        triggered = await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=5,
            title="t",
            message="m",
        )

    assert triggered is False
    mock_send.assert_not_awaited()


async def test_alert_swallow_redis_errors():
    """Si Redis falla (caído, timeout), no debe propagar — solo no alerta."""
    redis = MagicMock()
    redis.pipeline = MagicMock(side_effect=ConnectionError("redis caído"))

    with patch("app.shared.alerting.send_alert", new=AsyncMock()) as mock_send:
        triggered = await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=1,
            title="t",
            message="m",
        )

    assert triggered is False
    mock_send.assert_not_awaited()
