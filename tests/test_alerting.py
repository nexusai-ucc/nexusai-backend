"""Tests de la lógica de detección de umbral de app.shared.alerting (ADR-012).

No se testea el webhook real — se mockea `httpx.AsyncClient` igual que en
test_forums_router.py.
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


# ============================================================
# send_alert
# ============================================================

async def test_send_alert_posts_to_configured_webhook():
    with patch("app.shared.alerting.get_settings") as mock_settings, \
         patch("app.shared.alerting.httpx.AsyncClient") as mock_client_cls:
        mock_settings.return_value.alert_webhook_url = "https://hooks.slack.com/services/xxx"
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        await alerting.send_alert("Título", "Mensaje")

    mock_client.post.assert_awaited_once_with(
        "https://hooks.slack.com/services/xxx",
        json={"text": "Título: Mensaje", "content": "Título: Mensaje"},
    )


async def test_send_alert_without_webhook_url_only_logs():
    with patch("app.shared.alerting.get_settings") as mock_settings, \
         patch("app.shared.alerting.httpx.AsyncClient") as mock_client_cls:
        mock_settings.return_value.alert_webhook_url = None

        await alerting.send_alert("Título", "Mensaje")

    mock_client_cls.assert_not_called()


async def test_send_alert_webhook_failure_does_not_propagate():
    with patch("app.shared.alerting.get_settings") as mock_settings, \
         patch("app.shared.alerting.httpx.AsyncClient") as mock_client_cls:
        mock_settings.return_value.alert_webhook_url = "https://hooks.slack.com/services/xxx"
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("webhook host unreachable")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

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
