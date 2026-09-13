"""Tests de la lógica de detección de umbral de app.shared.alerting (ADR-012).

No se testea el envío real de email — se mockea `smtplib.SMTP_SSL` (vía
`_send_email_sync`, que corre en un thread aparte con `asyncio.to_thread`).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

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
        # send_alert se dispara fire-and-forget (asyncio.create_task) — hace
        # falta cederle el loop al menos una vez para que la task programada
        # llegue a correr antes de poder verificar que se llamó.
        await asyncio.sleep(0)

    assert triggered is True
    mock_send.assert_awaited_once_with("t", "m")


async def test_alert_send_is_fire_and_forget_does_not_block_caller():
    """record_event_and_maybe_alert no debe esperar a que send_alert
    termine — es la razón de ser del fix (antes: `await` inline agregaba
    la latencia completa del envío de email al request que cruzó el
    umbral, justo cuando ese request ya viene degradado)."""
    redis = _fake_redis(incr_result=5, set_result=True)

    started = asyncio.Event()

    async def _slow_send_alert(title, message):
        started.set()
        await asyncio.sleep(10)  # nunca debería bloquear al caller

    with patch("app.shared.alerting.send_alert", new=_slow_send_alert):
        start = time.perf_counter()
        triggered = await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=5,
            title="t",
            message="m",
        )
        elapsed = time.perf_counter() - start

    assert triggered is True
    assert elapsed < 1.0, "record_event_and_maybe_alert esperó al envío del email"

    # Limpieza: cancelamos la task de background que quedó durmiendo 10s
    # para no dejarla colgada entre tests.
    for task in list(alerting._background_tasks):
        task.cancel()


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


async def test_cooldown_key_has_no_window_bucket():
    """Hallazgo de audit (alert storm): antes, la bandera de dedupe estaba
    bucketeada por ventana (`{key_prefix}:alerted:{bucket}`) — en una caída
    sostenida, cada ventana nueva que cruzaba el umbral volvía a alertar
    (hasta ~60/hora con window_sec=60). La key de cooldown ahora es fija
    por `key_prefix` (sin bucket), así que persiste a través de varias
    ventanas seguidas en vez de resetearse en cada una."""
    redis = _fake_redis(incr_result=10, set_result=True)

    with patch("app.shared.alerting.send_alert", new=AsyncMock()):
        await alerting.record_event_and_maybe_alert(
            redis,
            key_prefix="test:key",
            window_sec=60,
            threshold=5,
            title="t",
            message="m",
        )
        await asyncio.sleep(0)

    cooldown_key = redis.set.call_args.args[0]
    assert cooldown_key == "test:key:cooldown"

    count_key = redis.pipeline.return_value.incr.call_args.args[0]
    assert count_key != cooldown_key
    assert count_key.startswith("test:key:count:")


async def test_second_window_within_cooldown_does_not_realert():
    """Simula el escenario completo del alert storm: dos ventanas
    CONSECUTIVAS (distinto bucket) del mismo `key_prefix`, ambas cruzando
    el umbral — la segunda no debe alertar porque el cooldown de la
    primera sigue activo, aunque sea una ventana de conteo distinta."""
    redis = _fake_redis(incr_result=5, set_result=True)  # primera ventana: dispara

    with patch("app.shared.alerting.send_alert", new=AsyncMock()) as mock_send:
        first = await alerting.record_event_and_maybe_alert(
            redis, key_prefix="test:key", window_sec=60, threshold=5,
            title="t", message="m",
        )
        await asyncio.sleep(0)

        # Segunda ventana: el contador vuelve a superar el umbral, pero
        # `redis.set(nx=True)` ahora devuelve False — el cooldown de la
        # primera alerta sigue activo (mismo key_prefix, TTL=alert_cooldown_sec
        # que es mucho mayor a window_sec en la config real).
        redis.set = AsyncMock(return_value=False)
        second = await alerting.record_event_and_maybe_alert(
            redis, key_prefix="test:key", window_sec=60, threshold=5,
            title="t", message="m",
        )
        await asyncio.sleep(0)

    assert first is True
    assert second is False
    mock_send.assert_awaited_once()


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
