"""Tests de scripts/health_check_alert.py (ADR-012): el watchdog externo de
/health que corre por cron. Es un script standalone (no vive en app.*), así
que se carga directo desde el path por importlib en vez de un import normal.

No se testea un email real ni una conexión de red real — se mockea
`urllib.request.urlopen` (chequeo de /health) y `smtplib.SMTP_SSL` (envío).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "health_check_alert.py"
_spec = importlib.util.spec_from_file_location("health_check_alert", _SCRIPT_PATH)
health_check_alert = importlib.util.module_from_spec(_spec)
sys.modules["health_check_alert"] = health_check_alert
_spec.loader.exec_module(health_check_alert)


# ============================================================
# should_alert — cruce de umbral, sin repetir mientras siga caído
# ============================================================

@pytest.mark.parametrize(
    "consecutive_failures,threshold,already_alerted,expected",
    [
        (1, 3, False, False),   # todavía no cruzó el umbral
        (2, 3, False, False),
        (3, 3, False, True),    # cruza el umbral por primera vez
        (4, 3, True, False),    # sigue caído pero ya se avisó — no repetir
        (10, 3, True, False),
    ],
)
def test_should_alert(consecutive_failures, threshold, already_alerted, expected):
    assert (
        health_check_alert.should_alert(consecutive_failures, threshold, already_alerted)
        == expected
    )


# ============================================================
# check_health
# ============================================================

def test_check_health_true_on_200():
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("health_check_alert.urllib.request.urlopen", return_value=fake_response):
        assert health_check_alert.check_health("http://x/health", timeout=1.0) is True


def test_check_health_false_on_non_200():
    fake_response = MagicMock()
    fake_response.status = 500
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("health_check_alert.urllib.request.urlopen", return_value=fake_response):
        assert health_check_alert.check_health("http://x/health", timeout=1.0) is False


def test_check_health_false_on_connection_error():
    with patch(
        "health_check_alert.urllib.request.urlopen",
        side_effect=ConnectionRefusedError("connection refused"),
    ):
        assert health_check_alert.check_health("http://x/health", timeout=1.0) is False


# ============================================================
# read_state / write_state
# ============================================================

def test_read_state_missing_file_returns_default(tmp_path):
    state = health_check_alert.read_state(tmp_path / "does_not_exist.json")
    assert state == {"consecutive_failures": 0, "alerted": False}


def test_read_state_corrupt_file_returns_default(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json{{{")
    assert health_check_alert.read_state(path) == {"consecutive_failures": 0, "alerted": False}


def test_write_then_read_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    health_check_alert.write_state(path, {"consecutive_failures": 4, "alerted": True})
    assert health_check_alert.read_state(path) == {"consecutive_failures": 4, "alerted": True}


# ============================================================
# send_email
# ============================================================

def _smtp_config(**overrides) -> dict:
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 465,
        "smtp_user": "nexusai.alertas@gmail.com",
        "smtp_password": "app-password-falsa",
        "email_to": "santiagotricherri@gmail.com",
        "timeout_sec": 1.0,
    }
    config.update(overrides)
    return config


def test_send_email_noop_without_smtp_config(capsys):
    config = _smtp_config(smtp_user="", smtp_password="", email_to="")

    with patch("health_check_alert.smtplib.SMTP_SSL") as mock_smtp_cls:
        health_check_alert.send_email(config, "asunto", "cuerpo del mensaje")

    mock_smtp_cls.assert_not_called()
    captured = capsys.readouterr()
    assert "cuerpo del mensaje" in captured.err


def test_send_email_logs_in_and_sends_message():
    config = _smtp_config()

    with patch("health_check_alert.smtplib.SMTP_SSL") as mock_smtp_cls:
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_smtp

        health_check_alert.send_email(config, "asunto", "cuerpo del mensaje")

    mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 465, timeout=1.0)
    mock_smtp.login.assert_called_once_with("nexusai.alertas@gmail.com", "app-password-falsa")
    mock_smtp.send_message.assert_called_once()
    sent_msg = mock_smtp.send_message.call_args[0][0]
    assert sent_msg["To"] == "santiagotricherri@gmail.com"
    assert sent_msg["Subject"] == "asunto"


def test_send_email_failure_does_not_raise():
    config = _smtp_config()

    with patch("health_check_alert.smtplib.SMTP_SSL", side_effect=Exception("smtp unreachable")):
        # No debe lanzar.
        health_check_alert.send_email(config, "asunto", "cuerpo")


# ============================================================
# main — integración de las piezas de arriba
# ============================================================

def _config(tmp_path, **overrides):
    config = {
        "health_url": "http://x/health",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 465,
        "smtp_user": "nexusai.alertas@gmail.com",
        "smtp_password": "app-password-falsa",
        "email_to": "santiagotricherri@gmail.com",
        "failure_threshold": 3,
        "timeout_sec": 1.0,
        "state_file": tmp_path / "state.json",
    }
    config.update(overrides)
    return config


def test_main_healthy_resets_state(tmp_path):
    config_path = tmp_path / "state.json"
    health_check_alert.write_state(config_path, {"consecutive_failures": 2, "alerted": False})

    with patch("health_check_alert._env_config", return_value=_config(tmp_path, state_file=config_path)), \
         patch("health_check_alert.check_health", return_value=True), \
         patch("health_check_alert.send_email") as mock_send:
        exit_code = health_check_alert.main()

    assert exit_code == 0
    assert health_check_alert.read_state(config_path) == {"consecutive_failures": 0, "alerted": False}
    mock_send.assert_not_called()


def test_main_recovery_sends_notice(tmp_path):
    config_path = tmp_path / "state.json"
    health_check_alert.write_state(config_path, {"consecutive_failures": 5, "alerted": True})

    with patch("health_check_alert._env_config", return_value=_config(tmp_path, state_file=config_path)), \
         patch("health_check_alert.check_health", return_value=True), \
         patch("health_check_alert.send_email") as mock_send:
        health_check_alert.main()

    mock_send.assert_called_once()
    assert "recuperó" in mock_send.call_args[0][1]


def test_main_failure_below_threshold_does_not_alert(tmp_path):
    config_path = tmp_path / "state.json"

    with patch("health_check_alert._env_config", return_value=_config(tmp_path, state_file=config_path, failure_threshold=3)), \
         patch("health_check_alert.check_health", return_value=False), \
         patch("health_check_alert.send_email") as mock_send:
        exit_code = health_check_alert.main()

    assert exit_code == 1
    assert health_check_alert.read_state(config_path)["consecutive_failures"] == 1
    mock_send.assert_not_called()


def test_main_failure_crossing_threshold_alerts_once(tmp_path):
    config_path = tmp_path / "state.json"
    health_check_alert.write_state(config_path, {"consecutive_failures": 2, "alerted": False})

    with patch("health_check_alert._env_config", return_value=_config(tmp_path, state_file=config_path, failure_threshold=3)), \
         patch("health_check_alert.check_health", return_value=False), \
         patch("health_check_alert.send_email") as mock_send:
        health_check_alert.main()

    mock_send.assert_called_once()
    assert health_check_alert.read_state(config_path) == {"consecutive_failures": 3, "alerted": True}


def test_main_stays_down_does_not_realert(tmp_path):
    config_path = tmp_path / "state.json"
    health_check_alert.write_state(config_path, {"consecutive_failures": 5, "alerted": True})

    with patch("health_check_alert._env_config", return_value=_config(tmp_path, state_file=config_path, failure_threshold=3)), \
         patch("health_check_alert.check_health", return_value=False), \
         patch("health_check_alert.send_email") as mock_send:
        health_check_alert.main()

    mock_send.assert_not_called()
