#!/usr/bin/env python3
"""Watchdog externo de /health — NexusAI (ver docs/adr/012-alertas-monitoreo-minimo.md).

Por qué un script aparte y no algo dentro de FastAPI: si el proceso de
uvicorn muere, se cuelga, o el container ni siquiera levanta, nada que corra
DENTRO de ese proceso puede avisar de su propia caída. Necesita correr
afuera — acá, vía cron en la misma VM (o en la otra VM de Oracle,
apuntando a la URL pública, para no depender de que la VM caída sea la que
hace el chequeo).

Sin dependencias de terceros: solo stdlib (urllib + smtplib), para no
requerir instalar nada además de Python 3 en una VM de 1 core / 1GB RAM.

Uso típico (crontab -e), cada minuto:

    * * * * * NEXUSAI_HEALTH_URL=https://api.<ip>.nip.io/health \
              NEXUSAI_ALERT_SMTP_USER=tu-cuenta@gmail.com \
              NEXUSAI_ALERT_SMTP_PASSWORD=xxxxxxxxxxxxxxxx \
              /usr/bin/python3 /opt/nexusai/services/api/scripts/health_check_alert.py \
              >> /var/log/nexusai-health.log 2>&1

`NEXUSAI_ALERT_SMTP_PASSWORD` es una App Password de Gmail (Configuración de
la cuenta de Google → Seguridad → Verificación en 2 pasos → Contraseñas de
aplicaciones), NO la contraseña normal de la cuenta — Gmail bloquea el login
SMTP con la contraseña de la cuenta directamente.

Estado entre corridas: cada invocación de cron es un proceso nuevo sin
memoria compartida, así que las fallas consecutivas y si ya se avisó se
guardan en un archivo JSON chico (NEXUSAI_HEALTH_STATE_FILE).

Variables de entorno:
    NEXUSAI_HEALTH_URL               default: http://localhost:8001/health
    NEXUSAI_ALERT_SMTP_HOST          default: smtp.gmail.com
    NEXUSAI_ALERT_SMTP_PORT          default: 465
    NEXUSAI_ALERT_SMTP_USER          default: "" (sin esto + password, solo loguea a stderr)
    NEXUSAI_ALERT_SMTP_PASSWORD      default: ""
    NEXUSAI_ALERT_EMAIL_TO           default: santiagotricherri@gmail.com
    NEXUSAI_HEALTH_FAILURE_THRESHOLD default: 3   (chequeos consecutivos fallidos antes de avisar)
    NEXUSAI_HEALTH_TIMEOUT_SEC       default: 5
    NEXUSAI_HEALTH_STATE_FILE        default: /tmp/nexusai_health_watch_state.json
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
import urllib.request
from email.message import EmailMessage
from pathlib import Path


def _env_config() -> dict:
    return {
        "health_url": os.environ.get("NEXUSAI_HEALTH_URL", "http://localhost:8001/health"),
        "smtp_host": os.environ.get("NEXUSAI_ALERT_SMTP_HOST", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("NEXUSAI_ALERT_SMTP_PORT", "465")),
        "smtp_user": os.environ.get("NEXUSAI_ALERT_SMTP_USER", ""),
        "smtp_password": os.environ.get("NEXUSAI_ALERT_SMTP_PASSWORD", ""),
        "email_to": os.environ.get("NEXUSAI_ALERT_EMAIL_TO", "santiagotricherri@gmail.com"),
        "failure_threshold": int(os.environ.get("NEXUSAI_HEALTH_FAILURE_THRESHOLD", "3")),
        "timeout_sec": float(os.environ.get("NEXUSAI_HEALTH_TIMEOUT_SEC", "5")),
        "state_file": Path(
            os.environ.get("NEXUSAI_HEALTH_STATE_FILE", "/tmp/nexusai_health_watch_state.json")
        ),
    }


def check_health(url: str, timeout: float) -> bool:
    """True si `url` respondió 200. False ante cualquier otra cosa: timeout,
    conexión rechazada, DNS roto, 4xx/5xx."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def read_state(path: Path) -> dict:
    """Estado previo: fallas consecutivas ya contadas y si ya se disparó la
    alerta para la racha actual. Estado vacío (primera corrida, archivo
    corrupto, o borrado a mano) equivale a "todo OK hasta ahora"."""
    try:
        data = json.loads(path.read_text())
        return {
            "consecutive_failures": int(data.get("consecutive_failures", 0)),
            "alerted": bool(data.get("alerted", False)),
        }
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return {"consecutive_failures": 0, "alerted": False}


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state))


def should_alert(consecutive_failures: int, threshold: int, already_alerted: bool) -> bool:
    """Dispara la alerta solo al CRUZAR el umbral, no en cada corrida
    mientras el backend siga caído — si no, con cron cada 1 min se floodea
    la casilla durante toda la caída."""
    return consecutive_failures >= threshold and not already_alerted


def send_email(config: dict, subject: str, body: str) -> None:
    """Best-effort: si falla el envío (o falta configuración SMTP), se
    loguea a stderr en vez de propagar — el watchdog no debe romper por esto."""
    if not (config["smtp_user"] and config["smtp_password"] and config["email_to"]):
        print(f"[nexusai-health] ALERTA (SMTP no configurado): {subject} — {body}", file=sys.stderr)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["smtp_user"]
    msg["To"] = config["email_to"]
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(
            config["smtp_host"], config["smtp_port"], timeout=config["timeout_sec"]
        ) as smtp:
            smtp.login(config["smtp_user"], config["smtp_password"])
            smtp.send_message(msg)
    except Exception as exc:
        print(f"[nexusai-health] Falló el envío del email: {exc}", file=sys.stderr)


def main() -> int:
    config = _env_config()
    ok = check_health(config["health_url"], config["timeout_sec"])
    state = read_state(config["state_file"])

    if ok:
        if state["alerted"]:
            send_email(
                config,
                "✅ NexusAI: /health se recuperó",
                f"/health volvió a responder OK ({config['health_url']}).",
            )
        write_state(config["state_file"], {"consecutive_failures": 0, "alerted": False})
        return 0

    consecutive_failures = state["consecutive_failures"] + 1
    alerted = state["alerted"]

    if should_alert(consecutive_failures, config["failure_threshold"], alerted):
        send_email(
            config,
            "🔴 NexusAI: /health no responde",
            f"/health no responde hace {consecutive_failures} chequeo(s) consecutivos "
            f"({config['health_url']}).",
        )
        alerted = True

    write_state(
        config["state_file"],
        {"consecutive_failures": consecutive_failures, "alerted": alerted},
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
