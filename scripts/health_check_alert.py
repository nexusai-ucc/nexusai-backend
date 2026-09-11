#!/usr/bin/env python3
"""Watchdog externo de /health — NexusAI (ver docs/adr/012-alertas-monitoreo-minimo.md).

Por qué un script aparte y no algo dentro de FastAPI: si el proceso de
uvicorn muere, se cuelga, o el container ni siquiera levanta, nada que corra
DENTRO de ese proceso puede avisar de su propia caída. Necesita correr
afuera — acá, vía cron en la misma VM (o en la otra VM de Oracle,
apuntando a la URL pública, para no depender de que la VM caída sea la que
hace el chequeo).

Sin dependencias de terceros: solo stdlib (urllib), para no requerir instalar
nada además de Python 3 en una VM de 1 core / 1GB RAM.

Uso típico (crontab -e), cada minuto:

    * * * * * NEXUSAI_HEALTH_URL=https://api.<ip>.nip.io/health \
              NEXUSAI_ALERT_WEBHOOK_URL=https://hooks.slack.com/services/... \
              /usr/bin/python3 /opt/nexusai/services/api/scripts/health_check_alert.py \
              >> /var/log/nexusai-health.log 2>&1

Estado entre corridas: cada invocación de cron es un proceso nuevo sin
memoria compartida, así que las fallas consecutivas y si ya se avisó se
guardan en un archivo JSON chico (NEXUSAI_HEALTH_STATE_FILE).

Variables de entorno:
    NEXUSAI_HEALTH_URL              default: http://localhost:8001/health
    NEXUSAI_ALERT_WEBHOOK_URL       default: "" (sin webhook, solo loguea a stderr)
    NEXUSAI_HEALTH_FAILURE_THRESHOLD default: 3   (chequeos consecutivos fallidos antes de avisar)
    NEXUSAI_HEALTH_TIMEOUT_SEC      default: 5
    NEXUSAI_HEALTH_STATE_FILE       default: /tmp/nexusai_health_watch_state.json
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _env_config() -> dict:
    return {
        "health_url": os.environ.get("NEXUSAI_HEALTH_URL", "http://localhost:8001/health"),
        "webhook_url": os.environ.get("NEXUSAI_ALERT_WEBHOOK_URL", ""),
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
    el webhook durante toda la caída."""
    return consecutive_failures >= threshold and not already_alerted


def send_webhook(url: str, text: str, timeout: float) -> None:
    """Best-effort: si falla el POST (o no hay URL configurada), se loguea a
    stderr en vez de propagar — el watchdog no debe romper por esto."""
    if not url:
        print(f"[nexusai-health] ALERTA (sin webhook configurado): {text}", file=sys.stderr)
        return

    body = json.dumps({"text": text, "content": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        print(f"[nexusai-health] Falló el envío del webhook: {exc}", file=sys.stderr)


def main() -> int:
    config = _env_config()
    ok = check_health(config["health_url"], config["timeout_sec"])
    state = read_state(config["state_file"])

    if ok:
        if state["alerted"]:
            send_webhook(
                config["webhook_url"],
                f"✅ NexusAI: /health volvió a responder OK ({config['health_url']}).",
                config["timeout_sec"],
            )
        write_state(config["state_file"], {"consecutive_failures": 0, "alerted": False})
        return 0

    consecutive_failures = state["consecutive_failures"] + 1
    alerted = state["alerted"]

    if should_alert(consecutive_failures, config["failure_threshold"], alerted):
        send_webhook(
            config["webhook_url"],
            f"🔴 NexusAI: /health no responde hace {consecutive_failures} chequeo(s) "
            f"consecutivos ({config['health_url']}).",
            config["timeout_sec"],
        )
        alerted = True

    write_state(
        config["state_file"],
        {"consecutive_failures": consecutive_failures, "alerted": alerted},
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
