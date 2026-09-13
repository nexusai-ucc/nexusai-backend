"""Alertas mínimas viables: notificación por email (Gmail SMTP) + umbral en
ventana fija.

Ver ADR-012 (docs/adr/012-alertas-monitoreo-minimo.md). Dos piezas:

  - `send_alert`: manda un email best-effort vía SMTP (pensado para Gmail
    con una App Password, pero cualquier SMTP con auth sirve cambiando
    `ALERT_SMTP_HOST`/`ALERT_SMTP_PORT`). El destinatario (`ALERT_EMAIL_TO`)
    ya tiene default; sin `ALERT_SMTP_USER` + `ALERT_SMTP_PASSWORD`
    (la cuenta remitente) configuradas, solo loguea — no rompe nada.

  - `record_event_and_maybe_alert`: cuenta eventos en una ventana fija de
    Redis (mismo patrón que app.shared.rate_limit) y dispara `send_alert`
    al cruzar el umbral — con un cooldown (`ALERT_COOLDOWN_SEC`, default
    900s) que persiste INDEPENDIENTE de la ventana, no solo dentro de ella
    (hallazgo de audit sobre la versión anterior: con dedupe solo
    por-ventana, una caída sostenida donde cada ventana nueva vuelve a
    cruzar el umbral generaba una alerta nueva cada `window_sec`, hasta
    ~60/hora con el umbral de 5xx). La reusan app.shared.error_monitoring
    (5xx) y app.chat.router (fallas/latencia del LLM).
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from email.message import EmailMessage

import redis.asyncio as redis_async

from app.shared.config import get_settings

logger = logging.getLogger("nexusai.alerts")

# El event loop solo mantiene una referencia DÉBIL a una task creada con
# asyncio.create_task() — si nada más la referencia, puede recolectarse a
# mitad de ejecución (documentado en la stdlib). Como record_event_and_maybe_alert
# retorna inmediatamente después de crear la task de send_alert (todo el
# punto de hacerlo fire-and-forget), hace falta este set para mantener una
# referencia fuerte hasta que termine.
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _send_email_sync(
    *,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    email_to: str,
    subject: str,
    body: str,
) -> None:
    """Parte bloqueante (smtplib no es async) — se corre en un thread aparte
    vía `asyncio.to_thread` para no trabar el event loop de FastAPI."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as smtp:
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


async def send_alert(title: str, message: str) -> None:
    """Manda un email de alerta. Best-effort: nunca propaga una excepción.

    Requiere `ALERT_SMTP_USER` (la cuenta de Gmail remitente) y
    `ALERT_SMTP_PASSWORD` (App Password de esa cuenta, no la contraseña
    normal). El destinatario (`ALERT_EMAIL_TO`) ya tiene default — solo
    falta la cuenta remitente para que esto ande. Sin esas dos, solo loguea.
    """
    settings = get_settings()

    if not (settings.alert_smtp_user and settings.alert_smtp_password):
        logger.warning(
            "ALERTA (falta ALERT_SMTP_USER/ALERT_SMTP_PASSWORD): %s: %s",
            title,
            message,
        )
        return

    try:
        await asyncio.to_thread(
            _send_email_sync,
            smtp_host=settings.alert_smtp_host,
            smtp_port=settings.alert_smtp_port,
            smtp_user=settings.alert_smtp_user,
            smtp_password=settings.alert_smtp_password,
            email_to=settings.alert_email_to,
            subject=title,
            body=message,
        )
    except Exception as exc:
        logger.warning("No se pudo enviar el email de alerta: %s", exc)


async def record_event_and_maybe_alert(
    redis: redis_async.Redis,
    *,
    key_prefix: str,
    window_sec: int,
    threshold: int,
    title: str,
    message: str,
) -> bool:
    """Incrementa el contador de `key_prefix` en la ventana actual y alerta
    si supera `threshold`.

    Ventana fija (fixed window): el bucket cambia cada `window_sec` segundos,
    igual que en `app.shared.rate_limit.check_rate_limit` — para alertas de
    "¿hay un problema ahora mismo?" el trade-off de picos 2x en el cruce de
    ventana es aceptable.

    Cooldown: a diferencia del contador (que sí es por-ventana), la
    supresión de alertas repetidas usa una key SIN bucket, con TTL fijo
    `settings.alert_cooldown_sec` (900s por default) — persiste a través de
    varias ventanas seguidas. Solo la llamada que logra setearla con
    `SET NX` dispara `send_alert`; mientras el cooldown no expiró, cualquier
    ventana nueva que vuelva a cruzar el umbral no re-alerta, aunque el
    contador de esa ventana sí siga incrementándose normalmente.

    El envío en sí (`send_alert`) se dispara con `asyncio.create_task` en
    vez de `await` inline: mandar el email no debe agregarle latencia al
    request/la llamada que cruzó el umbral, que ya viene degradada (un 5xx,
    o una falla/lentitud del LLM) — es best-effort y nunca propaga
    excepción, así que no hace falta esperarlo.

    Devuelve True si esta llamada disparó la alerta.
    """
    bucket = int(time.time()) // window_sec
    count_key = f"{key_prefix}:count:{bucket}"
    cooldown_key = f"{key_prefix}:cooldown"
    count_ttl = window_sec + 10

    try:
        pipe = redis.pipeline()
        pipe.incr(count_key)
        pipe.expire(count_key, count_ttl)
        results = await pipe.execute()
        count = int(results[0])

        if count < threshold:
            return False

        settings = get_settings()
        already_alerted = not await redis.set(
            cooldown_key, "1", nx=True, ex=settings.alert_cooldown_sec
        )
        if already_alerted:
            return False
    except Exception as exc:
        # Redis caído no puede tumbar el request/la llamada que lo disparó.
        logger.warning("record_event_and_maybe_alert falló (Redis no disponible?): %s", exc)
        return False

    _fire_and_forget(send_alert(title, message))
    return True
