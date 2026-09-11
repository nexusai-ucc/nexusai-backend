"""Alertas mínimas viables: notificación por webhook + umbral en ventana fija.

Ver ADR-012 (docs/adr/012-alertas-monitoreo-minimo.md). Dos piezas:

  - `send_alert`: dispara una notificación best-effort a un webhook
    configurable (Slack/Discord). Sin `ALERT_WEBHOOK_URL` seteada, solo
    loguea — no rompe nada.

  - `record_event_and_maybe_alert`: cuenta eventos en una ventana fija de
    Redis (mismo patrón que app.shared.rate_limit) y dispara `send_alert`
    como máximo UNA vez por ventana al cruzar el umbral, para no floodear
    el webhook mientras el problema persiste. La reusan
    app.shared.error_monitoring (5xx) y app.chat.router (fallas/latencia
    del LLM).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
import redis.asyncio as redis_async

from app.shared.config import get_settings

logger = logging.getLogger("nexusai.alerts")


async def send_alert(title: str, message: str) -> None:
    """Postea una alerta al webhook configurado. Best-effort: nunca propaga.

    El body incluye `text` (Slack) y `content` (Discord) — ambos formatos de
    webhook entrante leen el campo que entienden e ignoran el resto, así que
    un solo POST sirve para cualquiera de los dos sin tener que detectar el
    proveedor por la URL.
    """
    settings = get_settings()
    full_message = f"{title}: {message}"

    if not settings.alert_webhook_url:
        logger.warning("ALERTA (sin ALERT_WEBHOOK_URL configurada): %s", full_message)
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                settings.alert_webhook_url,
                json={"text": full_message, "content": full_message},
            )
    except Exception as exc:
        logger.warning("No se pudo enviar la alerta al webhook: %s", exc)


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

    Dedupe: además del contador, se setea una bandera `SET NX` con el mismo
    TTL que la ventana. Solo la llamada que logra setearla (la primera en
    cruzar el umbral dentro de esa ventana) dispara `send_alert` — las
    siguientes llamadas de la misma ventana ya la ven seteada y no vuelven a
    avisar, aunque el contador siga subiendo.

    Devuelve True si esta llamada disparó la alerta.
    """
    bucket = int(time.time()) // window_sec
    count_key = f"{key_prefix}:count:{bucket}"
    alerted_key = f"{key_prefix}:alerted:{bucket}"
    ttl = window_sec + 10

    try:
        pipe = redis.pipeline()
        pipe.incr(count_key)
        pipe.expire(count_key, ttl)
        results = await pipe.execute()
        count = int(results[0])

        if count < threshold:
            return False

        already_alerted = not await redis.set(alerted_key, "1", nx=True, ex=ttl)
        if already_alerted:
            return False
    except Exception as exc:
        # Redis caído no puede tumbar el request/la llamada que lo disparó.
        logger.warning("record_event_and_maybe_alert falló (Redis no disponible?): %s", exc)
        return False

    await send_alert(title, message)
    return True
