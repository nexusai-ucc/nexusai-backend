"""Alertas de error 5xx y de salud del proveedor LLM — ver ADR-012.

`record_5xx_and_maybe_alert` se llama desde app.shared.middleware por cada
response con status >= 500 y cubre cualquier 5xx del backend, venga de
donde venga.

`record_llm_failure_and_maybe_alert` / `record_llm_slow_and_maybe_alert` se
llaman desde app.chat.router (el call-site principal del LLM, el que afecta
directamente la experiencia del alumno). Son señales más específicas que el
5xx genérico:

  - Una falla del LLM en el endpoint streaming NO llega como 5xx (la
    response SSE ya arrancó en 200), así que el contador de 5xx no la ve.
  - Una respuesta lenta del LLM no es un error — sigue devolviendo 200 —
    pero degrada la experiencia igual, así que necesita su propio umbral de
    latencia en vez de contarse como fallo.
  - Una RateLimitError que agota toda la cadena de fallback es, específicamente,
    cuota agotada — se separa del resto de las fallas del LLM (timeouts,
    modelo caído, etc.) porque la acción correctiva es otra.
"""

from __future__ import annotations

import openai
import redis.asyncio as redis_async

from app.shared.alerting import record_event_and_maybe_alert
from app.shared.config import get_settings


async def record_5xx_and_maybe_alert(
    redis: redis_async.Redis,
    *,
    status_code: int,
    path: str,
) -> bool:
    """Cuenta esta respuesta 5xx y alerta si se superó el umbral de la ventana.

    No hace nada si `status_code` no es 5xx — se puede llamar sin chequear
    antes desde el caller.
    """
    if status_code < 500:
        return False

    settings = get_settings()
    return await record_event_and_maybe_alert(
        redis,
        key_prefix=f"nexusai:alerts:5xx:{settings.error_rate_window_sec}",
        window_sec=settings.error_rate_window_sec,
        threshold=settings.error_rate_threshold,
        title="🔴 NexusAI: tasa de error 5xx elevada",
        message=(
            f"{settings.error_rate_threshold}+ respuestas 5xx en los últimos "
            f"{settings.error_rate_window_sec}s. Último endpoint: {path} ({status_code})."
        ),
    )


async def record_llm_failure_and_maybe_alert(
    redis: redis_async.Redis,
    *,
    endpoint: str,
    error: BaseException,
) -> bool:
    """Cuenta esta falla del LLM (cadena de fallback agotada o error no
    reintentable) y alerta si se superó el umbral de la ventana.

    Un `RateLimitError` que llega hasta acá significa que TODA la cadena
    (primario + intermedios + secundario, ver providers/llm.py) devolvió
    cuota agotada — se trata aparte, con su propio umbral (más bajo) y un
    mensaje que apunta a "revisar cuota/billing del proveedor" en vez del
    genérico "el LLM está fallando".
    """
    settings = get_settings()

    if isinstance(error, openai.RateLimitError):
        return await record_event_and_maybe_alert(
            redis,
            key_prefix=f"nexusai:alerts:llm_quota:{settings.llm_quota_window_sec}",
            window_sec=settings.llm_quota_window_sec,
            threshold=settings.llm_quota_threshold,
            title="🔴 NexusAI: cuota del proveedor LLM agotada",
            message=(
                "La cadena completa de proveedores LLM (primario + fallback, si está "
                f"configurado) devolvió cuota agotada (429). Endpoint: {endpoint}. "
                "Revisar cuota/billing del proveedor."
            ),
        )

    return await record_event_and_maybe_alert(
        redis,
        key_prefix=f"nexusai:alerts:llm_failure:{settings.llm_failure_window_sec}",
        window_sec=settings.llm_failure_window_sec,
        threshold=settings.llm_failure_threshold,
        title="🔴 NexusAI: el proveedor LLM está fallando",
        message=(
            f"{settings.llm_failure_threshold}+ fallas del LLM en los últimos "
            f"{settings.llm_failure_window_sec}s. Último endpoint: {endpoint} "
            f"({type(error).__name__}: {error})."
        ),
    )


async def record_llm_slow_and_maybe_alert(
    redis: redis_async.Redis,
    *,
    endpoint: str,
    latency_ms: float,
) -> bool:
    """Cuenta esta respuesta lenta del LLM y alerta si se superó el umbral.

    Solo cuenta si `latency_ms` supera `settings.llm_slow_threshold_ms` — el
    caller puede llamar siempre, sin chequear antes.
    """
    settings = get_settings()
    if latency_ms < settings.llm_slow_threshold_ms:
        return False

    return await record_event_and_maybe_alert(
        redis,
        key_prefix=f"nexusai:alerts:llm_slow:{settings.llm_slow_window_sec}",
        window_sec=settings.llm_slow_window_sec,
        threshold=settings.llm_slow_threshold_count,
        title="🟡 NexusAI: el LLM está respondiendo lento",
        message=(
            f"{settings.llm_slow_threshold_count}+ respuestas del LLM por encima de "
            f"{settings.llm_slow_threshold_ms}ms en los últimos {settings.llm_slow_window_sec}s. "
            f"Último endpoint: {endpoint} ({latency_ms:.0f}ms)."
        ),
    )
