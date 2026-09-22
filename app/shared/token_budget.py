"""Presupuesto de tokens por user_id usando ventanas fijas en Redis.

Complementa a rate_limit.py: ese limita CANTIDAD de requests. Este limita
CANTIDAD DE TOKENS consumidos — proxy directo del costo real del LLM. Un
usuario podría hacer pocas consultas pero muy largas (o con historial
extenso) y agotar presupuesto real sin llegar nunca al límite de requests;
este módulo cubre ese caso.

Reserva atómica + ajuste posterior (no "leer, después sumar"):

  - reserve_token_budget: ANTES de llamar al LLM, reserva atómicamente una
    ESTIMACIÓN del costo (vía tiktoken, ver estimate_tokens/
    estimate_tokens_for_messages) con un INCRBY — no un simple GET. Si esa
    reserva hace que el acumulado ya alcance el límite, la revierte
    (DECRBY) y corta con 429 sin cobrarle una request que no va a
    ejecutarse. Esto es lo que hace que requests CONCURRENTES del mismo
    usuario no puedan leer el mismo acumulado "viejo" y pasar todas juntas
    — cada una reserva su lugar de inmediato, igual que check_rate_limit
    hace INCR dentro del mismo chequeo (una versión anterior de este módulo
    solo hacía GET acá, dejando una ventana de carrera del tamaño de la
    latencia completa del LLM — ver PR #516, hallazgo de /audit).
  - finalize_token_usage: DESPUÉS del LLM, ajusta la diferencia entre lo
    reservado y el costo real (conocido recién en ese momento) con un
    INCRBY de la diferencia (puede ser negativa, si tiktoken sobreestimó).

Nota sobre precisión de la reserva: el estimador (tiktoken cl100k_base, ver
estimate_tokens) es una aproximación — no es el tokenizer real de
Gemini/GPT-4o-mini, mismo criterio que ya usa app/documents/chunker.py para
dimensionar chunks de RAG — y en los endpoints de chat se reserva solo con
la PREGUNTA del alumno (lo único conocido en el punto del flujo donde se
llama, antes del retrieval RAG), no con el prompt final completo (que
también lleva contexto RAG + historial, armado más tarde). La reserva por
lo tanto es conservadora — cubre una fracción del costo real — pero alcanza
para cerrar la ventana de carrera concurrente: ya no hay ningún punto en el
que N requests paralelas lean "0 usado" al mismo tiempo. finalize_token_usage
corrige la diferencia completa (incluido el contexto RAG) una vez conocida.

Mismo esquema de bucket de ventana fija que rate_limit.py — ver ese archivo
para el razonamiento sobre epoch UTC vs. medianoche local, que aplica igual
acá. La key además separa por rol (student/teacher, resuelto server-side en
el plugin PHP — ver ChatRequest.is_teacher): antes de esto, un usuario con
distinto rol en distintos cursos (ej. ayudante de cátedra: editingteacher en
un curso, alumno en otro) compartía un único contador global evaluado contra
el límite de CUALQUIERA de los dos roles según en qué curso preguntara,
bloqueándolo de forma inconsistente o dándole de más. Separar por rol tiene
su propio costo: ese mismo usuario dual-rol tiene, en los hechos, dos
presupuestos independientes en vez de uno — aceptable porque es un caso
borde (la mayoría de los usuarios son puramente alumno o puramente docente
en toda la plataforma) y muchísimo mejor que la inconsistencia anterior.
"""

from __future__ import annotations

import logging
import time

import redis.asyncio as redis_async
import tiktoken
from fastapi import HTTPException, status

logger = logging.getLogger("nexusai.token_budget")

_MESSAGES = {
    "hourly": (
        "Alcanzaste tu límite de {limit} tokens por hora. "
        "Esperá un momento y volvé a intentarlo."
    ),
    "daily": "Alcanzaste tu límite de {limit} tokens de hoy. Volvé a intentarlo mañana.",
}

# El alumno ve este texto tal cual (el frontend muestra el mensaje de nuestro
# propio limitador), así que sigue el idioma de la pregunta cuando es inglés
# — mismo criterio que rate_limit.py.
_MESSAGES_EN = {
    "hourly": (
        "You reached your limit of {limit} tokens for this hour. "
        "Wait a moment and try again."
    ),
    "daily": "You reached your limit of {limit} tokens for today. Try again tomorrow.",
}

# cl100k_base es una aproximación (no el tokenizer real de Gemini/GPT-4o-mini)
# pero es gratis, local y consistente entre requests — igual que en
# app/documents/chunker.py. Se cachea a nivel de módulo: es un objeto sin
# estado de I/O (no es como asyncio.Semaphore(), que se ata al event loop —
# tiktoken.Encoding es solo tablas de BPE en memoria, seguro de compartir).
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """Estimación best-effort de tokens de `text` vía tiktoken (cl100k_base)."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def estimate_tokens_for_messages(messages: list[dict]) -> int:
    """Suma estimate_tokens sobre el `content` de una lista de mensajes LLM."""
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def _bucket_key(user_id: int, is_teacher: bool, window_sec: int) -> str:
    tier = "teacher" if is_teacher else "student"
    bucket = int(time.time()) // window_sec
    return f"nexusai:tokenbudget:{window_sec}:{tier}:{user_id}:{bucket}"


def _raise_budget_exceeded(
    scope: str, limit: int, used: int, window_sec: int, language: str | None
) -> None:
    retry_after = window_sec - (int(time.time()) % window_sec)
    messages = _MESSAGES_EN if language == "en" else _MESSAGES
    message_template = messages.get(scope, messages["daily"])
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error": "token_budget_exceeded",
            "scope": scope,
            "message": message_template.format(limit=limit),
            "limit": limit,
            "used": used,
            "window_sec": window_sec,
        },
        headers={"Retry-After": str(retry_after)},
    )


async def reserve_token_budget(
    user_id: int,
    is_teacher: bool,
    redis: redis_async.Redis,
    limit: int,
    window_sec: int,
    scope: str,
    estimated_tokens: int,
    language: str | None = None,
) -> int:
    """Reserva `estimated_tokens` atómicamente; lanza 429 si eso ya alcanza el límite.

    Devuelve la cantidad EFECTIVAMENTE reservada (para pasarla después a
    finalize_token_usage y calcular el ajuste con el costo real):
      - `estimated_tokens` si la reserva se aplicó (camino normal).
      - 0 si Redis falló y no se aplicó ningún incremento real — fail-open,
        mismo criterio que el resto del módulo (no tumbar la request por la
        caída de una dependencia auxiliar), pero finalize_token_usage tiene
        que sumar el costo completo después, no solo el delta.

    Args:
        user_id: ID del usuario de Moodle.
        is_teacher: Resuelto server-side en el plugin PHP (has_capability),
            nunca del JS del navegador — ver ChatRequest.is_teacher.
        redis: Cliente Redis async.
        limit: Techo de tokens de la ventana (por rol — ver Settings).
        window_sec: Tamaño de la ventana en segundos.
        scope: "hourly" | "daily" — identificador lógico para el mensaje.
        estimated_tokens: Estimación conocida ANTES de llamar al LLM (ver
            estimate_tokens/estimate_tokens_for_messages).
        language: "en" para mensaje en inglés, cualquier otro valor (o None)
            para español.
    """
    if estimated_tokens < 0:
        estimated_tokens = 0
    key = _bucket_key(user_id, is_teacher, window_sec)
    try:
        pipe = redis.pipeline()
        pipe.incrby(key, estimated_tokens)
        pipe.expire(key, window_sec + 10)
        results = await pipe.execute()
    except Exception:
        # Fail-open a propósito, mismo criterio que rate_limit.py: el
        # presupuesto de tokens es un guardrail de costo, no un control de
        # seguridad crítico.
        logger.error(
            "Token budget reserve falló (Redis no disponible), dejando pasar: scope=%s user_id=%s",
            scope,
            user_id,
        )
        return 0

    new_total = int(results[0])
    if new_total - estimated_tokens >= limit:
        # Ya estaba en (o sobre) el límite ANTES de esta reserva — revertir
        # (no cobrarle una request que no va a ejecutarse) y cortar.
        try:
            await redis.decrby(key, estimated_tokens)
        except Exception:
            logger.error(
                "No se pudo revertir la reserva de tokens tras 429 (Redis no disponible): "
                "user_id=%s scope=%s",
                user_id,
                scope,
            )
        _raise_budget_exceeded(
            scope, limit, new_total - estimated_tokens, window_sec, language
        )

    return estimated_tokens


async def finalize_token_usage(
    user_id: int,
    is_teacher: bool,
    redis: redis_async.Redis,
    window_sec: int,
    reserved_tokens: int,
    actual_tokens: int,
) -> None:
    """Ajusta la reserva de reserve_token_budget() al costo real ya conocido.

    delta = actual_tokens - reserved_tokens: positivo si el costo real fue
    mayor a lo reservado (agrega lo que faltaba — típicamente el contexto
    RAG + historial + completion, no reservados de antemano), negativo si
    tiktoken sobreestimó. Con reserved_tokens=0 (nada reservado — moderación,
    que no pasa por reserve_token_budget porque no es streaming y no tiene
    ventana de cancelación) equivale a sumar actual_tokens directo.

    Fail-open igual que reserve_token_budget: si Redis falla acá se pierde
    el ajuste de ESTA request pero no tiene sentido romper la respuesta ya
    generada — el LLM ya fue llamado (y facturado).
    """
    delta = actual_tokens - reserved_tokens
    if delta == 0:
        return
    key = _bucket_key(user_id, is_teacher, window_sec)
    try:
        pipe = redis.pipeline()
        pipe.incrby(key, delta)
        pipe.expire(key, window_sec + 10)
        await pipe.execute()
    except Exception:
        logger.error(
            "No se pudo ajustar el consumo real de tokens (Redis no disponible): "
            "user_id=%s delta=%s window_sec=%s",
            user_id,
            delta,
            window_sec,
        )
