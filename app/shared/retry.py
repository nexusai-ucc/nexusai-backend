"""Async retry helper with exponential backoff for transient API failures."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Tuple, Type, TypeVar

import openai

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errores de OpenAI/Gemini que valen la pena reintentar (transitorios).
# Auth, bad request y not-found son definitivos — no reintentamos.
_RETRYABLE_OPENAI: Tuple[Type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


async def async_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: Tuple[Type[BaseException], ...] = _RETRYABLE_OPENAI,
) -> T:
    """Llama a coro_factory() hasta max_attempts veces con backoff exponencial.

    Args:
        coro_factory: Callable sin args que retorna una coroutine (ej. lambda).
        max_attempts: Total de intentos antes de propagar la excepción.
        base_delay: Segundos de espera tras el 1er fallo. Se duplica en cada reintento.
        retryable: Tipos de excepción que disparan el reintento. El resto propaga de inmediato.

    Delays entre intentos: base_delay * 2^(intento-1)
      intento 1 falla → espera 1s
      intento 2 falla → espera 2s
      intento 3 falla → propaga

    Returns:
        El valor de retorno de coro_factory() al tener éxito.

    Raises:
        La última excepción retryable si todos los intentos fallan.
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except retryable as exc:
            last_exc = exc
            if attempt == max_attempts:
                logger.warning(
                    "Todos los %d intentos fallaron: %s: %s",
                    max_attempts, type(exc).__name__, exc,
                )
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Intento %d/%d falló (%s: %s). Reintentando en %.1fs...",
                attempt, max_attempts, type(exc).__name__, exc, delay,
            )
            await asyncio.sleep(delay)
        # Excepciones no-retryable propagán inmediatamente (sin except aquí).

    assert last_exc is not None
    raise last_exc
