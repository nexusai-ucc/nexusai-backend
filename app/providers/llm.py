"""
LLMProvider — abstracción para chat completions con retry automático.

Soporta cualquier proveedor compatible con el SDK de OpenAI cambiando
`LLM_BASE_URL` en el .env:

  - Gemini (default MVP):     https://generativelanguage.googleapis.com/v1beta/openai/
  - OpenAI (prod):            https://api.openai.com/v1
  - Ollama local (dev):       http://localhost:11434/v1
  - Groq:                     https://api.groq.com/openai/v1

Retry: 3 intentos con backoff 1s → 2s para errores transitorios
(RateLimitError, Timeout, ConnectionError, InternalServerError).
Los errores definitivos (AuthenticationError, BadRequestError) propagan de inmediato.

Fallback automático (INFRA-01 / issue #307): si el proveedor primario agota
sus 3 reintentos por cuota (429 RateLimitError) o caída del servidor (503
InternalServerError), y hay un proveedor secundario configurado vía
LLM_FALLBACK_API_KEY/LLM_FALLBACK_BASE_URL/LLM_FALLBACK_MODEL, se reintenta
automáticamente contra ese secundario antes de propagar el error al caller.
Sin esas 3 env vars, el fallback queda deshabilitado (comportamiento idéntico
al anterior).

Ver ADR-003 (decisión multi-provider) y ADR-004 (Gemini MVP / OpenAI prod).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Optional, Union

import openai
from openai import AsyncOpenAI

from app.shared.config import get_settings
from app.shared.retry import async_retry

logger = logging.getLogger(__name__)

# Errores que disparan el fallback al proveedor secundario: cuota agotada
# (429 RESOURCE_EXHAUSTED en Gemini) o servidor caído (503). Deliberadamente
# más angosto que _RETRYABLE_OPENAI de retry.py — timeouts/conexión no
# disparan fallback porque probablemente afecten a ambos proveedores por igual.
_FALLBACK_TRIGGERS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass
class CompletionResult:
    """Resultado de una chat completion con texto y métricas de tokens."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class StreamToken:
    """Un chunk de texto del LLM en modo streaming."""

    text: str


@dataclass(frozen=True)
class StreamUsage:
    """Conteo final de tokens del stream (último chunk cuando include_usage=True)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


StreamChunk = Union[StreamToken, StreamUsage]


class LLMProvider:
    """
    Wrapper async sobre el SDK de OpenAI configurado para el proveedor activo.

    No es un singleton estricto — `get_llm_provider()` (la dependency de FastAPI)
    cachea una instancia con `lru_cache`. Para tests, instanciar directo es ok.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model: str = model or settings.llm_model
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=base_url or settings.llm_base_url,
            timeout=120.0,
            max_retries=0,  # Retries manejados por async_retry, no por el SDK.
        )

        # Proveedor secundario opcional — ver _FALLBACK_TRIGGERS arriba.
        self.fallback_model: Optional[str] = settings.llm_fallback_model
        self.fallback_client: Optional[AsyncOpenAI] = None
        if (
            settings.llm_fallback_api_key
            and settings.llm_fallback_base_url
            and settings.llm_fallback_model
        ):
            self.fallback_client = AsyncOpenAI(
                api_key=settings.llm_fallback_api_key,
                base_url=settings.llm_fallback_base_url,
                timeout=120.0,
                max_retries=0,
            )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Devuelve la respuesta completa del LLM junto con el conteo de tokens.

        Reintenta automáticamente hasta 3 veces en errores transitorios
        (rate limit, timeout, servidor caído). Propaga inmediatamente en errores
        definitivos (auth inválida, bad request, etc.).

        Args:
            messages: lista en formato ChatML
                [{"role": "system|user|assistant", "content": "..."}, ...]
            **kwargs: parámetros adicionales del SDK (temperature, max_tokens, etc.)

        Returns:
            CompletionResult con el texto de respuesta y los token counts del LLM.

        Si el proveedor primario agota sus reintentos por cuota/servidor caído
        (ver _FALLBACK_TRIGGERS) y hay un proveedor secundario configurado,
        reintenta automáticamente contra ese secundario antes de propagar.

        Raises:
            openai.* — errores no-retryables, o del secundario si también falla,
            o del primario si no hay secundario configurado.
        """
        try:
            response = await self._create_completion(self.client, self.model, messages, **kwargs)
        except _FALLBACK_TRIGGERS as exc:
            if not self.fallback_client:
                raise
            logger.warning(
                "LLM fallback activado: proveedor primario agotado (%s: %s). "
                "Reintentando con proveedor secundario (%s).",
                type(exc).__name__, exc, self.fallback_model,
            )
            response = await self._create_completion(
                self.fallback_client, self.fallback_model, messages, **kwargs
            )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return CompletionResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    @staticmethod
    async def _create_completion(
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Chat completion no-streaming con retry, contra el client/model dados."""
        return await async_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        )

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Abre un stream contra el proveedor primario, con fallback al secundario
        si la apertura del stream falla por cuota/servidor caído.

        Nota: el fallback solo cubre la apertura del stream. Si la conexión se
        corta a mitad de la iteración (después de ya haber yieldeado texto),
        no se reintenta — mismo comportamiento que antes de INFRA-01, porque
        ya se envió output parcial al caller y no se puede "deshacer".
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
        except _FALLBACK_TRIGGERS as exc:
            if not self.fallback_client:
                raise
            logger.warning(
                "LLM fallback activado (stream): proveedor primario agotado "
                "(%s: %s). Reintentando con proveedor secundario (%s).",
                type(exc).__name__, exc, self.fallback_model,
            )
            return await self.fallback_client.chat.completions.create(
                model=self.fallback_model,
                messages=messages,
                stream=True,
                **kwargs,
            )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming de chunks de texto (útil para SSE hacia el frontend).

        El streaming no se reintenta automáticamente: si la conexión se corta
        a mitad del stream, el error propaga al caller. Para el chat MVP
        (no-streaming), usar `chat_completion`.

        Yields:
            Strings con incrementos de texto. Algunos chunks pueden ser "" —
            el caller debe ignorarlos al armar SSE.
        """
        stream = await self._create_stream(messages, **kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming con conteo de tokens al final.

        Yieldea StreamToken por cada chunk de texto. Cuando el provider lo soporta
        (Gemini compat OpenAI / OpenAI nativo), al final del stream yieldea un
        único StreamUsage con los token counts del prompt + completion. Útil para
        persistir métricas en la DB después del streaming.
        """
        stream = await self._create_stream(
            messages, stream_options={"include_usage": True}, **kwargs
        )

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield StreamToken(text=delta.content)
            # En el último chunk con include_usage=True, viene el usage poblado.
            if getattr(chunk, "usage", None):
                u = chunk.usage
                yield StreamUsage(
                    prompt_tokens=u.prompt_tokens or 0,
                    completion_tokens=u.completion_tokens or 0,
                    total_tokens=u.total_tokens or 0,
                )


# ============================================================
# FastAPI Dependency
# ============================================================

@lru_cache(maxsize=1)
def _cached_provider() -> LLMProvider:
    return LLMProvider()


def get_llm_provider() -> LLMProvider:
    """
    FastAPI Dependency. Inyecta el LLMProvider en endpoints y servicios.

    Uso:
        from fastapi import Depends
        from app.providers.llm import LLMProvider, get_llm_provider

        @router.post("/chat")
        async def chat(
            payload: ChatRequest,
            llm: LLMProvider = Depends(get_llm_provider),
        ):
            result = await llm.chat_completion([
                {"role": "user", "content": payload.question}
            ])
            return {"answer": result.text, "tokens": result.total_tokens}
    """
    return _cached_provider()
