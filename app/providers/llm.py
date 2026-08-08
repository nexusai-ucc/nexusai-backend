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

Cadena de modelos intermedios (INFRA-03 / issue #343): Google AI Studio da
cuota gratuita POR MODELO, no por cuenta — agotar `gemini-2.5-flash` no
agota `gemini-2.5-flash-lite` ni `gemini-2.0-flash`. `LLM_INTERMEDIATE_MODELS`
(opcional, coma-separado) prueba esos modelos extra usando el MISMO client
del primario (misma API key, mismo base_url — no son otro proveedor, solo
otro modelo) antes de recién ahí pasar al proveedor secundario configurado.
Cadena resultante: primario → intermedios (en orden) → secundario. Vacío =
comportamiento idéntico a antes de INFRA-03.

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

        # Modelos intermedios opcionales (INFRA-03) — mismo client que el
        # primario, ver _fallback_chain().
        self.intermediate_models: list[str] = [
            m.strip()
            for m in (settings.llm_intermediate_models or "").split(",")
            if m.strip()
        ]

    def _fallback_chain(self) -> list[tuple[AsyncOpenAI, str]]:
        """Cadena completa a intentar en orden (INFRA-03):

          primario → modelos intermedios (mismo client, ver __init__) →
          proveedor secundario final (otro client, si está configurado).

        Sin LLM_INTERMEDIATE_MODELS ni fallback configurado, devuelve una
        lista de un solo elemento — mismo comportamiento que antes de
        INFRA-01/INFRA-03 (un solo intento, sin fallback).
        """
        chain = [(self.client, self.model)]
        chain.extend((self.client, m) for m in self.intermediate_models)
        if self.fallback_client:
            chain.append((self.fallback_client, self.fallback_model))
        return chain

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

        Si un eslabón de la cadena (ver _fallback_chain) agota sus reintentos
        por cuota/servidor caído (ver _FALLBACK_TRIGGERS), pasa automáticamente
        al siguiente eslabón antes de propagar.

        Raises:
            openai.* — errores no-retryables, o el del último eslabón de la
            cadena si todos fallan.
        """
        response = await self._run_completion_chain(self._fallback_chain(), messages, **kwargs)
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

    @staticmethod
    async def _run_completion_chain(
        chain: list[tuple[AsyncOpenAI, str]],
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Recorre la cadena en orden (ver _fallback_chain). Cada eslabón ya
        tiene sus propios 3 reintentos vía _create_completion/async_retry —
        recién si esos 3 fallan por _FALLBACK_TRIGGERS pasa al siguiente
        eslabón. Con un solo eslabón (caso default sin fallback configurado),
        el comportamiento es idéntico al de antes de INFRA-01/INFRA-03."""
        for i, (client, model) in enumerate(chain):
            try:
                return await LLMProvider._create_completion(client, model, messages, **kwargs)
            except _FALLBACK_TRIGGERS as exc:
                if i == len(chain) - 1:
                    raise
                next_model = chain[i + 1][1]
                logger.warning(
                    "LLM fallback activado: %s agotado (%s: %s). Pasando a %s.",
                    model, type(exc).__name__, exc, next_model,
                )

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Abre un stream recorriendo la cadena de fallback en orden (ver
        _fallback_chain), pasando al siguiente eslabón si la apertura falla
        por cuota/servidor caído.

        Nota: el fallback solo cubre la apertura del stream. Si la conexión se
        corta a mitad de la iteración (después de ya haber yieldeado texto),
        no se reintenta — mismo comportamiento que antes de INFRA-01, porque
        ya se envió output parcial al caller y no se puede "deshacer".
        """
        chain = self._fallback_chain()
        for i, (client, model) in enumerate(chain):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )
            except _FALLBACK_TRIGGERS as exc:
                if i == len(chain) - 1:
                    raise
                next_model = chain[i + 1][1]
                logger.warning(
                    "LLM fallback activado (stream): %s agotado (%s: %s). Pasando a %s.",
                    model, type(exc).__name__, exc, next_model,
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
