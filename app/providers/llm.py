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

Ver ADR-003 (decisión multi-provider) y ADR-004 (Gemini MVP / OpenAI prod).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings
from app.shared.retry import async_retry


@dataclass
class CompletionResult:
    """Resultado de una chat completion con texto y métricas de tokens."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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

        Raises:
            openai.* — errores no-retryables o tras agotar todos los intentos.
        """
        response = await async_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return CompletionResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
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
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


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
