"""
LLMProvider — abstracción para chat completions.

Soporta cualquier proveedor compatible con el SDK de OpenAI cambiando
`LLM_BASE_URL` en el .env:

  - Gemini (default MVP):     https://generativelanguage.googleapis.com/v1beta/openai/
  - OpenAI (prod):            https://api.openai.com/v1
  - Ollama local (dev):       http://localhost:11434/v1
  - Groq:                     https://api.groq.com/openai/v1
  - Anthropic vía proxy:      depende del proxy

Métodos públicos:
  - `chat_completion(messages, **kwargs)` → respuesta completa, una sola string.
  - `chat_stream(messages, **kwargs)`     → AsyncIterator de chunks (para SSE).

Ver ADR-003 (decisión multi-provider) y ADR-004 (Gemini MVP / OpenAI prod).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings


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
        """
        Si los args vienen None, se leen de env vars (Settings). Útil para tests:
        se puede instanciar con valores explícitos sin tocar el global.
        """
        settings = get_settings()
        self.model: str = model or settings.llm_model
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=base_url or settings.llm_base_url,
            # Timeout generoso porque las respuestas LLM pueden tardar.
            # Si el LLM cuelga > 120s, mejor abortar y devolver error al usuario.
            timeout=120.0,
            max_retries=2,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Devuelve la respuesta como string única (sin streaming).

        Útil para flows que NO son chat (ej. resumen, generación de quiz),
        o si el cliente no soporta SSE. Para el chat normal con UX fluida,
        usar `chat_stream()`.

        Args:
            messages: lista en formato ChatML
                [{"role": "system|user|assistant", "content": "..."}, ...]
            **kwargs: cualquier param adicional del SDK (temperature, max_tokens,
                top_p, response_format, etc.). Se pasa tal cual al cliente.

        Returns:
            El texto de la respuesta.

        Raises:
            openai.* — errores del SDK (rate limit, auth, etc.) suben sin atrapar.
                El caller decide cómo manejarlos.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming de chunks de texto (útil para SSE hacia el frontend).

        Cada yield devuelve solo el delta de texto (no el JSON crudo del chunk).
        Quien consume puede emitir SSE así:

            async for delta in llm.chat_stream(messages):
                yield f"data: {json.dumps({'delta': delta})}\\n\\n"

        Args:
            messages: igual que `chat_completion`.
            **kwargs: igual que `chat_completion`.

        Yields:
            Strings con incrementos de texto. Algunos chunks pueden ser "" —
            el caller debe ignorar los vacíos si arma SSE.
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            # Defensive: algunos providers (no OpenAI) mandan chunks sin choices.
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
    """Instancia cacheada — un solo client por proceso."""
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
            answer = await llm.chat_completion([
                {"role": "user", "content": payload.question}
            ])
            return {"answer": answer}
    """
    return _cached_provider()
