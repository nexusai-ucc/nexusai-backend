"""
EmbeddingProvider — abstracción para vectorización de texto con retry automático.

Soporta cualquier proveedor compatible con el SDK de OpenAI cambiando
`EMBEDDING_BASE_URL`:

  - Gemini (default MVP):  https://generativelanguage.googleapis.com/v1beta/openai/
                           model: models/text-embedding-004 (768 dims)
  - OpenAI (prod):         https://api.openai.com/v1
                           model: text-embedding-3-small (1536 dims)
                           o text-embedding-3-large (3072 dims)

Retry: 3 intentos con backoff 1s → 2s para errores transitorios.

⚠️  IMPORTANTE: si cambia el modelo, las DIMENSIONES típicamente cambian
y hay que re-indexar TODA la base vectorial. Ver ADR-002 para el plan
de migración. Por eso `EMBEDDING_DIMENSIONS` es config explícita en .env
y se valida contra lo que devuelve el provider en el primer embed.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Awaitable, Callable, Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings
from app.shared.retry import async_retry
from app.shared.token_budget import estimate_tokens
from app.shared.usage_ledger import (
    UsageRecord,
    embedding_tokens,
    provider_from_base_url,
    record_usage,
    status_for_exception,
)


class EmbeddingProvider:
    """
    Wrapper async sobre el SDK de OpenAI para embeddings.

    Todos los métodos devuelven `list[float]` para un vector y
    `list[list[float]]` para N vectores.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model: str = model or settings.embedding_model
        self.dimensions: int = dimensions or settings.embedding_dimensions
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key or settings.embedding_api_key,
            base_url=base_url or settings.embedding_base_url,
            timeout=60.0,
            max_retries=0,  # Retries manejados por async_retry.
        )
        self.provider_name: str = (
            provider_from_base_url(base_url or settings.embedding_base_url) or ""
        )

    async def _create_recorded(
        self, call: Callable[[], Awaitable[Any]], texts: list[str]
    ) -> Any:
        """Llama al endpoint de embeddings y registra el consumo (COST-01).

        La API compatible de Gemini no informa tokens en embeddings: en ese
        caso se estiman con tiktoken y la fila queda marcada como estimada."""
        started = time.perf_counter()
        try:
            response = await async_retry(call)
        except Exception as exc:
            await record_usage(
                UsageRecord(
                    kind="embedding",
                    status=status_for_exception(exc),
                    provider=self.provider_name,
                    model=self.model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            )
            raise
        tokens = embedding_tokens(getattr(response, "usage", None))
        estimated = tokens == 0
        if estimated:
            tokens = sum(estimate_tokens(t) for t in texts)
        await record_usage(
            UsageRecord(
                kind="embedding",
                provider=self.provider_name,
                model=self.model,
                embedding_tokens=tokens,
                estimated=estimated,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        )
        return response

    async def embed(self, text: str) -> list[float]:
        """
        Vectoriza un solo texto con retry automático en errores transitorios.

        Para indexación masiva (ej. todos los chunks de un PDF), usar
        `embed_many()` que es N veces más eficiente (1 sola request).

        Args:
            text: el texto a embedear. No debe estar vacío.

        Returns:
            Vector como list[float] con `self.dimensions` elementos.

        Raises:
            ValueError: si el texto está vacío.
            RuntimeError: si la dimensión del vector no coincide con EMBEDDING_DIMENSIONS.
            openai.*: errores no-retryables o tras agotar los reintentos.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        response = await self._create_recorded(
            lambda: self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            ),
            [text],
        )
        vector = response.data[0].embedding

        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"EMBEDDING_DIMENSIONS={self.dimensions} no coincide con la "
                f"dimensión real del modelo {self.model!r} (devolvió {len(vector)}). "
                "Verificá el .env."
            )
        return vector

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Vectoriza N textos en una sola request (mucho más eficiente).

        Args:
            texts: lista no vacía de strings, ninguno vacío.

        Returns:
            Lista de N vectores, en el MISMO orden que el input.

        Raises:
            ValueError: si la lista está vacía o algún texto es vacío.
            RuntimeError: si algún vector tiene dimensión incorrecta.
            openai.*: errores no-retryables o tras agotar los reintentos.
        """
        if not texts:
            raise ValueError("Cannot embed empty list")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("All texts must be non-empty")

        response = await self._create_recorded(
            lambda: self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            ),
            texts,
        )

        # Re-ordenamos explícitamente por index (el SDK lo garantiza, pero por las dudas).
        sorted_data = sorted(response.data, key=lambda d: d.index)
        vectors = [d.embedding for d in sorted_data]

        for i, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise RuntimeError(
                    f"Vector #{i} tiene {len(vec)} dimensiones, esperaba "
                    f"{self.dimensions}. Modelo: {self.model!r}."
                )
        return vectors


# ============================================================
# FastAPI Dependency
# ============================================================


@lru_cache(maxsize=1)
def _cached_provider() -> EmbeddingProvider:
    return EmbeddingProvider()


def get_embedding_provider() -> EmbeddingProvider:
    """FastAPI Dependency. Inyecta el EmbeddingProvider en endpoints y pipelines."""
    return _cached_provider()
