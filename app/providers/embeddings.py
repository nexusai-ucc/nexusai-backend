"""
EmbeddingProvider — abstracción para vectorización de texto.

Soporta cualquier proveedor compatible con el SDK de OpenAI cambiando
`EMBEDDING_BASE_URL`:

  - Gemini (default MVP):  https://generativelanguage.googleapis.com/v1beta/openai/
                           model: models/text-embedding-004 (768 dims)
  - OpenAI (prod):         https://api.openai.com/v1
                           model: text-embedding-3-small (1536 dims)
                           o text-embedding-3-large (3072 dims)

⚠️  IMPORTANTE: si cambia el modelo, las DIMENSIONES típicamente cambian
y hay que **re-indexar TODA la base vectorial**. Ver ADR-002 para el plan
de migración. Por eso `EMBEDDING_DIMENSIONS` es config explícita en .env
y se valida contra lo que devuelve el provider en el primer embed.

Métodos públicos:
  - `embed(text)`            → np.ndarray | list[float], 1 vector
  - `embed_many(texts)`      → list[list[float]], N vectores en un solo request
  - `dimensions`             → int, cuántas dimensiones tiene este modelo
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings


class EmbeddingProvider:
    """
    Wrapper async sobre el SDK de OpenAI para embeddings.

    Convención: TODOS los métodos devuelven `list[float]` para el caso de
    1 vector y `list[list[float]]` para N vectores. No devolvemos `np.ndarray`
    para evitar pegar numpy en la firma pública (lo usan los consumidores
    si quieren — ej. ChromaDB / pgvector aceptan ambas).
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
            timeout=60.0,        # Embeddings son rápidos vs chat — 60s alcanza.
            max_retries=2,
        )

    async def embed(self, text: str) -> list[float]:
        """
        Vectoriza un solo texto.

        Para indexación masiva (ej. todos los chunks de un PDF), usar
        `embed_many()` que es N veces más eficiente porque hace 1 sola request
        en lugar de N.

        Args:
            text: el texto a embedear. No debe estar vacío. Si es muy largo
                (>8K tokens en text-embedding-3-small), el SDK truncará o
                fallará dependiendo del provider.

        Returns:
            Vector como list[float] con `self.dimensions` elementos.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Pasamos `dimensions` explícito para activar Matryoshka en
        # gemini-embedding-001 (default es 3072). Con 768 mantenemos
        # compatibilidad con la columna Vector(768) y el índice HNSW
        # de pgvector (que no soporta >2000 dims).
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        vector = response.data[0].embedding

        # Validación defensiva: si la dimensión real no coincide con la
        # configurada, hay un mismatch entre .env y el modelo real. Mejor
        # explotar acá con un error claro que escribir vectores rotos en pgvector.
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
            Lista de N vectores, en el MISMO orden que el input. Esto es
            crítico para mapear cada vector al chunk correspondiente.

        Raises:
            ValueError: si la lista está vacía o algún texto es vacío.
        """
        if not texts:
            raise ValueError("Cannot embed empty list")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("All texts must be non-empty")

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )

        # El SDK devuelve los embeddings ordenados por índice. La doc lo
        # garantiza, pero por las dudas los re-ordenamos explícitamente.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        vectors = [d.embedding for d in sorted_data]

        # Sanity check de dimensiones (mismo motivo que embed()).
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
    """
    FastAPI Dependency.

    Uso:
        from fastapi import Depends
        from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

        @router.post("/index")
        async def index_chunks(
            chunks: list[str],
            embedder: EmbeddingProvider = Depends(get_embedding_provider),
        ):
            vectors = await embedder.embed_many(chunks)
            # ... guardar en pgvector ...
    """
    return _cached_provider()
