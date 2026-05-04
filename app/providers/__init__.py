"""
Providers — abstracciones agnósticas de vendor para LLM y embeddings.

Decisión arquitectónica: ADR-003 (multi-provider LLM) y ADR-004 (Gemini MVP /
OpenAI prod). Resumen:

- Todos los proveedores relevantes (Gemini, OpenAI, Anthropic, Groq, Ollama)
  son compatibles con el SDK de OpenAI cambiando `base_url`.
- NexusAI nunca hardcodea un proveedor. Las clases `LLMProvider` y
  `EmbeddingProvider` leen config de env vars y proxyean al SDK.
- Cambio de proveedor = editar `.env` + reiniciar uvicorn.
"""

from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "get_llm_provider",
    "get_embedding_provider",
]
