"""
Integración contra el proveedor LLM REAL (TPC-07, #579).

Solo corre con NEXUSAI_REAL_LLM=1 y una key real en LLM_API_KEY /
EMBEDDING_API_KEY: en el CI de cada PR se saltea (no expone secrets ni gasta
cuota). La corre `.github/workflows/integration-external.yml` a diario, a
mano y cuando cambia app/providers/, con una key de Gemini free tier solo
para CI (secret GEMINI_API_KEY_CI).

Cada test anota su latencia en $INTEGRATION_REPORT (JSON lines) para el
artifact del workflow y para docs/calidad/INTEGRACION_EXTERNA.md.
"""

from __future__ import annotations

import json
import math
import os
import time

import pytest

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NEXUSAI_REAL_LLM") != "1",
        reason="integración real deshabilitada (NEXUSAI_REAL_LLM=1 para correrla)",
    ),
]

# SLA de primera respuesta (services/api/TESTING.md §SLAs): < 8 s.
SLA_FIRST_RESPONSE_S = 8.0


def _report(test: str, **data) -> None:
    path = os.environ.get("INTEGRATION_REPORT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"test": test, **data}, ensure_ascii=False) + "\n")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.mark.asyncio
async def test_real_chat_answers_within_the_sla():
    """Criterio: el modelo configurado responde texto no vacío, informa tokens
    y tarda menos que el SLA de primera respuesta."""
    # Arrange.
    provider = LLMProvider()
    provider.fallback_client = (
        None  # medir el proveedor configurado, sin red de seguridad
    )
    messages = [
        {"role": "system", "content": "Respondé en una sola oración, en español."},
        {"role": "user", "content": "¿Qué es una pila en estructuras de datos?"},
    ]

    # Act.
    start = time.perf_counter()
    result = await provider.chat_completion(messages=messages)
    latency = time.perf_counter() - start

    # Assert.
    _report(
        "chat",
        model=result.model,
        latency_s=round(latency, 2),
        total_tokens=result.total_tokens,
    )
    assert result.text.strip()
    assert result.total_tokens > 0
    assert result.fallback is False
    assert (
        latency < SLA_FIRST_RESPONSE_S
    ), f"{latency:.2f} s supera el SLA de {SLA_FIRST_RESPONSE_S} s"


@pytest.mark.asyncio
async def test_real_retired_model_falls_through_to_the_next_link():
    """Criterio: si el primer modelo de la cadena no existe (404 real del
    proveedor, como cuando Google retira un modelo), la respuesta sale del
    siguiente eslabón y queda marcada como fallback (escenario de PERF-01)."""
    # Arrange.
    provider = LLMProvider()
    configured = provider.model
    provider.model = "gemini-modelo-retirado-tpc07"
    provider.intermediate_models = [configured]
    provider.fallback_client = None

    # Act.
    start = time.perf_counter()
    result = await provider.chat_completion(
        messages=[{"role": "user", "content": "Decí solo: hola"}]
    )
    latency = time.perf_counter() - start

    # Assert.
    _report("fallback_404", model=result.model, latency_s=round(latency, 2))
    assert result.text.strip()
    assert result.model == configured
    assert result.fallback is True


@pytest.mark.asyncio
async def test_real_embeddings_have_the_contract_dimensions_and_meaning():
    """Criterio: el embedding real tiene las dimensiones que espera pgvector y
    dos frases sobre lo mismo quedan más cerca que dos frases distintas."""
    # Arrange.
    provider = EmbeddingProvider()
    texts = [
        "Una pila es una estructura de datos LIFO.",
        "En una pila, el último elemento en entrar es el primero en salir.",
        "La fotosíntesis transforma la luz solar en energía química.",
    ]

    # Act.
    start = time.perf_counter()
    vectors = await provider.embed_many(texts)
    latency = time.perf_counter() - start

    # Assert.
    _report(
        "embeddings",
        model=provider.model,
        latency_s=round(latency, 2),
        dimensions=len(vectors[0]),
    )
    assert all(len(v) == provider.dimensions for v in vectors)
    assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])
