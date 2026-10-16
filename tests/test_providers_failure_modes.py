"""
Contrato de fallas del proveedor LLM (TPC-07, #579).

Completa los dobles de `test_providers.py` (429/503/404/400 → fallback) con
los modos de falla que faltaban, y deja escrito qué hace NexusAI en cada uno:

| Falla del proveedor          | Reintentos | ¿Pasa al siguiente eslabón? |
|------------------------------|------------|-----------------------------|
| Timeout / error de conexión  | 3          | NO (decisión de diseño)     |
| 401 credencial inválida      | 0          | NO                          |
| Embeddings con 429 o timeout | 3          | NO hay fallback de embeddings |

La fila del timeout fija la decisión de `_FALLBACK_TRIGGERS` en
`app/providers/llm.py` ("timeouts y errores de conexión NO disparan
fallback"): si alguien la cambia, este test lo hace explícito. Está
registrada como observación de riesgo en docs/calidad/REGISTRO_DEFECTOS.md.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider

_REQUEST = httpx.Request("POST", "https://test.example/v1/chat/completions")


def _timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def _auth_error() -> openai.AuthenticationError:
    response = MagicMock()
    response.status_code = 401
    response.headers = {}
    return openai.AuthenticationError("invalid api key", response=response, body=None)


def _rate_limit_error() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return openai.RateLimitError("quota exhausted", response=response, body=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [_timeout, _connection_error], ids=["timeout", "conexion"]
)
async def test_timeout_and_connection_errors_are_retried_but_do_not_fall_back(failure):
    """Criterio: se reintenta 3 veces contra el primario y el error llega al
    caller sin probar el secundario (se asume que afecta a toda la cadena)."""
    # Arrange.
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=failure())
    provider.fallback_client.chat.completions.create = AsyncMock()

    # Act.
    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises((openai.APITimeoutError, openai.APIConnectionError)):
            await provider.chat_completion(
                messages=[{"role": "user", "content": "hola"}]
            )

    # Assert.
    assert provider.client.chat.completions.create.call_count == 3
    provider.fallback_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_credentials_fail_at_once_without_fallback():
    """Criterio: un 401 es definitivo: ni reintento ni fallback (una key mal
    cargada tiene que verse enseguida, no esconderse detrás del secundario)."""
    # Arrange.
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=_auth_error())
    provider.fallback_client.chat.completions.create = AsyncMock()

    # Act.
    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(openai.AuthenticationError):
            await provider.chat_completion(
                messages=[{"role": "user", "content": "hola"}]
            )

    # Assert.
    assert provider.client.chat.completions.create.call_count == 1
    provider.fallback_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [_rate_limit_error, _timeout], ids=["429", "timeout"]
)
async def test_embeddings_have_no_fallback(failure):
    """Criterio: los embeddings reintentan 3 veces y después propagan: no hay
    proveedor alternativo (punto único de falla documentado en el plan de
    pruebas; sin embeddings no se indexa material ni se responde con RAG)."""
    # Arrange.
    provider = EmbeddingProvider()
    provider.client.embeddings.create = AsyncMock(side_effect=failure())

    # Act.
    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises((openai.RateLimitError, openai.APITimeoutError)):
            await provider.embed("¿Qué es una pila?")

    # Assert.
    assert provider.client.embeddings.create.call_count == 3


@pytest.mark.asyncio
async def test_transient_error_recovers_on_the_second_attempt(
    fake_openai_chat_response,
):
    """Criterio: un timeout aislado no llega al alumno: el segundo intento responde."""
    # Arrange.
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(
        side_effect=[_timeout(), fake_openai_chat_response]
    )

    # Act.
    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await provider.chat_completion(
            messages=[{"role": "user", "content": "hola"}]
        )

    # Assert.
    assert result.text == "respuesta mockeada"
    assert provider.client.chat.completions.create.call_count == 2
    sleep.assert_awaited_once_with(1.0)
