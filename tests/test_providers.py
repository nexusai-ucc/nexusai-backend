"""
Tests de los providers (LLM y Embeddings).

Mockeamos el cliente AsyncOpenAI para no hacer llamadas reales en CI.
Si querés correr contra Gemini real, remové los mocks y exportá las env vars.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider


def _rate_limit_error() -> openai.RateLimitError:
    """Construye un RateLimitError (429) sin necesidad de una response HTTP real."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return openai.RateLimitError("quota exhausted", response=response, body=None)


def _server_error() -> openai.InternalServerError:
    """Construye un InternalServerError (503) sin necesidad de una response HTTP real."""
    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    return openai.InternalServerError("service unavailable", response=response, body=None)


# ============================================================
# LLMProvider — chat_completion
# ============================================================

@pytest.mark.asyncio
async def test_chat_completion_returns_message_content(fake_openai_chat_response):
    """chat_completion devuelve CompletionResult con .text y token counts."""
    provider = LLMProvider()
    fake_openai_chat_response.usage.prompt_tokens = 10
    fake_openai_chat_response.usage.completion_tokens = 5
    fake_openai_chat_response.usage.total_tokens = 15
    provider.client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    result = await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}]
    )
    assert result.text == "respuesta mockeada"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15


@pytest.mark.asyncio
async def test_chat_completion_empty_content_returns_empty_string():
    """Si el LLM devuelve content=None, .text es '' (no None)."""
    provider = LLMProvider()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    response.choices = [choice]
    response.usage = None  # sin usage → tokens en 0
    provider.client.chat.completions.create = AsyncMock(return_value=response)

    result = await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}]
    )
    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


# ============================================================
# LLMProvider — chat_stream
# ============================================================

@pytest.mark.asyncio
async def test_chat_stream_yields_deltas():
    """chat_stream debería ceder solo los deltas de texto, no chunks crudos."""
    provider = LLMProvider()

    async def fake_stream():
        for word in ["Hola", " ", "mundo"]:
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta.content = word
            chunk.choices = [choice]
            yield chunk

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    deltas = []
    async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
        deltas.append(d)

    assert deltas == ["Hola", " ", "mundo"]


@pytest.mark.asyncio
async def test_chat_stream_skips_empty_chunks():
    """Chunks sin choices o sin content no deberían yieldearse."""
    provider = LLMProvider()

    async def fake_stream():
        # Chunk válido
        chunk1 = MagicMock()
        choice1 = MagicMock()
        choice1.delta.content = "ok"
        chunk1.choices = [choice1]
        yield chunk1

        # Chunk sin choices (algunos providers lo mandan al final)
        chunk2 = MagicMock()
        chunk2.choices = []
        yield chunk2

        # Chunk con delta vacío
        chunk3 = MagicMock()
        choice3 = MagicMock()
        choice3.delta.content = None
        chunk3.choices = [choice3]
        yield chunk3

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    deltas = []
    async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
        deltas.append(d)

    assert deltas == ["ok"]


# ============================================================
# LLMProvider — fallback automático entre proveedores (INFRA-01 / #307)
# ============================================================

@pytest.mark.asyncio
async def test_chat_completion_falls_back_to_secondary_on_quota_error(
    fake_openai_chat_response, caplog
):
    """Si el primario agota sus reintentos por cuota, la request se completa
    igual usando el secundario, sin que el error llegue al caller."""
    provider = LLMProvider()
    assert provider.fallback_client is not None  # configurado en conftest.py

    provider.client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with caplog.at_level("WARNING", logger="app.providers.llm"):
            result = await provider.chat_completion(
                messages=[{"role": "user", "content": "hola"}]
            )

    assert result.text == "respuesta mockeada"
    # El primario se llamó 3 veces (async_retry) antes de rendirse.
    assert provider.client.chat.completions.create.call_count == 3
    # El secundario se llamó una sola vez, con su propio modelo configurado.
    provider.fallback_client.chat.completions.create.assert_called_once()
    assert provider.fallback_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
    assert "fallback" in caplog.text.lower()


@pytest.mark.asyncio
async def test_chat_completion_falls_back_on_server_error_503():
    """InternalServerError (503) también dispara el fallback, no solo 429."""
    provider = LLMProvider()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "desde el secundario"
    fake_response.usage = None

    provider.client.chat.completions.create = AsyncMock(side_effect=_server_error())
    provider.fallback_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "desde el secundario"


@pytest.mark.asyncio
async def test_chat_completion_propagates_when_no_fallback_configured():
    """Sin proveedor secundario configurado, el error de cuota propaga tal
    cual (comportamiento anterior a INFRA-01, sin regresión)."""
    provider = LLMProvider()
    provider.fallback_client = None  # simula deploy sin LLM_FALLBACK_* configurado

    provider.client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(openai.RateLimitError):
            await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])


@pytest.mark.asyncio
async def test_chat_stream_falls_back_to_secondary_on_quota_error(caplog):
    """chat_stream también cae al secundario si la apertura del stream
    primario falla por cuota (antes de yieldear ningún token)."""
    provider = LLMProvider()

    async def fake_fallback_stream():
        for word in ["Hola", " ", "secundario"]:
            chunk = MagicMock()
            choice = MagicMock()
            choice.delta.content = word
            chunk.choices = [choice]
            yield chunk

    provider.client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=fake_fallback_stream()
    )

    with caplog.at_level("WARNING", logger="app.providers.llm"):
        deltas = []
        async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
            deltas.append(d)

    assert deltas == ["Hola", " ", "secundario"]
    assert "fallback" in caplog.text.lower()


# ============================================================
# LLMProvider — cadena de modelos intermedios (INFRA-03 / #343)
# ============================================================

@pytest.mark.asyncio
async def test_chat_completion_uses_intermediate_model_before_secondary(
    fake_openai_chat_response, caplog
):
    """Con LLM_INTERMEDIATE_MODELS configurado, un 429 en el primario prueba
    el modelo intermedio ANTES de tocar el proveedor secundario."""
    provider = LLMProvider()
    provider.intermediate_models = ["gemini-2.5-flash-lite"]

    provider.client.chat.completions.create = AsyncMock(
        side_effect=[
            _rate_limit_error(), _rate_limit_error(), _rate_limit_error(),  # primario: 3 intentos
            fake_openai_chat_response,  # intermedio: responde OK al primer intento
        ]
    )
    provider.fallback_client.chat.completions.create = AsyncMock()

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with caplog.at_level("WARNING", logger="app.providers.llm"):
            result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "respuesta mockeada"
    assert provider.client.chat.completions.create.call_count == 4  # 3 primario + 1 intermedio
    provider.fallback_client.chat.completions.create.assert_not_called()
    assert "gemini-2.5-flash-lite" in caplog.text


@pytest.mark.asyncio
async def test_chat_completion_falls_through_full_chain_to_secondary(fake_openai_chat_response):
    """Si el primario Y todos los intermedios fallan, recién ahí se usa el
    proveedor secundario — la cadena completa se recorre en orden."""
    provider = LLMProvider()
    provider.intermediate_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]

    # 3 intentos primario + 3 intentos c/u de los 2 intermedios = 9 llamadas, todas 429.
    provider.client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "respuesta mockeada"
    assert provider.client.chat.completions.create.call_count == 9
    provider.fallback_client.chat.completions.create.assert_called_once()
    assert provider.fallback_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_chat_completion_cascades_between_gemini_models_without_secondary_configured():
    """Los modelos intermedios de Gemini funcionan solos, sin necesidad de
    tener un proveedor secundario (Groq) configurado."""
    provider = LLMProvider()
    provider.intermediate_models = ["gemini-2.5-flash-lite"]
    provider.fallback_client = None  # sin secundario configurado

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "desde el modelo intermedio"
    response.usage = None

    provider.client.chat.completions.create = AsyncMock(
        side_effect=[_rate_limit_error(), _rate_limit_error(), _rate_limit_error(), response]
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "desde el modelo intermedio"


@pytest.mark.asyncio
async def test_chat_completion_without_intermediate_models_behaves_like_infra01():
    """Sin LLM_INTERMEDIATE_MODELS (default), la cadena tiene un solo salto —
    comportamiento idéntico a antes de INFRA-03, sin regresión."""
    provider = LLMProvider()
    assert provider.intermediate_models == []
    assert provider._fallback_chain() == [
        (provider.client, provider.model),
        (provider.fallback_client, provider.fallback_model),
    ]


@pytest.mark.asyncio
async def test_chat_stream_cascades_through_intermediate_model():
    """chat_stream también recorre modelos intermedios antes del secundario."""
    provider = LLMProvider()
    provider.intermediate_models = ["gemini-2.5-flash-lite"]

    async def fake_intermediate_stream():
        chunk = MagicMock()
        choice = MagicMock()
        choice.delta.content = "desde intermedio"
        chunk.choices = [choice]
        yield chunk

    provider.client.chat.completions.create = AsyncMock(
        side_effect=[_rate_limit_error(), fake_intermediate_stream()]
    )
    provider.fallback_client.chat.completions.create = AsyncMock()

    deltas = []
    async for d in provider.chat_stream(messages=[{"role": "user", "content": "x"}]):
        deltas.append(d)

    assert deltas == ["desde intermedio"]
    provider.fallback_client.chat.completions.create.assert_not_called()


# ============================================================
# EmbeddingProvider
# ============================================================

@pytest.mark.asyncio
async def test_embed_returns_vector_with_correct_dimensions(
    fake_openai_embedding_response,
):
    provider = EmbeddingProvider()
    provider.client.embeddings.create = AsyncMock(
        return_value=fake_openai_embedding_response
    )

    vector = await provider.embed("texto cualquiera")
    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_embed_rejects_empty_text():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="empty"):
        await provider.embed("")
    with pytest.raises(ValueError, match="empty"):
        await provider.embed("   ")  # solo whitespace también es "vacío"


@pytest.mark.asyncio
async def test_embed_validates_dimensions_mismatch():
    """Si el modelo devuelve N dimensiones distintas a las configuradas, error."""
    provider = EmbeddingProvider()
    response = MagicMock()
    data = MagicMock()
    data.index = 0
    data.embedding = [0.1] * 1024  # esperamos 768, devuelve 1024
    response.data = [data]
    provider.client.embeddings.create = AsyncMock(return_value=response)

    with pytest.raises(RuntimeError, match="no coincide"):
        await provider.embed("texto")


@pytest.mark.asyncio
async def test_embed_many_preserves_order():
    """Si el SDK devuelve los embeddings desordenados, los re-ordenamos por índice."""
    provider = EmbeddingProvider()
    response = MagicMock()
    # Devolvemos en orden 2, 0, 1 — el provider tiene que re-ordenarlos.
    d2 = MagicMock(); d2.index = 2; d2.embedding = [0.3] * 768
    d0 = MagicMock(); d0.index = 0; d0.embedding = [0.1] * 768
    d1 = MagicMock(); d1.index = 1; d1.embedding = [0.2] * 768
    response.data = [d2, d0, d1]
    provider.client.embeddings.create = AsyncMock(return_value=response)

    vectors = await provider.embed_many(["a", "b", "c"])
    assert vectors[0][0] == pytest.approx(0.1)
    assert vectors[1][0] == pytest.approx(0.2)
    assert vectors[2][0] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_embed_many_rejects_empty_list():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="empty list"):
        await provider.embed_many([])


@pytest.mark.asyncio
async def test_embed_many_rejects_empty_strings():
    provider = EmbeddingProvider()
    with pytest.raises(ValueError, match="non-empty"):
        await provider.embed_many(["ok", ""])


# ============================================================
# LLMProvider — control del thinking (PERF-01)
# ============================================================

def _not_found_error() -> openai.NotFoundError:
    """404: el modelo fue retirado por el proveedor (pasa seguido en Gemini)."""
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return openai.NotFoundError("model not found", response=response, body=None)


def _bad_request_error() -> openai.BadRequestError:
    """400: el modelo rechaza un parámetro de la request."""
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    return openai.BadRequestError("invalid argument", response=response, body=None)


@pytest.mark.asyncio
async def test_chat_completion_sends_default_reasoning_effort(fake_openai_chat_response):
    """El default de LLM_REASONING_EFFORT viaja en extra_body en cada llamada.

    Es el cambio que baja el tiempo hasta el primer token: sin este parámetro,
    los flash 2.5+ de Gemini razonan en silencio antes de empezar a responder.
    """
    provider = LLMProvider(reasoning_effort="none")
    provider.client.chat.completions.create = AsyncMock(return_value=fake_openai_chat_response)

    await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}


@pytest.mark.asyncio
async def test_chat_completion_call_site_overrides_reasoning_effort(fake_openai_chat_response):
    """Un call-site puede pedir más razonamiento que el default global — es lo
    que hace la generación de quiz/examen con 'low'."""
    provider = LLMProvider(reasoning_effort="none")
    provider.client.chat.completions.create = AsyncMock(return_value=fake_openai_chat_response)

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}],
        reasoning_effort="low",
    )

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "low"}


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["", "default", "auto", "AUTO"])
async def test_chat_completion_omits_parameter_when_effort_unset(effort, fake_openai_chat_response):
    """Con "default"/"auto"/vacío no se manda el parámetro y decide el modelo,
    o sea el comportamiento previo a PERF-01. Es la vía de rollback por env var,
    sin tocar código."""
    provider = LLMProvider(reasoning_effort=effort)
    provider.client.chat.completions.create = AsyncMock(return_value=fake_openai_chat_response)

    await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert "extra_body" not in provider.client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_completion_does_not_clobber_caller_extra_body(fake_openai_chat_response):
    """Si el call-site ya arma su propio extra_body, se respeta lo que puso."""
    provider = LLMProvider(reasoning_effort="none")
    provider.client.chat.completions.create = AsyncMock(return_value=fake_openai_chat_response)

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hola"}],
        extra_body={"reasoning_effort": "high", "otro_param": 1},
    )

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "high", "otro_param": 1}


@pytest.mark.asyncio
async def test_chat_completion_stream_sends_reasoning_effort():
    """El streaming del chat también manda el parámetro — es justamente donde
    más se nota, porque el thinking retrasa el primer token visible."""
    provider = LLMProvider(reasoning_effort="none")

    async def fake_stream():
        chunk = MagicMock()
        choice = MagicMock()
        choice.delta.content = "ok"
        chunk.choices = [choice]
        chunk.usage = None
        yield chunk

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    async for _ in provider.chat_completion_stream(messages=[{"role": "user", "content": "x"}]):
        pass

    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"reasoning_effort": "none"}


# ============================================================
# LLMProvider — fallback ante modelo retirado / parámetro rechazado (PERF-01)
# ============================================================

@pytest.mark.asyncio
async def test_chat_completion_falls_back_when_model_was_retired(fake_openai_chat_response):
    """Un 404 por modelo retirado cede el paso al siguiente eslabón.

    Antes de PERF-01, NotFoundError no estaba en _FALLBACK_TRIGGERS: un modelo
    muerto en la cadena la cortaba entera y le propagaba el error al alumno en
    vez de seguir con el siguiente.
    """
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=_not_found_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "respuesta mockeada"
    # Un 404 es definitivo: no se reintenta el mismo modelo 3 veces, se pasa al
    # siguiente eslabón enseguida.
    assert provider.client.chat.completions.create.call_count == 1
    provider.fallback_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completion_falls_back_when_model_rejects_parameter(fake_openai_chat_response):
    """Un 400 (p. ej. un modelo que no acepta reasoning_effort) también cede el
    paso al siguiente eslabón en vez de fallar la request entera."""
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=_bad_request_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=fake_openai_chat_response
    )

    result = await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])

    assert result.text == "respuesta mockeada"
    assert provider.client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_chat_completion_propagates_404_when_chain_is_exhausted():
    """Con la cadena agotada el 404 propaga — el error no se traga."""
    provider = LLMProvider()
    provider.fallback_client = None
    provider.client.chat.completions.create = AsyncMock(side_effect=_not_found_error())

    with pytest.raises(openai.NotFoundError):
        await provider.chat_completion(messages=[{"role": "user", "content": "hola"}])
