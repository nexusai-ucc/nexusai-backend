"""
Tests del registro de consumo por llamada a proveedor (COST-01, issue #519).

La escritura en la base (`usage_ledger._persist`) se reemplaza por un
AsyncMock que junta las filas, así los tests no necesitan Postgres.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_lib
import json
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.analytics.logger import log_moderation_block
from app.auth.hmac import verify_hmac
from app.infrastructure.redis_client import get_redis
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider, StreamToken, StreamUsage
from app.providers.transcription import TranscriptionProvider
from app.shared import usage_ledger
from app.shared.config import get_settings
from app.shared.usage_ledger import (
    Price,
    UsageContext,
    UsageRecord,
    compute_cost,
    context_from_request,
    feature_from_route_path,
    get_usage_context,
    provider_from_base_url,
    record_usage,
    resolve_role,
    set_usage_context,
    usage_scope,
)


@pytest.fixture
def ledger(monkeypatch):
    """Prende el registro y captura las filas en vez de escribirlas."""
    monkeypatch.setattr(get_settings(), "usage_ledger_enabled", True)
    persist = AsyncMock()
    monkeypatch.setattr(usage_ledger, "_persist", persist)
    set_usage_context(UsageContext())
    yield persist
    set_usage_context(UsageContext())


def _rows(persist: AsyncMock) -> list[dict]:
    return [call.args[0] for call in persist.await_args_list]


def _usage(prompt: int, completion: int, cached: int = 0) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    usage.prompt_tokens_details.cached_tokens = cached
    return usage


def _chat_response(
    prompt: int = 100, completion: int = 20, cached: int = 0
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "respuesta"
    response.usage = _usage(prompt, completion, cached)
    return response


def _rate_limit_error() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return openai.RateLimitError("quota exhausted", response=response, body=None)


# ============================================================
# Funciones puras
# ============================================================


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/v1/chat/stream", "chat.stream"),
        ("/api/v1/quiz/generate-exam", "quiz.generate-exam"),
        ("/api/v1/documents/{document_id}/preview", "documents.preview"),
        ("/api/v1/courses/{course_id}/stats", "courses.stats"),
        ("/api/v1/search", "search"),
        ("/", "unknown"),
    ],
)
def test_feature_from_route_path(path, expected):
    assert feature_from_route_path(path) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "google"),
        ("https://api.openai.com/v1", "openai"),
        ("https://api.groq.com/openai/v1", "groq"),
        ("http://localhost:11434/v1", "local"),
        ("https://otro.example/v1", "otro.example"),
        (None, None),
    ],
)
def test_provider_from_base_url(url, expected):
    assert provider_from_base_url(url) == expected


def test_resolve_role_prefers_header_then_is_teacher():
    assert resolve_role("teacher", False) == "teacher"
    assert resolve_role("SYSTEM", None) == "system"
    assert resolve_role("admin", True) == "teacher"  # valor inválido: se ignora
    assert resolve_role(None, False) == "student"
    assert resolve_role(None, None) == "unknown"


def test_context_from_request_reads_json_body():
    body = json.dumps({"course_id": 7, "user_id": 42, "is_teacher": True}).encode()
    ctx = context_from_request(
        route_path="/api/v1/chat/stream",
        body=body,
        query={},
        role_header=None,
        request_id="req-1",
        api_key="clave",
    )
    assert ctx.feature == "chat.stream"
    assert (ctx.course_id, ctx.user_id, ctx.role) == (7, 42, "teacher")
    assert ctx.request_id == "req-1"
    assert ctx.client_id == hashlib.sha256(b"clave").hexdigest()[:12]


def test_context_from_request_reads_query_for_get_and_tolerates_bad_body():
    ctx = context_from_request(
        route_path="/api/v1/admin/analytics",
        body=b"no es json",
        query={"course_id": "9"},
        role_header="student",
        request_id=None,
        api_key="clave",
    )
    assert (ctx.feature, ctx.course_id, ctx.user_id, ctx.role) == (
        "admin.analytics",
        9,
        None,
        "student",
    )


_PRICE = Price(
    input_per_mtok=Decimal("0.15"),
    output_per_mtok=Decimal("0.60"),
    cached_input_per_mtok=Decimal("0.075"),
    embedding_per_mtok=Decimal("0.02"),
    audio_per_minute=Decimal("0.006"),
)


def test_compute_cost_llm_with_cached_tokens():
    record = UsageRecord(
        kind="llm",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        cached_prompt_tokens=400_000,
    )
    # 600k * 0.15 + 400k * 0.075 + 500k * 0.60 = 0.09 + 0.03 + 0.30
    assert compute_cost(_PRICE, record) == Decimal("0.42000000")


def test_compute_cost_embeddings_and_audio():
    assert compute_cost(
        _PRICE, UsageRecord(kind="embedding", embedding_tokens=2_000_000)
    ) == Decimal("0.04000000")
    assert compute_cost(
        _PRICE, UsageRecord(kind="transcription", audio_seconds=30)
    ) == Decimal("0.00300000")


def test_compute_cost_without_price_and_without_consumption():
    assert compute_cost(None, UsageRecord(kind="llm", prompt_tokens=10)) is None
    assert compute_cost(None, UsageRecord(kind="moderation", status="blocked")) == 0


# ============================================================
# record_usage
# ============================================================


async def test_record_usage_disabled_does_not_persist(monkeypatch):
    persist = AsyncMock()
    monkeypatch.setattr(usage_ledger, "_persist", persist)
    monkeypatch.setattr(get_settings(), "usage_ledger_enabled", False)
    await record_usage(UsageRecord(kind="llm", prompt_tokens=5))
    persist.assert_not_awaited()


async def test_record_usage_merges_request_context(ledger):
    set_usage_context(
        UsageContext(
            feature="quiz.generate",
            course_id=3,
            user_id=8,
            role="student",
            request_id="r",
            client_id="c",
        )
    )
    await record_usage(UsageRecord(kind="llm", model="m", prompt_tokens=5))
    row = _rows(ledger)[0]
    assert row["feature"] == "quiz.generate"
    assert (row["course_id"], row["user_id"], row["role"]) == (3, 8, "student")
    assert (row["request_id"], row["client_id"]) == ("r", "c")
    assert row["prompt_tokens"] == 5


async def test_record_usage_never_raises(ledger):
    ledger.side_effect = RuntimeError("base caída")
    await record_usage(UsageRecord(kind="llm", prompt_tokens=5))  # no propaga


# ============================================================
# Captura en los proveedores
# ============================================================


async def test_chat_completion_records_model_and_tokens(ledger):
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_chat_response(prompt=120, completion=30, cached=20)
    )

    result = await provider.chat_completion([{"role": "user", "content": "hola"}])

    # El modelo y el proveedor salen de la configuración del entorno (en CI y en
    # local no son los mismos), así que se comparan contra los del propio objeto.
    assert (result.model, result.fallback) == (provider.model, False)
    [row] = _rows(ledger)
    assert row["kind"] == "llm" and row["status"] == "ok"
    assert (row["provider"], row["model"], row["fallback"]) == (
        provider.provider_name,
        provider.model,
        False,
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (120, 30)
    assert row["cached_prompt_tokens"] == 20
    assert row["latency_ms"] is not None


async def test_chat_completion_records_failed_link_and_fallback(ledger):
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
    provider.fallback_client.chat.completions.create = AsyncMock(
        return_value=_chat_response()
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert (result.model, result.provider, result.fallback) == (
        provider.fallback_model,
        provider.fallback_provider_name,
        True,
    )
    failed, ok = _rows(ledger)
    assert (failed["status"], failed["model"], failed["prompt_tokens"]) == (
        "quota",
        provider.model,
        0,
    )
    assert (ok["status"], ok["model"], ok["fallback"]) == (
        "ok",
        provider.fallback_model,
        True,
    )


async def test_chat_completion_records_error_that_does_not_fall_back(ledger):
    provider = LLMProvider()
    provider.client.chat.completions.create = AsyncMock(side_effect=ValueError("x"))

    with pytest.raises(ValueError):
        await provider.chat_completion([{"role": "user", "content": "hola"}])

    [row] = _rows(ledger)
    assert (row["status"], row["model"]) == ("error", provider.model)


async def test_chat_completion_stream_records_final_usage(ledger):
    provider = LLMProvider()

    async def fake_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hola"
        chunk.usage = None
        yield chunk
        last = MagicMock()
        last.choices = []
        last.usage = _usage(50, 10, 5)
        yield last

    provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

    chunks = [
        c
        async for c in provider.chat_completion_stream(
            [{"role": "user", "content": "x"}]
        )
    ]

    assert isinstance(chunks[0], StreamToken)
    usage = chunks[-1]
    assert isinstance(usage, StreamUsage)
    assert (usage.model, usage.fallback, usage.cached_prompt_tokens) == (
        provider.model,
        False,
        5,
    )
    [row] = _rows(ledger)
    assert (
        row["prompt_tokens"],
        row["completion_tokens"],
        row["cached_prompt_tokens"],
    ) == (
        50,
        10,
        5,
    )
    assert row["status"] == "ok"


async def test_embed_records_tokens_and_errors(ledger):
    provider = EmbeddingProvider()
    response = MagicMock()
    data = MagicMock()
    data.embedding = [0.1] * 768
    response.data = [data]
    response.usage.prompt_tokens = 12
    provider.client.embeddings.create = AsyncMock(return_value=response)

    await provider.embed("texto")

    provider.client.embeddings.create = AsyncMock(side_effect=ValueError("x"))
    with pytest.raises(ValueError):
        await provider.embed("texto")

    ok, failed = _rows(ledger)
    assert (ok["kind"], ok["embedding_tokens"], ok["model"]) == (
        "embedding",
        12,
        provider.model,
    )
    assert failed["status"] == "error"


async def test_embed_estimates_tokens_when_provider_omits_them(ledger):
    """La API compatible de Gemini no informa tokens en embeddings."""
    provider = EmbeddingProvider()
    response = MagicMock()
    data = MagicMock()
    data.embedding = [0.1] * 768
    response.data = [data]
    response.usage = None
    provider.client.embeddings.create = AsyncMock(return_value=response)

    await provider.embed("una pregunta sobre procesos")

    [row] = _rows(ledger)
    assert row["estimated"] is True
    assert row["embedding_tokens"] > 0


async def test_usage_scope_marks_internal_steps(ledger):
    set_usage_context(UsageContext(feature="chat.stream"))
    with usage_scope("moderation"):
        await record_usage(UsageRecord(kind="llm", prompt_tokens=1))
    await record_usage(UsageRecord(kind="llm", prompt_tokens=1))

    inner, outer = _rows(ledger)
    assert inner["feature"] == "chat.stream:moderation"
    assert outer["feature"] == "chat.stream"


async def test_transcription_records_audio_seconds(ledger):
    provider = TranscriptionProvider(api_key="k")
    provider.client = MagicMock()
    provider.client.audio.transcriptions.create = AsyncMock(
        return_value=MagicMock(text=" hola ", duration=4.5)
    )

    assert await provider.transcribe(b"audio", "q.webm", "audio/webm") == "hola"

    _, kwargs = provider.client.audio.transcriptions.create.call_args
    assert kwargs["response_format"] == "verbose_json"
    [row] = _rows(ledger)
    assert (row["kind"], row["provider"], row["audio_seconds"]) == (
        "transcription",
        "groq",
        4.5,
    )


async def test_moderation_block_is_recorded(ledger):
    log_moderation_block(
        endpoint="chat.stream",
        course_id=4,
        user_id=9,
        source="llm",
        categories=["insulto"],
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    [row] = _rows(ledger)
    assert (row["kind"], row["status"], row["feature"]) == (
        "moderation",
        "blocked",
        "chat.stream",
    )
    assert (row["course_id"], row["user_id"]) == (4, 9)


# ============================================================
# Contexto desde verify_hmac
# ============================================================


def test_verify_hmac_binds_usage_context(fake_redis):
    app = FastAPI()

    @app.post("/api/v1/quiz/generate")
    async def endpoint(_body: bytes = Depends(verify_hmac)):
        ctx = get_usage_context()
        return {
            "feature": ctx.feature,
            "course_id": ctx.course_id,
            "user_id": ctx.user_id,
            "role": ctx.role,
        }

    app.dependency_overrides[get_redis] = lambda: fake_redis
    settings = get_settings()
    body = json.dumps({"course_id": 5, "user_id": 11}).encode()
    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())
    signature = hmac_lib.new(
        settings.nexusai_shared_secret.encode(),
        (timestamp + nonce).encode() + body,
        hashlib.sha256,
    ).hexdigest()

    response = TestClient(app).post(
        "/api/v1/quiz/generate",
        content=body,
        headers={
            "Authorization": f"Bearer {settings.nexusai_api_key}",
            "X-Timestamp": timestamp,
            "X-Nonce": nonce,
            "X-Signature": signature,
            "X-NexusAI-Role": "teacher",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "feature": "quiz.generate",
        "course_id": 5,
        "user_id": 11,
        "role": "teacher",
    }
