"""
Tests de la API key propia del usuario (BYOK) como último eslabón de la cadena.

Ver app/providers/user_llm.py. Todo con el SDK mockeado: ninguna llamada real.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from fastapi import HTTPException, Request
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.providers.llm import LLMProvider, StreamLink, get_llm_provider
from app.providers.user_llm import (
    PROVIDERS,
    REASON_MODEL,
    REASON_QUOTA,
    REASON_REJECTED,
    UserLLMConfig,
    UserLLMError,
    clear_clients,
    client_for,
    find_user_llm_error,
    parse_user_llm,
    user_llm_from_body,
)
from app.shared import usage_ledger
from app.shared.config import get_settings
from app.shared.usage_ledger import UsageContext, UsageRecord, set_usage_context

_VALID = {
    "provider": "anthropic",
    "model": "claude-sonnet-5-5",
    "api_key": "sk-ant-test-1234",
}


@pytest.fixture(autouse=True)
def _fresh_clients():
    """Los clients del usuario se cachean por (proveedor, key): cada test
    arranca con clients nuevos para que los mocks no se filtren entre tests."""
    clear_clients()
    yield
    clear_clients()


def _status_error(cls, code: int, message: str = "x"):
    response = MagicMock()
    response.status_code = code
    response.headers = {}
    return cls(message, response=response, body=None)


def _response(text: str = "ok", total: int = 30) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock(
        prompt_tokens=total - 10,
        completion_tokens=10,
        total_tokens=total,
        prompt_tokens_details=None,
    )
    return response


def _with_user(**overrides) -> LLMProvider:
    config = parse_user_llm({**_VALID, **overrides})
    assert config is not None
    return LLMProvider().with_user_llm(config)


# ============================================================
# Validación de `user_llm`
# ============================================================


def test_parse_accepts_a_known_provider_and_hides_the_key_in_repr():
    config = parse_user_llm(_VALID)
    assert config == UserLLMConfig("anthropic", "claude-sonnet-5-5", "sk-ant-test-1234")
    assert config.base_url == PROVIDERS["anthropic"][1]
    assert "sk-ant" not in repr(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "http://169.254.169.254"},  # nunca una URL del pedido
        {"provider": "desconocido"},
        {"model": ""},
        {"model": "modelo con espacios"},
        {"model": "x" * 200},
        {"api_key": "corta"},
        {"api_key": "sk-con espacio-adentro"},
        {"api_key": 12345678},
    ],
)
def test_parse_ignores_invalid_values(overrides):
    assert parse_user_llm({**_VALID, **overrides}) is None


def test_parse_ignores_missing_or_non_object():
    assert parse_user_llm(None) is None
    assert parse_user_llm("sk-ant") is None


def test_user_llm_from_body_reads_the_signed_body():
    body = json.dumps({"course_id": 3, "user_llm": _VALID}).encode()
    assert user_llm_from_body(body).provider == "anthropic"
    assert user_llm_from_body(b"") is None
    assert user_llm_from_body(b"no json") is None
    assert user_llm_from_body(json.dumps({"course_id": 3}).encode()) is None


# ============================================================
# Caché de clients del usuario
# ============================================================


async def test_client_cache_reuses_per_key_and_closes_what_it_evicts(monkeypatch):
    import asyncio

    from app.providers import user_llm

    monkeypatch.setattr(user_llm, "_CLIENT_CACHE_SIZE", 2)
    monkeypatch.setattr(user_llm, "_CLOSE_DELAY_SEC", 0)
    first = client_for("openai", "sk-first-000000")
    assert client_for("openai", "sk-first-000000") is first
    first.close = AsyncMock()

    client_for("openai", "sk-second-00000")
    client_for("openai", "sk-third-000000")  # saca al primero
    await asyncio.sleep(0.01)

    first.close.assert_awaited_once()
    assert client_for("openai", "sk-first-000000") is not first
    assert all("sk-" not in key_hash for _, key_hash in user_llm._clients)


# ============================================================
# Cadena: el eslabón del usuario va último
# ============================================================


def test_user_link_is_the_last_link_and_the_shared_instance_is_untouched():
    base = LLMProvider()
    before = list(base._fallback_chain())
    provider = base.with_user_llm(parse_user_llm(_VALID))

    chain = provider._fallback_chain()
    assert chain[:-1] == before
    assert chain[-1] == (provider.user_client, "claude-sonnet-5-5")
    assert base._fallback_chain() == before  # la copia no tocó la instancia
    assert provider.institutional()._fallback_chain() == before
    assert provider.user_only()._fallback_chain() == [chain[-1]]


async def test_falls_back_to_the_user_key_when_the_institution_is_out_of_quota():
    provider = _with_user()
    provider.client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.RateLimitError, 429)
    )
    provider.fallback_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.InternalServerError, 503)
    )
    provider.user_client.chat.completions.create = AsyncMock(
        return_value=_response("de tu key")
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert result.text == "de tu key"
    assert (result.byok, result.provider, result.model) == (
        True,
        "anthropic",
        "claude-sonnet-5-5",
    )
    # reasoning_effort es de los modelos de la facultad: al usuario no se le manda.
    kwargs = provider.user_client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in (kwargs.get("extra_body") or {})


async def test_the_user_key_is_not_used_while_the_institution_answers():
    provider = _with_user()
    provider.client.chat.completions.create = AsyncMock(
        return_value=_response("facultad")
    )
    provider.user_client.chat.completions.create = AsyncMock()

    result = await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert (result.text, result.byok) == ("facultad", False)
    provider.user_client.chat.completions.create.assert_not_called()


async def test_a_timeout_of_the_institution_jumps_straight_to_the_user_key():
    """Un timeout no es trigger de fallback entre eslabones de la facultad,
    pero sí pasa a la key del usuario, que es otro proveedor y otra cuenta."""
    provider = _with_user()
    timeout = openai.APITimeoutError(request=MagicMock())
    provider.client.chat.completions.create = AsyncMock(side_effect=timeout)
    provider.fallback_client.chat.completions.create = AsyncMock()
    provider.user_client.chat.completions.create = AsyncMock(return_value=_response())

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        result = await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert result.byok is True
    provider.fallback_client.chat.completions.create.assert_not_called()
    # Con key del usuario, el timeout de la facultad no se reintenta: no daría
    # tiempo a llegar a su key antes de que el plugin corte.
    assert provider.client.chat.completions.create.call_count == 1


async def test_without_user_key_a_timeout_still_propagates():
    provider = LLMProvider()
    timeout = openai.APITimeoutError(request=MagicMock())
    provider.client.chat.completions.create = AsyncMock(side_effect=timeout)

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(openai.APITimeoutError):
            await provider.chat_completion([{"role": "user", "content": "hola"}])


@pytest.mark.parametrize(
    "error, reason",
    [
        (_status_error(openai.AuthenticationError, 401), REASON_REJECTED),
        (_status_error(openai.PermissionDeniedError, 403), REASON_REJECTED),
        (_status_error(openai.RateLimitError, 429), REASON_QUOTA),
        (_status_error(openai.NotFoundError, 404), REASON_MODEL),
    ],
)
async def test_a_failing_user_key_raises_user_llm_error(error, reason):
    provider = _with_user().user_only()
    provider.user_client.chat.completions.create = AsyncMock(side_effect=error)

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(UserLLMError) as caught:
            await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert (caught.value.reason, caught.value.provider) == (reason, "anthropic")
    assert "Anthropic" in caught.value.message("es")
    assert caught.value.detail("en")["error"] == "user_llm_failed"


async def test_user_key_out_of_credit_is_not_retried():
    """Un 429 de la cuenta del usuario no se arregla esperando: falla de una."""
    provider = _with_user().user_only()
    provider.user_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.RateLimitError, 429)
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()) as slept:
        with pytest.raises(UserLLMError):
            await provider.chat_completion([{"role": "user", "content": "hola"}])

    assert provider.user_client.chat.completions.create.call_count == 1
    slept.assert_not_awaited()


async def test_user_error_keeps_the_institution_failure_for_the_alerts():
    provider = _with_user()
    quota = _status_error(openai.RateLimitError, 429)
    provider.client.chat.completions.create = AsyncMock(side_effect=quota)
    provider.fallback_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.InternalServerError, 503)
    )
    provider.user_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.AuthenticationError, 401)
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(UserLLMError) as caught:
            await provider.chat_completion([{"role": "user", "content": "hola"}])

    # La cuota agotada de la facultad se prioriza, igual que sin BYOK (ADR-012).
    assert caught.value.institutional_error is quota


async def test_a_request_both_reject_as_invalid_is_not_blamed_on_the_user_key():
    provider = _with_user()
    institution_400 = _status_error(openai.BadRequestError, 400, "context too long")
    provider.client.chat.completions.create = AsyncMock(side_effect=institution_400)
    provider.fallback_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.BadRequestError, 400)
    )
    provider.user_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.BadRequestError, 400)
    )

    with patch("app.shared.retry.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(openai.BadRequestError):
            await provider.chat_completion([{"role": "user", "content": "hola"}])


async def test_chat_alerts_with_the_institution_failure_and_not_for_the_user_key(
    monkeypatch,
):
    from app.chat import router as chat_router

    alerted = AsyncMock()
    monkeypatch.setattr(chat_router, "record_llm_failure_and_maybe_alert", alerted)
    quota = _status_error(openai.RateLimitError, 429)

    with_institution = UserLLMError(
        REASON_REJECTED, "openai", "m", institutional_error=quota
    )
    only_user_key = UserLLMError(REASON_QUOTA, "openai", "m")

    assert await chat_router._alert_llm_failure(None, "stream", with_institution) is (
        with_institution
    )
    alerted.assert_awaited_once_with(None, endpoint="stream", error=quota)

    alerted.reset_mock()
    await chat_router._alert_llm_failure(None, "stream", only_user_key)
    alerted.assert_not_awaited()


async def test_stream_falls_back_to_the_user_key_and_reports_it_on_open():
    provider = _with_user()
    provider.client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.RateLimitError, 429)
    )
    provider.fallback_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.RateLimitError, 429)
    )

    async def _chunks():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hola"
        chunk.usage = None
        yield chunk

    provider.user_client.chat.completions.create = AsyncMock(return_value=_chunks())
    opened: list[StreamLink] = []

    texts = [
        c.text
        async for c in provider.chat_completion_stream(
            [{"role": "user", "content": "hola"}], on_open=opened.append
        )
    ]

    assert texts == ["hola"]
    assert opened == [
        StreamLink(
            model="claude-sonnet-5-5", provider="anthropic", fallback=True, byok=True
        )
    ]


async def test_stream_with_a_rejected_user_key_raises_user_llm_error():
    provider = _with_user().user_only()
    provider.user_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.AuthenticationError, 401)
    )

    with pytest.raises(UserLLMError) as caught:
        async for _ in provider.chat_completion_stream(
            [{"role": "user", "content": "x"}]
        ):
            pass
    assert caught.value.reason == REASON_REJECTED


def test_find_user_llm_error_looks_through_the_router_503():
    original = UserLLMError(REASON_REJECTED, "openai", "m")
    try:
        try:
            raise original
        except Exception as exc:
            raise HTTPException(status_code=503, detail="no disponible") from exc
    except HTTPException as wrapped:
        assert find_user_llm_error(wrapped) is original
    assert find_user_llm_error(HTTPException(status_code=503)) is None


# ============================================================
# Dependency de FastAPI
# ============================================================


def _request(method: str, body: bytes) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": method, "headers": []}, receive)


async def test_dependency_adds_the_user_link_only_when_the_body_brings_one(monkeypatch):
    with_key = await get_llm_provider(
        _request("POST", json.dumps({"user_llm": _VALID}).encode())
    )
    without = await get_llm_provider(_request("POST", b'{"course_id": 1}'))
    get_request = await get_llm_provider(_request("GET", b""))

    assert with_key.has_user_llm and not without.has_user_llm
    assert not get_request.has_user_llm

    monkeypatch.setattr(get_settings(), "user_llm_enabled", False)
    switched_off = await get_llm_provider(
        _request("POST", json.dumps({"user_llm": _VALID}).encode())
    )
    assert not switched_off.has_user_llm


# ============================================================
# Registro de consumo: costo 0 para la facultad
# ============================================================


async def test_byok_usage_costs_nothing_to_the_institution(monkeypatch):
    added = []
    session = MagicMock()
    session.add = added.append
    session.commit = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    lookup = AsyncMock()
    monkeypatch.setattr("app.db.session.get_ledger_session_factory", lambda: factory)
    monkeypatch.setattr(usage_ledger, "_lookup_price", lookup)

    record = UsageRecord(
        kind="llm", provider="anthropic", model="m", byok=True, prompt_tokens=500
    )
    row = usage_ledger.build_row(record, UsageContext())
    summary = usage_ledger.call_summary(record, UsageContext())
    await usage_ledger._persist(row, record, summary)

    lookup.assert_not_awaited()
    assert added[0].byok is True and added[0].cost_usd == 0
    assert summary["byok"] is True and summary["cost_usd"] == "0"


# ============================================================
# Chat: presupuesto agotado → sigue con la key del usuario
# ============================================================


def _chat_payload():
    from app.chat.schemas import ChatRequest

    return ChatRequest(question="¿Qué es una matriz?", course_id=3, user_id=7)


def _budget_redis(used_hourly: int, used_daily: int) -> MagicMock:
    """Redis falso donde la ventana horaria y la diaria ya traen ese consumo."""
    totals = {3600: used_hourly, 86400: used_daily}
    redis = MagicMock()
    redis.increments = []  # (ventana, cantidad) de cada incrby, para verificar

    def pipeline():
        pipe = MagicMock()
        state = {}

        def incrby(key, amount):
            window = int(key.split(":")[2])
            redis.increments.append((window, amount))
            state["total"] = totals[window] + amount

        pipe.incrby = incrby
        pipe.expire = MagicMock()
        pipe.execute = AsyncMock(side_effect=lambda: [state["total"], True])
        return pipe

    redis.pipeline = pipeline
    redis.decrby = AsyncMock()
    redis.incrby = AsyncMock()
    redis.expire = AsyncMock()
    return redis


async def test_chat_budget_exhausted_with_user_key_continues_with_it_only():
    from app.chat.router import _reserve_chat_budget

    request = _request("POST", b"")
    redis = _budget_redis(used_hourly=0, used_daily=10**9)
    llm = _with_user()

    hourly, daily, chosen = await _reserve_chat_budget(
        request=request, payload=_chat_payload(), redis=redis, llm=llm
    )

    assert (hourly, daily) == (0, 0)
    assert chosen.is_user_only
    # La reserva horaria que sí se tomó se devolvió entera.
    hourly_moves = [amount for window, amount in redis.increments if window == 3600]
    assert len(hourly_moves) == 2 and sum(hourly_moves) == 0


async def test_chat_budget_exhausted_without_user_key_still_answers_429():
    from app.chat.router import _reserve_chat_budget

    with pytest.raises(HTTPException) as caught:
        await _reserve_chat_budget(
            request=_request("POST", b""),
            payload=_chat_payload(),
            redis=_budget_redis(used_hourly=10**9, used_daily=0),
            llm=LLMProvider(),
        )
    assert caught.value.status_code == 429


# ============================================================
# Endpoint de prueba y handler 424
# ============================================================


@pytest.fixture
async def api():
    from app.main import app

    app.dependency_overrides[verify_hmac] = lambda: json.dumps(
        {"user_id": 7, "user_llm": _VALID}
    ).encode()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_test_endpoint_answers_ok_when_the_provider_responds(api):
    from app.main import app

    provider = _with_user()
    provider.user_client.chat.completions.create = AsyncMock(return_value=_response())
    app.dependency_overrides[get_llm_provider] = lambda: provider

    response = await api.post(
        "/api/v1/user-llm/test", json={"user_id": 7, "user_llm": _VALID}
    )

    assert response.status_code == 200
    assert response.json() | {"latency_ms": 0} == {
        "ok": True,
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "latency_ms": 0,
    }


async def test_test_endpoint_returns_424_with_the_reason_when_the_key_is_rejected(api):
    from app.main import app

    provider = _with_user()
    provider.user_client.chat.completions.create = AsyncMock(
        side_effect=_status_error(openai.AuthenticationError, 401)
    )
    app.dependency_overrides[get_llm_provider] = lambda: provider

    response = await api.post(
        "/api/v1/user-llm/test",
        json={"user_id": 7, "user_llm": _VALID},
        headers={"Accept-Language": "en"},
    )

    assert response.status_code == 424
    detail = response.json()["detail"]
    assert (detail["error"], detail["reason"], detail["provider"]) == (
        "user_llm_failed",
        "rejected",
        "anthropic",
    )
    assert detail["message"].startswith("Anthropic rejected your API key")
    assert "sk-ant" not in response.text


async def test_router_503_caused_by_the_user_key_becomes_424(api):
    """Los routers envuelven la falla del LLM en un 503: el handler lo
    convierte en el error de la key del usuario."""
    from app.main import app

    @app.post("/__test_wrapped_user_llm")
    async def wrapped():
        try:
            raise UserLLMError(REASON_QUOTA, "openai", "m")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="no disponible") from exc

    try:
        response = await api.post("/__test_wrapped_user_llm")
    finally:
        app.router.routes = [
            r
            for r in app.router.routes
            if getattr(r, "path", "") != "/__test_wrapped_user_llm"
        ]

    assert response.status_code == 424
    assert response.json()["detail"]["reason"] == "quota"


async def test_plain_http_errors_keep_their_shape(api):
    response = await api.get("/no-existe")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@pytest.fixture(autouse=True)
def _clean_usage_context():
    set_usage_context(UsageContext())
    yield
    set_usage_context(UsageContext())
