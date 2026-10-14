"""
Consumo por pedido en el encabezado X-NexusAI-Usage (DATA-04, issue #524).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.shared.middleware import RequestIDMiddleware
from app.shared.usage_ledger import (
    MAX_HEADER_CALLS,
    UsageRecord,
    record_usage,
    usage_header_value,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/two-calls")
    async def two_calls():
        await record_usage(
            UsageRecord(kind="llm", model="m", provider="p", prompt_tokens=10)
        )
        await record_usage(UsageRecord(kind="embedding", embedding_tokens=4))
        return {"ok": True}

    @app.get("/no-calls")
    async def no_calls():
        return {"ok": True}

    return app


async def test_json_response_carries_the_calls_of_the_request():
    with patch("app.shared.usage_ledger._persist", new=AsyncMock()):
        async with AsyncClient(
            transport=ASGITransport(app=_app()), base_url="http://t"
        ) as c:
            response = await c.get("/two-calls")

    usage = json.loads(response.headers["X-NexusAI-Usage"])
    assert [call["kind"] for call in usage["calls"]] == ["llm", "embedding"]
    assert usage["calls"][0]["prompt_tokens"] == 10
    assert usage["calls"][1]["embedding_tokens"] == 4


async def test_no_header_when_nothing_was_called():
    async with AsyncClient(
        transport=ASGITransport(app=_app()), base_url="http://t"
    ) as c:
        response = await c.get("/no-calls")
    assert "X-NexusAI-Usage" not in response.headers


async def test_calls_are_counted_even_when_the_ledger_is_off():
    with patch(
        "app.shared.usage_ledger.get_settings",
        return_value=type("S", (), {"usage_ledger_enabled": False})(),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=_app()), base_url="http://t"
        ) as c:
            response = await c.get("/two-calls")
    assert len(json.loads(response.headers["X-NexusAI-Usage"])["calls"]) == 2


def test_many_calls_are_merged_into_one_entry():
    calls = [
        {"kind": "embedding", "embedding_tokens": 1, "cost_usd": "0.00000001"}
        for _ in range(MAX_HEADER_CALLS + 5)
    ]
    items = json.loads(usage_header_value(calls))["calls"]
    assert len(items) == MAX_HEADER_CALLS + 1
    assert items[-1]["merged_calls"] == 5
    assert items[-1]["embedding_tokens"] == 5
    assert items[-1]["cost_usd"] == "0.00000005"


def test_header_is_ascii():
    value = usage_header_value([{"feature": "chat:ñandú"}])
    value.encode("ascii")


def test_ledger_can_stop_storing_the_user():
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from app.shared.usage_ledger import UsageContext, build_row

    ctx = UsageContext(feature="chat.stream", course_id=3, user_id=7, role="student")
    record = UsageRecord(kind="llm", prompt_tokens=1)
    with _patch(
        "app.shared.usage_ledger.get_settings",
        return_value=SimpleNamespace(usage_ledger_store_user_id=False),
    ):
        row = build_row(record, ctx)
    assert row["user_id"] is None
    assert row["course_id"] == 3
    assert row["role"] == "student"
