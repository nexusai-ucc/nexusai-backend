"""Regresión del hallazgo de audit sobre ADR-012: sin capturar la excepción
DENTRO de RequestIDMiddleware, un bug no manejado (no HTTPException) nunca
pasaba por el chequeo de record_5xx_and_maybe_alert — se resolvía en
Starlette's ServerErrorMiddleware, que queda afuera de RequestIDMiddleware,
así que el 5xx más grave (un crash real) no disparaba ninguna alerta.

Nota: un @app.exception_handler(Exception) NO alcanza para arreglar esto —
Starlette lo resuelve en ServerErrorMiddleware igual (trata Exception/500
como caso especial), que sigue quedando afuera del middleware de usuario.
Se probó y falló antes de este fix; el arreglo real es atajar la excepción
adentro de dispatch() (ver app/shared/middleware.py).

No se importa app.main (arrastra DB engine, routers reales, etc. — mismo
criterio que test_hmac.py) — se arma una app mínima con el mismo
RequestIDMiddleware.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.shared.middleware import RequestIDMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("bug no anticipado")

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    return app


def test_unhandled_exception_returns_500_and_triggers_5xx_alert():
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch(
        "app.shared.middleware.record_5xx_and_maybe_alert", new=AsyncMock()
    ) as mock_alert:
        response = client.get("/boom")

    assert response.status_code == 500
    mock_alert.assert_awaited_once()
    _, kwargs = mock_alert.call_args
    assert kwargs["status_code"] == 500
    assert kwargs["path"] == "/boom"


def test_unhandled_exception_still_gets_request_id_header():
    """El crash tiene que seguir devolviendo X-Request-ID — no solo alertar."""
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch("app.shared.middleware.record_5xx_and_maybe_alert", new=AsyncMock()):
        response = client.get("/boom")

    assert "X-Request-ID" in response.headers


def test_happy_path_does_not_trigger_5xx_alert():
    app = _build_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch(
        "app.shared.middleware.record_5xx_and_maybe_alert", new=AsyncMock()
    ) as mock_alert:
        response = client.get("/ok")

    assert response.status_code == 200
    mock_alert.assert_not_awaited()
