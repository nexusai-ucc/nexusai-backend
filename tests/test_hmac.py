"""
Tests del módulo app.auth.hmac.

Cubre los caminos de fallo críticos:
  - API key inválida o ausente
  - Timestamp expirado / fuera de ventana
  - Firma HMAC inválida (body modificado, secret distinto, etc.)
  - Replay (nonce ya usado)
  - Happy path: request bien firmada pasa

Convención: usamos `hmac` y `hashlib` directamente para construir firmas
válidas en los tests, así verificamos contra el algoritmo real (no contra
otra implementación nuestra).
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.hmac import verify_hmac
from app.infrastructure.redis_client import get_redis
from app.shared.config import get_settings


def _build_signature(secret: str, timestamp: str, nonce: str, body: bytes) -> str:
    """Réplica de la firma del lado PHP (ver investigacion/05-backend-fastapi/)."""
    signed_string = (timestamp + nonce).encode("utf-8") + body
    return hmac_lib.new(
        secret.encode("utf-8"),
        signed_string,
        hashlib.sha256,
    ).hexdigest()


@pytest.fixture
def app(fake_redis):
    """
    Mini FastAPI con un endpoint protegido por verify_hmac y Redis mockeado.
    No depende del main.py real para mantener los tests aislados.
    """
    settings = get_settings()
    test_app = FastAPI()

    @test_app.post("/protected")
    async def protected(_body: bytes = pytest.importorskip("fastapi").Depends(verify_hmac)):
        return {"ok": True}

    # Override de la dependency get_redis para usar el mock.
    test_app.dependency_overrides[get_redis] = lambda: fake_redis
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


def _good_headers(body: bytes) -> dict[str, str]:
    """Headers válidos para una request firmada correctamente."""
    settings = get_settings()
    timestamp = str(int(time.time()))
    nonce = "test-nonce-12345"
    signature = _build_signature(
        settings.nexusai_shared_secret, timestamp, nonce, body
    )
    return {
        "Authorization": f"Bearer {settings.nexusai_api_key}",
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


# ============================================================
# Happy path
# ============================================================

def test_happy_path(client):
    body = b'{"hello": "world"}'
    response = client.post("/protected", headers=_good_headers(body), content=body)
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ============================================================
# Capa 1 — Bearer API key
# ============================================================

def test_missing_authorization(client):
    body = b'{"hello": "world"}'
    headers = _good_headers(body)
    del headers["Authorization"]
    response = client.post("/protected", headers=headers, content=body)
    # FastAPI devuelve 422 si falta un Header marcado required (no nuestro 401).
    assert response.status_code == 422


def test_malformed_authorization(client):
    body = b'{"hello": "world"}'
    headers = _good_headers(body)
    headers["Authorization"] = "NotBearer xxx"
    response = client.post("/protected", headers=headers, content=body)
    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"]


def test_invalid_api_key(client):
    body = b'{"hello": "world"}'
    headers = _good_headers(body)
    headers["Authorization"] = "Bearer wrong-api-key"
    response = client.post("/protected", headers=headers, content=body)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


# ============================================================
# Capa 2 — Timestamp / firma
# ============================================================

def test_invalid_timestamp_format(client):
    body = b'{"hello": "world"}'
    headers = _good_headers(body)
    headers["X-Timestamp"] = "not-a-number"
    response = client.post("/protected", headers=headers, content=body)
    assert response.status_code == 401
    assert "X-Timestamp" in response.json()["detail"]


def test_expired_timestamp(client):
    body = b'{"hello": "world"}'
    settings = get_settings()
    # 10 minutos en el pasado, fuera de la ventana de 5 min.
    expired_ts = str(int(time.time()) - 600)
    nonce = "expired-test-nonce"
    signature = _build_signature(
        settings.nexusai_shared_secret, expired_ts, nonce, body
    )
    headers = {
        "Authorization": f"Bearer {settings.nexusai_api_key}",
        "X-Timestamp": expired_ts,
        "X-Nonce": nonce,
        "X-Signature": signature,
    }
    response = client.post("/protected", headers=headers, content=body)
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_invalid_signature(client):
    """Body modificado → firma ya no matchea → 401."""
    original_body = b'{"hello": "world"}'
    headers = _good_headers(original_body)
    # Mandamos un body distinto al firmado.
    response = client.post(
        "/protected", headers=headers, content=b'{"hello": "tampered"}'
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid signature"


def test_signature_with_wrong_secret(client):
    body = b'{"hello": "world"}'
    settings = get_settings()
    timestamp = str(int(time.time()))
    nonce = "wrong-secret-nonce"
    # Firmamos con un secret distinto al del server.
    bad_signature = _build_signature("otro-secret-distinto", timestamp, nonce, body)
    headers = {
        "Authorization": f"Bearer {settings.nexusai_api_key}",
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": bad_signature,
    }
    response = client.post("/protected", headers=headers, content=body)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid signature"


# ============================================================
# Capa 3 — Anti-replay (nonce ya usado)
# ============================================================

def test_replay_detected(client, fake_redis):
    """Si Redis dice que el nonce ya existe (set NX = False), rechazar."""
    body = b'{"hello": "world"}'
    # Simular que el nonce ya estaba seteado.
    fake_redis.set.return_value = False
    response = client.post("/protected", headers=_good_headers(body), content=body)
    assert response.status_code == 401
    assert "replay" in response.json()["detail"].lower()


def test_two_distinct_nonces_pass(client, fake_redis):
    """Dos requests con nonce distinto deberían pasar ambas."""
    body = b'{"hello": "world"}'
    # Default del fixture: set devuelve True (nonce nuevo).
    response1 = client.post("/protected", headers=_good_headers(body), content=body)
    response2 = client.post("/protected", headers=_good_headers(body), content=body)
    assert response1.status_code == 200
    assert response2.status_code == 200
