"""
Contrato de firma HMAC con el plugin (TPC-07, #579).

`NexusAI/contracts/hmac_vectors.json` es la fuente única que leen los dos
lados: este test verifica que el `verify_hmac` REAL acepta cada vector, y
`plugin/local/nexusai/tests/contract_test.php` verifica que el plugin produce
la misma firma. Si un lado cambia el algoritmo (orden de concatenación,
encoding, hex en mayúsculas…), falla el CI de ese lado aunque sus propios
tests sigan pasando.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import hmac as hmac_module
from app.auth.hmac import verify_hmac
from app.infrastructure.redis_client import get_redis
from app.shared.config import get_settings

CONTRACT = Path(__file__).resolve().parents[3] / "contracts" / "hmac_vectors.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT.exists(), reason="contracts/ no está en este checkout"
)


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _vectors() -> list[dict]:
    return _contract()["vectors"] if CONTRACT.exists() else []


@pytest.fixture
def client(fake_redis, monkeypatch):
    # Arrange compartido: el backend usa el secret del contrato.
    monkeypatch.setattr(get_settings(), "nexusai_shared_secret", _contract()["secret"])
    app = FastAPI()

    @app.post("/protected")
    async def protected(_body: bytes = Depends(verify_hmac)):
        return {"ok": True}

    app.dependency_overrides[get_redis] = lambda: fake_redis
    return TestClient(app)


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: v["name"])
def test_backend_accepts_the_contract_signature(client, monkeypatch, vector):
    """Criterio: una request firmada como dice el contrato pasa la verificación real."""
    # Arrange: el reloj del backend en el instante del vector (ventana anti-replay).
    monkeypatch.setattr(hmac_module.time, "time", lambda: int(vector["timestamp"]))
    headers = {
        "Authorization": f"Bearer {get_settings().nexusai_api_key}",
        "X-Timestamp": vector["timestamp"],
        "X-Nonce": vector["nonce"],
        "X-Signature": vector["signature"],
        "Content-Type": "application/json",
    }

    # Act.
    response = client.post(
        "/protected", headers=headers, content=vector["body"].encode("utf-8")
    )

    # Assert.
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("vector", _vectors(), ids=lambda v: v["name"])
def test_backend_rejects_the_signature_of_another_body(client, monkeypatch, vector):
    """Criterio: la misma firma con un byte distinto en el body se rechaza (integridad)."""
    # Arrange.
    monkeypatch.setattr(hmac_module.time, "time", lambda: int(vector["timestamp"]))
    headers = {
        "Authorization": f"Bearer {get_settings().nexusai_api_key}",
        "X-Timestamp": vector["timestamp"],
        "X-Nonce": vector["nonce"] + "-otro",
        "X-Signature": vector["signature"],
    }

    # Act.
    response = client.post(
        "/protected", headers=headers, content=(vector["body"] + " ").encode("utf-8")
    )

    # Assert.
    assert response.status_code == 401
