"""
Tests del router de voz — VOICE-01 (issue #314).

Misma estrategia de aislamiento que test_forums_router.py: mini FastAPI solo
con el router de voz, verify_hmac y get_transcription_provider reemplazados
con mocks. Sin llamadas reales a Groq.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.providers.transcription import TranscriptionProvider, get_transcription_provider

_AUDIO_BYTES = b"fake-audio-bytes-not-a-real-webm-file"
_AUDIO_B64 = base64.b64encode(_AUDIO_BYTES).decode()


@pytest.fixture
def mock_provider():
    provider = AsyncMock(spec=TranscriptionProvider)
    provider.transcribe.return_value = "¿Qué entra en el parcial?"
    return provider


@pytest.fixture
async def client(mock_provider):
    from app.voice.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/voice")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_transcription_provider] = lambda: mock_provider

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
async def client_unconfigured():
    """Sin GROQ_API_KEY configurada — get_transcription_provider devuelve None."""
    from app.voice.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/voice")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_transcription_provider] = lambda: None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_transcribe_returns_text(client, mock_provider):
    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": _AUDIO_B64, "mime_type": "audio/webm"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "¿Qué entra en el parcial?"}
    mock_provider.transcribe.assert_awaited_once()


async def test_transcribe_defaults_language_to_spanish(client, mock_provider):
    await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": _AUDIO_B64, "mime_type": "audio/webm"},
    )

    _, kwargs = mock_provider.transcribe.call_args
    assert kwargs["language"] == "es"


async def test_transcribe_rejects_invalid_base64(client):
    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": "not-valid-base64!!!", "mime_type": "audio/webm"},
    )

    assert response.status_code == 400


async def test_transcribe_rejects_empty_audio(client):
    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": "", "mime_type": "audio/webm"},
    )

    assert response.status_code == 422  # min_length=1 en el schema


async def test_transcribe_rejects_unsupported_mime_type(client):
    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": _AUDIO_B64, "mime_type": "video/mp4"},
    )

    assert response.status_code == 422


async def test_transcribe_rejects_audio_too_large(client):
    huge = base64.b64encode(b"x" * (10 * 1024 * 1024 + 1)).decode()

    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": huge, "mime_type": "audio/webm"},
    )

    assert response.status_code == 413


async def test_transcribe_returns_503_when_not_configured(client_unconfigured):
    """Sin GROQ_API_KEY: 503 explícito, sin intentar nada."""
    response = await client_unconfigured.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": _AUDIO_B64, "mime_type": "audio/webm"},
    )

    assert response.status_code == 503
    assert "no configurada" in response.json()["detail"].lower()


async def test_transcribe_propagates_provider_error_as_503(client, mock_provider):
    mock_provider.transcribe.side_effect = Exception("Groq API down")

    response = await client.post(
        "/api/v1/voice/transcribe",
        json={"content_b64": _AUDIO_B64, "mime_type": "audio/webm"},
    )

    assert response.status_code == 503
