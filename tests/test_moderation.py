"""
Tests de la capa de moderación de contenido (app/shared/moderation.py).

Cubre:
  - Contenido aceptable pasa (vía Moderation API de OpenAI y vía fallback LLM).
  - Contenido problemático se bloquea con un mensaje claro.
  - Fail-safe: si la moderación misma falla, no rompe — respeta
    MODERATION_FAIL_OPEN (default true = deja pasar).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.providers.llm import LLMProvider
from app.shared import moderation


def _settings(
    *,
    moderation_enabled: bool = True,
    moderation_api_key: str | None = None,
    moderation_fail_open: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        moderation_enabled=moderation_enabled,
        moderation_api_key=moderation_api_key,
        moderation_fail_open=moderation_fail_open,
    )


def _mock_llm(response_text: str) -> AsyncMock:
    llm = AsyncMock(spec=LLMProvider)
    llm.chat_completion.return_value = MagicMock(text=response_text)
    return llm


def _openai_response(*, flagged: bool, categories: dict[str, bool] | None = None) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "results": [{"flagged": flagged, "categories": categories or {}}]
    }
    return response


# ─────────────────────────────────────────────────────────────
# Deshabilitado / texto vacío
# ─────────────────────────────────────────────────────────────

async def test_disabled_allows_everything():
    with patch("app.shared.moderation.get_settings", return_value=_settings(moderation_enabled=False)):
        result = await moderation.moderate_text("cualquier cosa horrible", llm=None)

    assert result.allowed is True
    assert result.source == "disabled"


async def test_empty_text_is_allowed_without_calling_anything():
    with patch("app.shared.moderation.get_settings", return_value=_settings(moderation_api_key="sk-test")):
        result = await moderation.moderate_text("   ", llm=None)

    assert result.allowed is True
    assert result.source == "disabled"


# ─────────────────────────────────────────────────────────────
# Camino primario: Moderation API de OpenAI
# ─────────────────────────────────────────────────────────────

async def test_openai_api_allows_acceptable_content():
    settings = _settings(moderation_api_key="sk-test")
    with patch("app.shared.moderation.get_settings", return_value=settings):
        with patch("app.shared.moderation.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _openai_response(flagged=False)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await moderation.moderate_text("¿Cómo resuelvo esta integral?", llm=None)

    assert result.allowed is True
    assert result.source == "openai_api"


async def test_openai_api_blocks_flagged_content_with_clear_message():
    settings = _settings(moderation_api_key="sk-test")
    with patch("app.shared.moderation.get_settings", return_value=settings):
        with patch("app.shared.moderation.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _openai_response(
                flagged=True, categories={"harassment": True, "violence": False}
            )
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            result = await moderation.moderate_text("contenido inapropiado", llm=None)

    assert result.allowed is False
    assert result.source == "openai_api"
    assert result.categories == ["harassment"]
    assert result.blocked_message == moderation.BLOCKED_MESSAGE
    assert result.blocked_message  # mensaje no vacío / claro para el frontend


# ─────────────────────────────────────────────────────────────
# Fallback agnóstico: clasificación vía el LLM activo (sin MODERATION_API_KEY)
# ─────────────────────────────────────────────────────────────

async def test_llm_fallback_allows_acceptable_content_when_no_api_key():
    settings = _settings(moderation_api_key=None)
    llm = _mock_llm(json.dumps({"flagged": False, "categories": []}))

    with patch("app.shared.moderation.get_settings", return_value=settings):
        result = await moderation.moderate_text("¿Qué es el teorema de Bayes?", llm=llm)

    assert result.allowed is True
    assert result.source == "llm_fallback"
    llm.chat_completion.assert_awaited_once()


async def test_llm_fallback_blocks_flagged_content():
    settings = _settings(moderation_api_key=None)
    llm = _mock_llm(json.dumps({"flagged": True, "categories": ["hate"]}))

    with patch("app.shared.moderation.get_settings", return_value=settings):
        result = await moderation.moderate_text("mensaje de odio", llm=llm)

    assert result.allowed is False
    assert result.source == "llm_fallback"
    assert result.categories == ["hate"]
    assert result.blocked_message == moderation.BLOCKED_MESSAGE


async def test_llm_fallback_tolerates_markdown_fenced_json():
    settings = _settings(moderation_api_key=None)
    llm = _mock_llm('```json\n{"flagged": false, "categories": []}\n```')

    with patch("app.shared.moderation.get_settings", return_value=settings):
        result = await moderation.moderate_text("pregunta normal", llm=llm)

    assert result.allowed is True


async def test_falls_back_to_llm_when_openai_api_raises():
    settings = _settings(moderation_api_key="sk-test")
    llm = _mock_llm(json.dumps({"flagged": False, "categories": []}))

    with patch("app.shared.moderation.get_settings", return_value=settings):
        with patch("app.shared.moderation.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__.side_effect = Exception("openai down")

            result = await moderation.moderate_text("pregunta normal", llm=llm)

    assert result.allowed is True
    assert result.source == "llm_fallback"
    llm.chat_completion.assert_awaited_once()


# ─────────────────────────────────────────────────────────────
# Fail-safe: la moderación NO debe romper el endpoint que la usa.
# ─────────────────────────────────────────────────────────────

async def test_fail_open_by_default_when_every_path_fails():
    """Con MODERATION_FAIL_OPEN=true (default), un fallo total de la
    moderación deja pasar el mensaje en vez de bloquear al alumno."""
    settings = _settings(moderation_api_key="sk-test", moderation_fail_open=True)
    llm = AsyncMock(spec=LLMProvider)
    llm.chat_completion.side_effect = Exception("LLM también caído")

    with patch("app.shared.moderation.get_settings", return_value=settings):
        with patch("app.shared.moderation.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__.side_effect = Exception("openai down")

            result = await moderation.moderate_text("texto cualquiera", llm=llm)

    assert result.allowed is True
    assert result.source == "fail_open"


async def test_fail_closed_when_configured_and_everything_fails():
    """Con MODERATION_FAIL_OPEN=false, un fallo total bloquea en vez de
    dejar pasar contenido no verificado."""
    settings = _settings(moderation_api_key="sk-test", moderation_fail_open=False)
    llm = AsyncMock(spec=LLMProvider)
    llm.chat_completion.side_effect = Exception("LLM también caído")

    with patch("app.shared.moderation.get_settings", return_value=settings):
        with patch("app.shared.moderation.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__.side_effect = Exception("openai down")

            result = await moderation.moderate_text("texto cualquiera", llm=llm)

    assert result.allowed is False
    assert result.source == "fail_closed"
    assert result.blocked_message


async def test_fail_open_without_any_llm_available():
    """Sin API key y sin LLM inyectado (llm=None), no hay ningún camino de
    clasificación posible — debe resolver vía el fail-safe, no explotar."""
    settings = _settings(moderation_api_key=None, moderation_fail_open=True)

    with patch("app.shared.moderation.get_settings", return_value=settings):
        result = await moderation.moderate_text("texto cualquiera", llm=None)

    assert result.allowed is True
    assert result.source == "fail_open"


async def test_malformed_llm_json_falls_back_to_fail_safe_not_crash():
    settings = _settings(moderation_api_key=None, moderation_fail_open=True)
    llm = _mock_llm("esto no es JSON")

    with patch("app.shared.moderation.get_settings", return_value=settings):
        result = await moderation.moderate_text("texto cualquiera", llm=llm)

    assert result.allowed is True
    assert result.source == "fail_open"
