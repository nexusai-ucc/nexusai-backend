"""
Tests del router de foros — Épica 06 (detección de duplicados).

Misma estrategia de aislamiento que test_documents_router.py:
  - Mini FastAPI solo con el forums router.
  - verify_hmac, get_db y get_embedding_provider reemplazados con mocks.
  - Sin llamadas reales a Postgres, Redis ni APIs externas.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_CONTENT = "Cómo se resuelve una integral por partes es un tema importante."
_CONTENT_HASH = __import__("hashlib").sha256(_CONTENT.encode()).hexdigest()

_INDEX_PAYLOAD = {
    "post_id": 10,
    "discussion_id": 5,
    "course_id": 1,
    "content": _CONTENT,
}

_SIMILAR_PAYLOAD = {
    "course_id": 1,
    "text": "¿Cómo se resuelve una integral por partes?",
}


def _make_post_embedding(**kwargs):
    """Record de ForumPostEmbedding simulado."""
    record = SimpleNamespace(
        id=uuid4(),
        forum_post_id=10,
        discussion_id=5,
        course_id=1,
        content_hash=_CONTENT_HASH,
        content=_CONTENT,
        embedding=[0.1] * 768,
    )
    for k, v in kwargs.items():
        setattr(record, k, v)
    return record


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalar_one_or_none.return_value = None
    db.execute.return_value.all.return_value = []
    db.add = MagicMock()
    return db


@pytest.fixture
def mock_embeddings():
    emb = AsyncMock(spec=EmbeddingProvider)
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMProvider)
    llm.chat_completion.return_value = MagicMock(text="Resumen de la semana.")
    return llm


@pytest.fixture
async def client(mock_db, mock_embeddings, mock_llm):
    from app.forums.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/forums")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ─────────────────────────────────────────────────────────────
# POST /index-post — nuevo post
# ─────────────────────────────────────────────────────────────

async def test_index_new_post_returns_indexed(client, mock_db):
    """Post que no existe en DB → se indexa y devuelve status='indexed'."""
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    response = await client.post("/api/v1/forums/index-post", json=_INDEX_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["post_id"] == 10
    assert data["status"] == "indexed"
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


async def test_index_post_skips_if_content_unchanged(client, mock_db):
    """Post ya existente con mismo contenido → status='skipped', sin commit."""
    existing = _make_post_embedding(content_hash=_CONTENT_HASH)
    mock_db.execute.return_value.scalar_one_or_none.return_value = existing

    response = await client.post("/api/v1/forums/index-post", json=_INDEX_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped"
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()


async def test_index_post_reindexes_if_content_changed(client, mock_db):
    """Post existente con contenido distinto (editado) → se re-embeddea."""
    existing = _make_post_embedding(content_hash="hash-viejo-diferente")
    mock_db.execute.return_value.scalar_one_or_none.return_value = existing

    response = await client.post("/api/v1/forums/index-post", json=_INDEX_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    # add NO debe llamarse (actualiza el objeto existente, no crea uno nuevo)
    mock_db.add.assert_not_called()
    mock_db.commit.assert_called_once()
    # El hash debe haber sido actualizado en el objeto existente
    assert existing.content_hash == _CONTENT_HASH


async def test_index_post_rejects_empty_content(client):
    """Contenido vacío o muy corto → 422 de Pydantic."""
    payload = {**_INDEX_PAYLOAD, "content": ""}
    response = await client.post("/api/v1/forums/index-post", json=payload)
    assert response.status_code == 422


async def test_index_post_propagates_embedding_error(client, mock_db, mock_embeddings):
    """Si el proveedor de embeddings falla → 503."""
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    mock_embeddings.embed.side_effect = Exception("embedding API down")

    response = await client.post("/api/v1/forums/index-post", json=_INDEX_PAYLOAD)

    assert response.status_code == 503


# ─────────────────────────────────────────────────────────────
# DELETE /index-post/{post_id}
# ─────────────────────────────────────────────────────────────

async def test_delete_post_embedding_returns_204(client, mock_db):
    response = await client.delete("/api/v1/forums/index-post/10")

    assert response.status_code == 204
    mock_db.execute.assert_called_once()
    mock_db.commit.assert_called_once()


async def test_delete_nonexistent_post_still_returns_204(client, mock_db):
    """DELETE es idempotente — no falla si el post no tenía embedding."""
    response = await client.delete("/api/v1/forums/index-post/9999")
    assert response.status_code == 204


# ─────────────────────────────────────────────────────────────
# POST /similar-posts
# ─────────────────────────────────────────────────────────────

async def test_similar_posts_returns_matches(client, mock_db, mock_embeddings):
    """Hay posts similares → los devuelve con preview y similarity."""
    row = SimpleNamespace(
        forum_post_id=10,
        discussion_id=5,
        content=_CONTENT,
        similarity=0.89,
    )
    mock_db.execute.return_value.all.return_value = [row]

    response = await client.post("/api/v1/forums/similar-posts", json=_SIMILAR_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert len(data["similar_posts"]) == 1
    post = data["similar_posts"][0]
    assert post["forum_post_id"] == 10
    assert post["discussion_id"] == 5
    assert post["similarity"] == 0.89
    assert post["preview"] == _CONTENT[:200].strip()


async def test_similar_posts_empty_when_no_matches(client, mock_db):
    """Sin posts similares → lista vacía (no error)."""
    mock_db.execute.return_value.all.return_value = []

    response = await client.post("/api/v1/forums/similar-posts", json=_SIMILAR_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["similar_posts"] == []


async def test_similar_posts_rejects_short_text(client):
    """Texto muy corto (< 10 chars) → 422."""
    payload = {**_SIMILAR_PAYLOAD, "text": "corto"}
    response = await client.post("/api/v1/forums/similar-posts", json=payload)
    assert response.status_code == 422


async def test_similar_posts_respects_threshold_in_response(client, mock_db):
    """El threshold custom se refleja en la respuesta."""
    mock_db.execute.return_value.all.return_value = []

    payload = {**_SIMILAR_PAYLOAD, "threshold": 0.9}
    response = await client.post("/api/v1/forums/similar-posts", json=payload)

    assert response.status_code == 200
    assert response.json()["threshold_used"] == 0.9


async def test_similar_posts_excludes_post_id(client, mock_db, mock_embeddings):
    """Con exclude_post_id, el endpoint llama a DB con el parámetro correcto."""
    mock_db.execute.return_value.all.return_value = []

    payload = {**_SIMILAR_PAYLOAD, "exclude_post_id": 10}
    response = await client.post("/api/v1/forums/similar-posts", json=payload)

    assert response.status_code == 200
    # similar_posts usa el ORM de SQLAlchemy (select().where(...)), no text()
    # con params separados — el exclude_post_id queda bindeado directo en el
    # WHERE del statement compilado, no como segundo arg posicional de execute().
    stmt = mock_db.execute.call_args[0][0]
    assert 10 in stmt.compile().params.values()


async def test_similar_posts_propagates_embedding_error(client, mock_embeddings):
    """Si embeddings falla al vectorizar el texto → 503."""
    mock_embeddings.embed.side_effect = Exception("timeout")

    response = await client.post("/api/v1/forums/similar-posts", json=_SIMILAR_PAYLOAD)

    assert response.status_code == 503


# ─────────────────────────────────────────────────────────────
# FOR-05 (#366) — heurística de urgencia/frustración
# ─────────────────────────────────────────────────────────────

from app.forums.router import detect_urgency  # noqa: E402


def test_detect_urgency_true_for_clearly_urgent_post():
    text = "URGENTE necesito ayuda para el examen de mañana, no entiendo nada!!!"
    assert detect_urgency(text) is True


def test_detect_urgency_true_for_frustrated_english_post():
    text = "I'm so desperate, I can't understand this at all, please help!!!"
    assert detect_urgency(text) is True


def test_detect_urgency_false_for_neutral_question():
    text = "¿Podrían confirmar la fecha del parcial?"
    assert detect_urgency(text) is False


def test_detect_urgency_false_for_neutral_thanks_message():
    text = "Gracias por la clase de hoy, muy clara la explicación."
    assert detect_urgency(text) is False


def test_detect_urgency_false_for_single_signal_only():
    # Una sola palabra clave, sin más señales, no alcanza (evita falsos positivos).
    text = "Necesito ayuda con el ejercicio 3 cuando puedas."
    assert detect_urgency(text) is False


def test_detect_urgency_false_for_empty_or_blank_text():
    assert detect_urgency("") is False
    assert detect_urgency("   ") is False


def test_detect_urgency_true_for_sustained_caps_plus_keyword():
    # "URGENTISIMOOOOO" es una sola palabra en mayúsculas de 16 caracteres
    # (señal 3: grito sostenido) que además contiene "urgent" (señal 1).
    text = "URGENTISIMOOOOO alguien me ayuda"
    assert detect_urgency(text) is True


# ─────────────────────────────────────────────────────────────
# FOR-06 (#367) — digest semanal del foro
# ─────────────────────────────────────────────────────────────

_DIGEST_PAYLOAD = {
    "course_id": 1,
    "days": 7,
    "discussions": [
        {
            "discussion_id": 100,
            "discussion_name": "Dudas sobre el TP2",
            "forum_name": "Consultas generales",
            "posts": [
                {"post_id": 1, "author": "Ana", "content": "¿Cuándo entrega el TP2?"},
                {"post_id": 2, "author": "Docente", "content": "El viernes que viene."},
            ],
        },
        {
            "discussion_id": 101,
            "discussion_name": "No entiendo nada del parcial",
            "forum_name": "Consultas generales",
            "posts": [
                {"post_id": 3, "author": "Juan", "content": "URGENTE no entiendo nada del parcial, ayuda por favor!!!"},
            ],
        },
    ],
}


async def test_weekly_digest_returns_summary_and_per_discussion_urgency(client, mock_llm):
    response = await client.post("/api/v1/forums/weekly-digest", json=_DIGEST_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["discussion_count"] == 2
    assert data["summary"] == "Resumen de la semana."

    by_id = {d["discussion_id"]: d for d in data["discussions"]}
    assert by_id[100]["urgent"] is False
    assert by_id[101]["urgent"] is True
    assert by_id[100]["post_count"] == 2
    assert by_id[101]["post_count"] == 1

    mock_llm.chat_completion.assert_awaited_once()


async def test_weekly_digest_skips_llm_call_when_no_discussions(client, mock_llm):
    payload = {**_DIGEST_PAYLOAD, "discussions": []}

    response = await client.post("/api/v1/forums/weekly-digest", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "course_id": 1,
        "period_days": 7,
        "discussion_count": 0,
        "discussions": [],
        "summary": None,
    }
    mock_llm.chat_completion.assert_not_awaited()


async def test_weekly_digest_propagates_llm_error(client, mock_llm):
    mock_llm.chat_completion.side_effect = Exception("provider down")

    response = await client.post("/api/v1/forums/weekly-digest", json=_DIGEST_PAYLOAD)

    assert response.status_code == 503


async def test_weekly_digest_rejects_too_many_discussions(client):
    payload = {
        **_DIGEST_PAYLOAD,
        "discussions": [
            {
                "discussion_id": i,
                "discussion_name": f"Hilo {i}",
                "forum_name": "Foro",
                "posts": [{"post_id": i, "author": "A", "content": "hola"}],
            }
            for i in range(20)
        ],
    }

    response = await client.post("/api/v1/forums/weekly-digest", json=payload)

    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# FOR-07 (#378) — webhook externo del digest semanal
# ─────────────────────────────────────────────────────────────

async def test_weekly_digest_posts_to_webhook_when_configured(client, mock_db, mock_llm):
    mock_db.execute.return_value.scalar_one_or_none.return_value = "https://hooks.slack.com/services/xxx"

    with patch("app.forums.router.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        response = await client.post("/api/v1/forums/weekly-digest", json=_DIGEST_PAYLOAD)

    assert response.status_code == 200
    mock_client.post.assert_awaited_once_with(
        "https://hooks.slack.com/services/xxx", json={"text": "Resumen de la semana."}
    )


async def test_weekly_digest_does_not_call_webhook_when_not_configured(client, mock_db, mock_llm):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    with patch("app.forums.router.httpx.AsyncClient") as mock_client_cls:
        response = await client.post("/api/v1/forums/weekly-digest", json=_DIGEST_PAYLOAD)

    assert response.status_code == 200
    mock_client_cls.assert_not_called()


async def test_weekly_digest_webhook_failure_does_not_break_response(client, mock_db, mock_llm):
    """Criterio de aceptación explícito: un fallo del webhook no debe romper
    la generación del digest en la UI."""
    mock_db.execute.return_value.scalar_one_or_none.return_value = "https://hooks.slack.com/services/xxx"

    with patch("app.forums.router.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("webhook host unreachable")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        response = await client.post("/api/v1/forums/weekly-digest", json=_DIGEST_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["summary"] == "Resumen de la semana."


async def test_weekly_digest_does_not_call_webhook_when_no_activity(client, mock_db, mock_llm):
    mock_db.execute.return_value.scalar_one_or_none.return_value = "https://hooks.slack.com/services/xxx"
    payload = {**_DIGEST_PAYLOAD, "discussions": []}

    with patch("app.forums.router.httpx.AsyncClient") as mock_client_cls:
        response = await client.post("/api/v1/forums/weekly-digest", json=payload)

    assert response.status_code == 200
    mock_client_cls.assert_not_called()


async def test_save_webhook_config_upserts_url(client, mock_db):
    mock_db.execute.return_value.fetchone.return_value = SimpleNamespace(
        webhook_url="https://hooks.slack.com/services/xxx"
    )

    response = await client.post(
        "/api/v1/forums/webhook-config/save",
        json={"course_id": 1, "webhook_url": "https://hooks.slack.com/services/xxx"},
    )

    assert response.status_code == 200
    assert response.json() == {"webhook_url": "https://hooks.slack.com/services/xxx"}
    mock_db.commit.assert_called_once()


async def test_save_webhook_config_empty_url_deletes(client, mock_db):
    response = await client.post(
        "/api/v1/forums/webhook-config/save",
        json={"course_id": 1, "webhook_url": ""},
    )

    assert response.status_code == 200
    assert response.json() == {"webhook_url": None}
    mock_db.commit.assert_called_once()


async def test_save_webhook_config_rejects_non_http_url(client):
    response = await client.post(
        "/api/v1/forums/webhook-config/save",
        json={"course_id": 1, "webhook_url": "javascript:alert(1)"},
    )

    assert response.status_code == 422


async def test_get_webhook_config_returns_saved_url(client, mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = "https://hooks.slack.com/services/xxx"

    response = await client.post("/api/v1/forums/webhook-config/get", json={"course_id": 1})

    assert response.status_code == 200
    assert response.json() == {"webhook_url": "https://hooks.slack.com/services/xxx"}


async def test_get_webhook_config_returns_none_when_not_configured(client, mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    response = await client.post("/api/v1/forums/webhook-config/get", json={"course_id": 1})

    assert response.status_code == 200
    assert response.json() == {"webhook_url": None}
