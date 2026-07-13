"""
Tests del router de foros — Épica 06 (detección de duplicados).

Misma estrategia de aislamiento que test_documents_router.py:
  - Mini FastAPI solo con el forums router.
  - verify_hmac, get_db y get_embedding_provider reemplazados con mocks.
  - Sin llamadas reales a Postgres, Redis ni APIs externas.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider


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
async def client(mock_db, mock_embeddings):
    from app.forums.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/forums")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings

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
    # Verificar que se pasó el parámetro a execute (el SQL incluye :exclude_post_id)
    call_kwargs = mock_db.execute.call_args
    assert call_kwargs is not None
    params = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1].get("params", {})
    # El parámetro pasa como dict posicional en text()
    executed_params = mock_db.execute.call_args[0][1]
    assert executed_params["exclude_post_id"] == 10


async def test_similar_posts_propagates_embedding_error(client, mock_embeddings):
    """Si embeddings falla al vectorizar el texto → 503."""
    mock_embeddings.embed.side_effect = Exception("timeout")

    response = await client.post("/api/v1/forums/similar-posts", json=_SIMILAR_PAYLOAD)

    assert response.status_code == 503
