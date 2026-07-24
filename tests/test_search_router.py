"""
Tests del buscador híbrido — BUS-02.

Router sin cobertura previa. Como la query es SQL crudo (no ORM), lo que
se puede testear de forma significativa con mocks es la lógica de Python
alrededor de la query: dedup, threshold, el corte temprano cuando no hay
course_ids válidos, manejo de errores, y el threading de material_type/
mime_type. Misma estrategia de aislamiento que test_forums_router.py /
test_quiz_router.py: mini FastAPI solo con el search router, verify_hmac/
get_db/get_embedding_provider reemplazados con mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider


def _row(**kwargs):
    defaults = dict(
        document_id="doc-1",
        content="El teorema de Bayes relaciona probabilidades condicionales.",
        chunk_index=0,
        document_filename="apunte1.pdf",
        course_id=1,
        mime_type="application/pdf",
        has_file=True,
        combined_score=0.6,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.all.return_value = [_row()]
    return db


@pytest.fixture
def mock_embeddings():
    emb = AsyncMock(spec=EmbeddingProvider)
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
async def client(mock_db, mock_embeddings):
    from app.search.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/search")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


_BASE_PAYLOAD = {"query": "bayes", "course_id": 1, "user_id": 1}


async def test_search_returns_200_with_expected_shape(client):
    response = await client.post("/api/v1/search", json=_BASE_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["document_filename"] == "apunte1.pdf"
    assert data["results"][0]["mime_type"] == "application/pdf"


async def test_search_dedups_keeping_highest_score_per_document(client, mock_db):
    mock_db.execute.return_value.all.return_value = [
        _row(document_id="doc-1", combined_score=0.5, content="fragmento débil"),
        _row(document_id="doc-1", combined_score=0.9, content="fragmento fuerte"),
    ]

    response = await client.post("/api/v1/search", json=_BASE_PAYLOAD)

    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["content"] == "fragmento fuerte"


async def test_search_filters_rows_below_threshold(client, mock_db):
    mock_db.execute.return_value.all.return_value = [
        _row(document_id="doc-1", combined_score=0.10),
        _row(document_id="doc-2", combined_score=0.40),
    ]

    response = await client.post("/api/v1/search", json=_BASE_PAYLOAD)

    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["document_id"] == "doc-2"


async def test_search_returns_empty_when_no_valid_course_ids(client, mock_db):
    payload = {**_BASE_PAYLOAD, "course_ids": [0, -5]}

    response = await client.post("/api/v1/search", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data == {"query": "bayes", "results": [], "total": 0}
    mock_db.execute.assert_not_called()


async def test_search_returns_503_on_db_error(client, mock_db):
    mock_db.execute.side_effect = RuntimeError("connection lost")

    response = await client.post("/api/v1/search", json=_BASE_PAYLOAD)

    assert response.status_code == 503


async def test_material_type_is_threaded_into_query_params(client, mock_db):
    payload = {**_BASE_PAYLOAD, "material_type": "application/pdf"}

    await client.post("/api/v1/search", json=payload)

    params = mock_db.execute.call_args[0][1]
    assert params["material_type"] == "application/pdf"


async def test_material_type_defaults_to_none(client, mock_db):
    await client.post("/api/v1/search", json=_BASE_PAYLOAD)

    params = mock_db.execute.call_args[0][1]
    assert params["material_type"] is None
