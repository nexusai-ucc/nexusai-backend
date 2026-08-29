"""
Tests del router de documentos (POST, GET, DELETE).

Estrategia de aislamiento:
  - Mini FastAPI solo con el documents router (sin lifespan de main.py).
  - `verify_hmac`, `get_db` y `get_embedding_provider` se reemplazan con mocks.
  - `_index_document_task` se parchea para que el background task no intente
    abrir una sesión de DB real.
  - Sin llamadas reales a Postgres, Redis ni APIs externas.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider

# ============================================================
# Helpers de fixtures
# ============================================================

_PDF_BYTES = b"%PDF-1.4 minimal"
_PDF_B64 = base64.b64encode(_PDF_BYTES).decode()
_PDF_HASH = hashlib.sha256(_PDF_B64.encode()).hexdigest()

_BASE_PAYLOAD: dict = {
    "course_id": 1,
    "uploader_id": 42,
    "filename": "apuntes.pdf",
    "mime_type": "application/pdf",
    "content_b64": _PDF_B64,
}


def _make_doc(**kwargs) -> SimpleNamespace:
    """Documento simulado compatible con DocumentOut.from_orm()."""
    now = datetime.now(timezone.utc)
    doc = SimpleNamespace(
        id=uuid4(),
        course_id=1,
        uploader_id=42,
        filename="apuntes.pdf",
        mime_type="application/pdf",
        section=None,
        status="pending",
        error_message=None,
        file_hash=_PDF_HASH,
        created_at=now,
        updated_at=now,
    )
    for k, v in kwargs.items():
        setattr(doc, k, v)
    return doc


def _exec_result(scalar=None, scalars_all=None) -> MagicMock:
    """MagicMock que simula el valor de retorno de db.execute()."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = scalar
    r.scalars.return_value.all.return_value = scalars_all if scalars_all is not None else []
    return r


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_db():
    """AsyncSession mockeada. Por default: ningún documento existente.

    `db.add` es MagicMock (sync) porque SQLAlchemy AsyncSession.add() es sync,
    y aprovechamos el side_effect para popular `doc.id` como haría SQLAlchemy
    al ejecutar el INSERT en una DB real.
    """
    db = AsyncMock()
    db.execute.return_value = _exec_result()
    db.add = MagicMock(
        side_effect=lambda doc: setattr(doc, "id", doc.id or uuid4())
    )
    return db


@pytest.fixture
def mock_embeddings():
    emb = AsyncMock(spec=EmbeddingProvider)
    emb.embed.return_value = [0.1] * 768
    return emb


@pytest.fixture
async def client(mock_db, mock_embeddings):
    """AsyncClient contra una mini app con solo el documents router."""
    from app.documents.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/documents")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_embedding_provider] = lambda: mock_embeddings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _no_bg_task():
    """Previene que el background task intente abrir una sesión de DB real."""
    with patch("app.documents.router._index_document_task", AsyncMock()):
        yield


# ============================================================
# POST /api/v1/documents — validaciones
# ============================================================

async def test_upload_rejects_unsupported_mime_type(client):
    payload = {**_BASE_PAYLOAD, "mime_type": "image/png"}
    response = await client.post("/api/v1/documents", json=payload)
    assert response.status_code == 415


async def test_upload_rejects_invalid_base64(client):
    payload = {**_BASE_PAYLOAD, "content_b64": "!!!esto-no-es-base64!!!"}
    response = await client.post("/api/v1/documents", json=payload)
    assert response.status_code == 400
    assert "base64" in response.json()["detail"].lower()


# ============================================================
# POST /api/v1/documents — happy path (nuevo documento)
# ============================================================

async def test_upload_new_document_returns_202(client, mock_db):
    """Documento nuevo: crea el record en DB y devuelve 202 con status=pending."""
    # No existe documento con ese hash → execute devuelve None
    mock_db.execute.return_value = _exec_result(scalar=None)

    response = await client.post("/api/v1/documents", json=_BASE_PAYLOAD)

    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["status"] == "pending"
    assert data["filename"] == "apuntes.pdf"
    assert data["course_id"] == 1


async def test_upload_with_section_round_trips_in_response(client, mock_db):
    """El campo `section` (BUS-05) viaja del payload a la respuesta sin perderse."""
    mock_db.execute.return_value = _exec_result(scalar=None)

    payload = {**_BASE_PAYLOAD, "section": 2}
    response = await client.post("/api/v1/documents", json=payload)

    assert response.status_code == 202
    assert response.json()["section"] == 2


async def test_upload_without_section_defaults_to_none(client, mock_db):
    mock_db.execute.return_value = _exec_result(scalar=None)

    response = await client.post("/api/v1/documents", json=_BASE_PAYLOAD)

    assert response.json()["section"] is None


# ============================================================
# POST /api/v1/documents — dedup CONT-04
# ============================================================

async def test_upload_existing_hash_returns_200(client, mock_db):
    """Si el mismo archivo ya está indexado, devuelve 200 con el doc existente."""
    existing = _make_doc(status="indexed")
    mock_db.execute.return_value = _exec_result(scalar=existing)

    response = await client.post("/api/v1/documents", json=_BASE_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert str(existing.id) == data["id"]
    assert data["status"] == "indexed"
    # No debe haber creado un nuevo documento
    mock_db.add.assert_not_called()


# ============================================================
# GET /api/v1/documents — lista por curso
# ============================================================

async def test_list_documents_empty(client, mock_db):
    """Curso sin documentos → lista vacía."""
    mock_db.execute.return_value = _exec_result(scalars_all=[])
    mock_db.scalar.return_value = 0

    response = await client.get("/api/v1/documents?course_id=1")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_list_documents_includes_timestamps(client, mock_db):
    """Lista retorna created_at y updated_at (CONT-05)."""
    doc1 = _make_doc(status="indexed")
    doc2 = _make_doc(status="error", error_message="embedding falló")
    mock_db.execute.return_value = _exec_result(scalars_all=[doc1, doc2])
    mock_db.scalar.return_value = 2

    response = await client.get("/api/v1/documents?course_id=1")

    assert response.status_code == 200
    body = response.json()
    items = body["items"]
    assert len(items) == 2
    assert body["total"] == 2
    # CONT-05: campos de timestamps presentes
    assert "created_at" in items[0]
    assert "updated_at" in items[0]
    assert items[0]["created_at"] is not None
    assert items[1]["status"] == "error"
    assert items[1]["error_message"] == "embedding falló"


async def test_list_documents_total_reflects_course_count_not_page_size(client, mock_db):
    """UX-17 (#387): total es la cantidad real de documentos del curso, no
    la cantidad de items ya recortada por limit — así el frontend sabe si
    hay más para paginar."""
    page = [_make_doc() for _ in range(2)]
    mock_db.execute.return_value = _exec_result(scalars_all=page)
    mock_db.scalar.return_value = 37  # el curso tiene 37 documentos en total

    response = await client.get("/api/v1/documents?course_id=1&limit=2&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["total"] == 37


async def test_list_documents_without_limit_uses_default_cap(client, mock_db):
    """Sin limit/offset (caso ExamGeneratorPanel.jsx), sigue funcionando
    igual que antes — trae todo hasta el tope por default, no una página chica."""
    docs = [_make_doc() for _ in range(5)]
    mock_db.execute.return_value = _exec_result(scalars_all=docs)
    mock_db.scalar.return_value = 5

    response = await client.get("/api/v1/documents?course_id=1")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 5
    assert body["total"] == 5


# ============================================================
# GET /api/v1/documents/{id} — estado individual
# ============================================================

async def test_get_document_status_found(client, mock_db):
    doc = _make_doc(status="indexing")
    mock_db.execute.return_value = _exec_result(scalar=doc)

    response = await client.get(f"/api/v1/documents/{doc.id}")

    assert response.status_code == 200
    assert response.json()["status"] == "indexing"


async def test_get_document_status_not_found(client, mock_db):
    mock_db.execute.return_value = _exec_result(scalar=None)

    response = await client.get(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404


# ============================================================
# GET /api/v1/documents/{id}/preview — CONT-08 (#357)
# ============================================================

async def test_document_preview_returns_extracted_text(client, mock_db):
    doc = _make_doc(status="indexed")
    mock_db.execute.return_value = _exec_result(scalar=doc)
    mock_db.scalar.return_value = "  Este es el texto extraído del PDF.  "

    response = await client.get(f"/api/v1/documents/{doc.id}/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["preview"] == "Este es el texto extraído del PDF."
    assert body["char_count"] == len("Este es el texto extraído del PDF.")
    assert body["truncated"] is False
    assert body["status"] == "indexed"


async def test_document_preview_truncates_long_text(client, mock_db):
    doc = _make_doc(status="indexed")
    mock_db.execute.return_value = _exec_result(scalar=doc)
    mock_db.scalar.return_value = "x" * 5000

    response = await client.get(f"/api/v1/documents/{doc.id}/preview")

    body = response.json()
    assert body["char_count"] == 600
    assert body["truncated"] is True


async def test_document_preview_no_chunks_yet(client, mock_db):
    """Documento sin chunks todavía (indexando o error) → preview None, sin 500."""
    doc = _make_doc(status="indexing")
    mock_db.execute.return_value = _exec_result(scalar=doc)
    mock_db.scalar.return_value = None

    response = await client.get(f"/api/v1/documents/{doc.id}/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["preview"] is None
    assert body["char_count"] == 0


async def test_document_preview_not_found(client, mock_db):
    mock_db.execute.return_value = _exec_result(scalar=None)

    response = await client.get(f"/api/v1/documents/{uuid4()}/preview")

    assert response.status_code == 404


# ============================================================
# DELETE /api/v1/documents/{id}
# ============================================================

async def test_delete_document_returns_204(client, mock_db):
    doc = _make_doc(status="indexed")
    mock_db.execute.return_value = _exec_result(scalar=doc)

    response = await client.delete(f"/api/v1/documents/{doc.id}")

    assert response.status_code == 204
    mock_db.delete.assert_called_once_with(doc)
    mock_db.commit.assert_called_once()


async def test_delete_document_not_found(client, mock_db):
    mock_db.execute.return_value = _exec_result(scalar=None)

    response = await client.delete(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404
    mock_db.delete.assert_not_called()
