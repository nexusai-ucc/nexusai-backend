"""
Tests del gaps router — agrupamiento semántico de vacíos de contenido
(DOC-D06, issue #313).

_cluster_gaps es una función pura (sin DB) a propósito — se testea
directamente sin mockear una sesión de SQLAlchemy. El endpoint se testea
aparte, con el mismo patrón de aislamiento que el resto de los routers
(mini FastAPI, verify_hmac/get_db reemplazados con mocks).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth.hmac import verify_hmac
from app.db.session import get_db
from app.gaps.router import SEMANTIC_GAP_SIMILARITY_THRESHOLD, _cluster_gaps, router

_NOW = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)


def _row(question: str, minutes_ago: int, embedding=None, max_similarity=None, row_id=None, archived_at=None):
    return (
        row_id or uuid.uuid4(),
        question,
        _NOW - timedelta(minutes=minutes_ago),
        max_similarity,
        embedding,
        archived_at,
    )


# ============================================================
# _cluster_gaps — algoritmo puro
# ============================================================

def test_cluster_gaps_merges_semantically_similar_questions_with_different_wording():
    """Caso central de la issue: sinónimos/reformulación se agrupan como
    un solo gap aunque el texto sea completamente distinto."""
    # Embeddings casi idénticos (coseno ~0.999) — misma idea, dos redacciones.
    emb_a = [1.0, 0.05, 0.0]
    emb_b = [0.99, 0.1, 0.0]

    rows = [
        _row("¿qué es overfitting?", minutes_ago=10, embedding=emb_a),
        _row("¿cuándo un modelo memoriza en vez de generalizar?", minutes_ago=5, embedding=emb_b),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 1
    assert items[0].count == 2
    # El representante es la redacción más reciente.
    assert items[0].question == "¿cuándo un modelo memoriza en vez de generalizar?"


def test_cluster_gaps_does_not_over_group_dissimilar_questions():
    """No sobre-agrupar: preguntas genuinamente distintas quedan separadas."""
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]  # ortogonal — similitud coseno 0

    rows = [
        _row("¿qué es overfitting?", minutes_ago=10, embedding=emb_a),
        _row("¿cómo se calcula una derivada parcial?", minutes_ago=5, embedding=emb_b),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 2
    assert {i.question for i in items} == {
        "¿qué es overfitting?",
        "¿cómo se calcula una derivada parcial?",
    }


def test_cluster_gaps_exact_text_dedup_still_works():
    """Mismo texto exacto (con distinto casing/espacios) sigue agrupando,
    aunque no haya embeddings — comportamiento previo a DOC-D06 preservado."""
    rows = [
        _row("¿Qué es Overfitting?  ", minutes_ago=10, embedding=None),
        _row("¿qué es overfitting?", minutes_ago=5, embedding=None),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 1
    assert items[0].count == 2


def test_cluster_gaps_rows_without_embedding_dont_merge_semantically():
    """Sin embedding no hay forma de saber si dos preguntas son 'lo mismo'
    semánticamente — deben quedar separadas en vez de asumir que sí."""
    rows = [
        _row("pregunta uno", minutes_ago=10, embedding=None),
        _row("pregunta dos", minutes_ago=5, embedding=None),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 2


def test_cluster_gaps_respects_similarity_threshold():
    """Similitud justo debajo del umbral no fusiona."""
    emb_a = [1.0, 0.0]
    # Vector con coseno ~0.6 respecto a emb_a (< 0.75 default)
    emb_b = [0.6, 0.8]

    rows = [
        _row("pregunta A", minutes_ago=10, embedding=emb_a),
        _row("pregunta B", minutes_ago=5, embedding=emb_b),
    ]

    items = _cluster_gaps(rows, similarity_threshold=SEMANTIC_GAP_SIMILARITY_THRESHOLD)

    assert len(items) == 2


def test_cluster_gaps_sorts_by_count_desc_then_recency():
    emb = [1.0, 0.0]
    rows = [
        _row("pregunta poco frecuente", minutes_ago=1, embedding=[0.0, 1.0]),
        _row("pregunta frecuente v1", minutes_ago=20, embedding=emb),
        _row("pregunta frecuente v2", minutes_ago=10, embedding=[0.99, 0.14]),
    ]

    items = _cluster_gaps(rows)

    assert items[0].count == 2  # la frecuente va primero
    assert items[1].question == "pregunta poco frecuente"


def test_cluster_gaps_averages_similarity_across_merged_rows():
    emb = [1.0, 0.0]
    rows = [
        _row("pregunta A", minutes_ago=10, embedding=emb, max_similarity=0.2),
        _row("pregunta A reformulada", minutes_ago=5, embedding=[0.99, 0.14], max_similarity=0.4),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 1
    assert items[0].avg_similarity == pytest.approx(0.3)


def test_cluster_gaps_exposes_underlying_row_ids():
    """DOC-D08 (#383): el gap mostrado debe traer los IDs reales de las
    filas fusionadas — es la clave que el docente manda para archivar,
    ya que el texto representante no es estable entre clusters."""
    id_a, id_b = uuid.uuid4(), uuid.uuid4()
    emb_a = [1.0, 0.05, 0.0]
    emb_b = [0.99, 0.1, 0.0]
    rows = [
        _row("¿qué es overfitting?", minutes_ago=10, embedding=emb_a, row_id=id_a),
        _row("¿cuándo un modelo memoriza en vez de generalizar?", minutes_ago=5, embedding=emb_b, row_id=id_b),
    ]

    items = _cluster_gaps(rows)

    assert len(items) == 1
    assert set(items[0].question_ids) == {id_a, id_b}


def test_cluster_gaps_is_archived_true_only_when_all_rows_archived():
    """Un grupo queda marcado is_archived solo si TODAS sus filas están
    archivadas. Si una pregunta nueva y sin archivar se fusiona con un
    grupo previamente archivado, el gap vuelve a considerarse activo."""
    emb_a = [1.0, 0.0]
    emb_b = [0.99, 0.14]

    fully_archived = _cluster_gaps([
        _row("pregunta archivada", minutes_ago=10, embedding=emb_a, archived_at=_NOW),
    ])
    assert fully_archived[0].is_archived is True

    mixed = _cluster_gaps([
        _row("pregunta archivada", minutes_ago=10, embedding=emb_a, archived_at=_NOW),
        _row("pregunta archivada reformulada", minutes_ago=1, embedding=emb_b, archived_at=None),
    ])
    assert mixed[0].is_archived is False


# ============================================================
# Endpoint /list — end-to-end con DB mockeada
# ============================================================

@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
async def client(mock_db):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/gaps")
    app.dependency_overrides[verify_hmac] = lambda: b"test-body"
    app.dependency_overrides[get_db] = lambda: mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_gaps_list_merges_synonyms_end_to_end(client, mock_db):
    rows = [
        _row("¿qué es overfitting?", minutes_ago=10, embedding=[1.0, 0.05, 0.0]),
        _row("¿cuándo un modelo memoriza en vez de generalizar?", minutes_ago=5, embedding=[0.99, 0.1, 0.0]),
    ]
    db_result = MagicMock()
    db_result.all.return_value = rows
    mock_db.execute = AsyncMock(return_value=db_result)

    resp = await client.post("/api/v1/gaps/list", json={"course_id": 1, "days": 30, "limit": 20})

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["count"] == 2


@pytest.mark.asyncio
async def test_gaps_list_applies_limit_after_clustering(client, mock_db):
    rows = [
        _row(f"pregunta distinta {i}", minutes_ago=i, embedding=[float(i), 0.0])
        for i in range(5)
    ]
    db_result = MagicMock()
    db_result.all.return_value = rows
    mock_db.execute = AsyncMock(return_value=db_result)

    resp = await client.post("/api/v1/gaps/list", json={"course_id": 1, "days": 30, "limit": 2})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 2
