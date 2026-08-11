"""
Gaps router — Feature G.

Endpoints para que el docente consulte qué preguntas de los alumnos NO
pudo responder el material indexado del curso. Es el feedback loop
pedagógico: "estos temas no están cubiertos por tu material".

Agrupamiento semántico (DOC-D06, issue #313): dos preguntas que dicen lo
mismo con palabras distintas ("¿qué es overfitting?" vs "¿cuándo un modelo
memoriza en vez de generalizar?") antes aparecían como gaps separados,
porque solo se agrupaba por texto normalizado (lower+trim). Ahora se hace
en dos pasadas sobre las filas crudas del curso:

  1. Dedup exacto por texto normalizado (igual que antes — barato y siempre
     correcto para literales idénticos, se hace primero para no gastar
     comparaciones de embedding en algo que el texto ya resuelve).
  2. Clustering semántico de esos grupos entre sí por similitud coseno,
     usando el embedding ya guardado en cada fila (ver
     UnansweredQuestion.embedding, poblado en app/gaps/recorder.py).

Filas sin embedding (creadas antes de esta feature, o con un embed que
falló) participan igual del paso 1, pero no se agregan a otros grupos por
similitud semántica en el paso 2 — quedan como su propio grupo, mismo
comportamiento que el código viejo tenía para todas las filas.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.db.models import UnansweredQuestion
from app.db.session import get_db

router = APIRouter()

# Mismo umbral que forums/router.py usa para detectar posts duplicados por
# similitud coseno (_DEFAULT_SIMILARITY_THRESHOLD) — mismo tipo de problema
# (texto corto redactado por un usuario, ¿es esto "lo mismo" que aquello?),
# reusamos el valor ya validado por el equipo en vez de inventar uno nuevo
# sin dataset real para calibrar.
SEMANTIC_GAP_SIMILARITY_THRESHOLD = 0.75

# Tope de filas a traer por consulta — el clustering es O(n²) en el peor
# caso (compara cada grupo nuevo contra los grupos ya armados). Para el
# volumen esperado de gaps por curso (decenas/cientos, no miles) esto sobra
# de margen sin necesitar una estructura de índice más compleja.
_MAX_ROWS_TO_CLUSTER = 2000


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class GapsListRequest(BaseModel):
    course_id: int = Field(gt=0)
    days: int = Field(default=30, ge=1, le=365)
    limit: int = Field(default=20, ge=1, le=100)
    # UX-15 (#385): paginación — offset sobre los grupos ya clusterizados,
    # no sobre las filas crudas (el clustering agrupa filas de distintas
    # posiciones bajo un mismo gap, así que paginar antes de clusterizar
    # daría resultados inconsistentes entre páginas).
    offset: int = Field(default=0, ge=0)
    # DOC-D08 (#383): por default solo gaps activos. True trae también los
    # archivados, cada uno marcado con is_archived en la respuesta.
    include_archived: bool = False


class GapItem(BaseModel):
    question: str
    count: int
    last_asked_at: datetime
    avg_similarity: Optional[float] = None
    # IDs reales de unanswered_questions detrás de este grupo (puede ser
    # varias filas fusionadas por texto exacto o por similitud semántica) —
    # es la clave que el docente manda de vuelta para archivar/desarchivar
    # este gap puntual, ya que el "question" mostrado es solo el
    # representante más reciente del cluster, no una clave estable.
    question_ids: List[uuid.UUID] = Field(default_factory=list)
    # True solo si TODAS las filas del grupo están archivadas. Si alguien
    # vuelve a preguntar algo equivalente después de archivar, esa fila
    # nueva entra sin archivar y el grupo pasa a False otra vez.
    is_archived: bool = False


class GapsListResponse(BaseModel):
    course_id: int
    days: int
    total: int
    items: List[GapItem]


class GapsArchiveRequest(BaseModel):
    course_id: int = Field(gt=0)
    question_ids: List[uuid.UUID] = Field(min_length=1)
    archived: bool


class GapsArchiveResponse(BaseModel):
    course_id: int
    archived: bool
    affected: int


class _TextGroup:
    """Un grupo de gaps — arranca como dedup exacto por texto normalizado,
    después el clustering semántico puede fusionar varios de estos entre sí."""

    __slots__ = (
        "representative_question", "count", "last_asked_at", "_sim_values",
        "embedding", "row_ids", "_all_archived",
    )

    def __init__(
        self,
        row_id: uuid.UUID,
        question: str,
        created_at: datetime,
        max_similarity: Optional[float],
        embedding: Optional[List[float]],
        archived_at: Optional[datetime],
    ) -> None:
        self.representative_question = question
        self.count = 1
        self.last_asked_at = created_at
        self._sim_values: List[float] = [max_similarity] if max_similarity is not None else []
        self.embedding = embedding
        self.row_ids: List[uuid.UUID] = [row_id]
        self._all_archived = archived_at is not None

    def add_row(
        self,
        row_id: uuid.UUID,
        question: str,
        created_at: datetime,
        max_similarity: Optional[float],
        embedding: Optional[List[float]],
        archived_at: Optional[datetime],
    ) -> None:
        self.count += 1
        if created_at > self.last_asked_at:
            self.last_asked_at = created_at
            self.representative_question = question
        if max_similarity is not None:
            self._sim_values.append(max_similarity)
        if self.embedding is None and embedding is not None:
            self.embedding = embedding
        self.row_ids.append(row_id)
        self._all_archived = self._all_archived and archived_at is not None

    def merge(self, other: "_TextGroup") -> None:
        self.count += other.count
        if other.last_asked_at > self.last_asked_at:
            self.last_asked_at = other.last_asked_at
            self.representative_question = other.representative_question
        self._sim_values.extend(other._sim_values)
        if self.embedding is None and other.embedding is not None:
            self.embedding = other.embedding
        self.row_ids.extend(other.row_ids)
        self._all_archived = self._all_archived and other._all_archived

    @property
    def avg_similarity(self) -> Optional[float]:
        if not self._sim_values:
            return None
        return sum(self._sim_values) / len(self._sim_values)

    @property
    def is_archived(self) -> bool:
        return self._all_archived


def _cluster_gaps(
    rows: List[tuple],
    *,
    similarity_threshold: float = SEMANTIC_GAP_SIMILARITY_THRESHOLD,
) -> List[GapItem]:
    """Agrupa filas crudas de gaps en dos pasadas — ver docstring del módulo.

    `rows`: iterable de tuplas
    (id, question, created_at, max_similarity, embedding, archived_at).
    Función pura (sin DB) a propósito, para poder testear el clustering en
    aislamiento sin mockear una sesión de SQLAlchemy.
    """
    # Pasada 1 — dedup exacto por texto normalizado.
    text_groups: "dict[str, _TextGroup]" = {}
    order: List[str] = []
    for row_id, question, created_at, max_similarity, embedding, archived_at in rows:
        norm = question.strip().lower()
        if norm in text_groups:
            text_groups[norm].add_row(row_id, question, created_at, max_similarity, embedding, archived_at)
        else:
            text_groups[norm] = _TextGroup(row_id, question, created_at, max_similarity, embedding, archived_at)
            order.append(norm)

    # Pasada 2 — clustering semántico de esos grupos entre sí. Filas sin
    # embedding no se fusionan con nada más (quedan como su propio grupo,
    # mismo comportamiento que antes de esta feature para esos casos).
    merged: List[_TextGroup] = []
    for norm in order:
        group = text_groups[norm]
        best_match: Optional[_TextGroup] = None
        if group.embedding is not None:
            best_sim = -1.0
            for existing in merged:
                if existing.embedding is None:
                    continue
                sim = _cosine_similarity(group.embedding, existing.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_match = existing
            if best_match is not None and best_sim < similarity_threshold:
                best_match = None

        if best_match is not None:
            best_match.merge(group)
        else:
            merged.append(group)

    merged.sort(key=lambda g: (g.count, g.last_asked_at), reverse=True)

    return [
        GapItem(
            question=g.representative_question,
            count=g.count,
            last_asked_at=g.last_asked_at,
            avg_similarity=g.avg_similarity,
            question_ids=g.row_ids,
            is_archived=g.is_archived,
        )
        for g in merged
    ]


@router.post("/list", response_model=GapsListResponse)
async def gaps_list(
    payload: GapsListRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> GapsListResponse:
    """Devuelve los gaps del curso agrupados por similitud semántica
    (DOC-D06 / issue #313) — ver docstring del módulo para el algoritmo.
    """
    since = datetime.now(timezone.utc) - timedelta(days=payload.days)

    stmt = (
        select(
            UnansweredQuestion.id,
            UnansweredQuestion.question,
            UnansweredQuestion.created_at,
            UnansweredQuestion.max_similarity,
            UnansweredQuestion.embedding,
            UnansweredQuestion.archived_at,
        )
        .where(UnansweredQuestion.course_id == payload.course_id)
        .where(UnansweredQuestion.created_at >= since)
        .order_by(UnansweredQuestion.created_at.desc())
        .limit(_MAX_ROWS_TO_CLUSTER)
    )
    if not payload.include_archived:
        stmt = stmt.where(UnansweredQuestion.archived_at.is_(None))

    result = await db.execute(stmt)
    rows = result.all()

    all_items = _cluster_gaps(rows)
    items = all_items[payload.offset : payload.offset + payload.limit]

    return GapsListResponse(
        course_id=payload.course_id,
        days=payload.days,
        total=len(all_items),
        items=items,
    )


@router.post("/archive", response_model=GapsArchiveResponse)
async def gaps_archive(
    payload: GapsArchiveRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> GapsArchiveResponse:
    """Archiva o desarchiva un gap puntual (DOC-D08, issue #383).

    Opera sobre `question_ids` — los IDs reales de las filas que el
    clustering de `/list` fusionó bajo un mismo gap mostrado — no sobre el
    texto de la pregunta, que solo es el representante del cluster y no una
    clave estable (ver docstring del módulo).
    """
    new_value = datetime.now(timezone.utc) if payload.archived else None
    stmt = (
        update(UnansweredQuestion)
        .where(UnansweredQuestion.course_id == payload.course_id)
        .where(UnansweredQuestion.id.in_(payload.question_ids))
        .values(archived_at=new_value)
    )
    result = await db.execute(stmt)
    await db.commit()

    return GapsArchiveResponse(
        course_id=payload.course_id,
        archived=payload.archived,
        affected=result.rowcount,
    )
