"""Pydantic schemas that define the contract between the Moodle PHP plugin and the NexusAI backend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.shared.visibility import VisibleCmids


class HistoryItem(BaseModel):
    """Un mensaje previo de la conversación, tal como lo guarda Moodle (DATA-04)."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    session_id: Optional[UUID] = None
    # Feature B — chat multi-curso. Si viene poblada, el retriever busca en
    # todos los cursos de la lista en lugar de solo en course_id.
    course_ids: Optional[List[int]] = None
    # Mapa {str(course_id): nombre} para que el LLM cite la materia en multi-curso.
    course_names: Optional[Dict[str, str]] = None
    # Presupuesto de tokens por rol (app/shared/token_budget.py). Resuelto
    # server-side en el plugin PHP vía has_capability('local/nexusai:manage', ...)
    # — el mismo criterio que ya usa visibility_helper.php — NUNCA se confía en
    # un rol mandado por el JS del navegador. Default False = alumno (límite
    # más conservador) si algún cliente viejo no manda el campo.
    is_teacher: bool = False
    # Actividades del curso que el usuario puede ver, calculadas por el plugin
    # (VIS-01, ver app/shared/visibility.py). Sin la lista el pedido se rechaza.
    visible_cmids: VisibleCmids = None
    # DATA-04 (#524), opción C: la conversación vive en Moodle. Cuando el
    # plugin manda `history` (aunque sea vacía), el backend no guarda la
    # pregunta, la respuesta, las métricas ni el gap: los devuelve en la
    # respuesta (o en el evento `done`) para que Moodle los guarde. Sin
    # `history` se sigue el flujo anterior, con la conversación en el backend.
    history: Optional[List[HistoryItem]] = Field(default=None, max_length=10)
    # Límites de tokens que configuró el admin de Moodle para el rol de este
    # usuario. Sin ellos se usan los de la configuración del backend.
    token_limit_hourly: Optional[int] = Field(default=None, ge=1, le=10_000_000)
    token_limit_daily: Optional[int] = Field(default=None, ge=1, le=100_000_000)

    @property
    def stateless(self) -> bool:
        return self.history is not None


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    # Token counts solo presentes en mensajes role='assistant'. NULL → None.
    token_count_prompt: Optional[int] = None
    token_count_completion: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ChatMetrics(BaseModel):
    """Métricas de una respuesta, para que Moodle guarde la interacción (DATA-04)."""

    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    fallback: bool = False
    chunks_retrieved: int = 0
    has_relevant_context: bool = False
    grounded: bool = False
    multicourse: bool = False


class GapSignal(BaseModel):
    """Si la pregunta no la pudo responder el material (DATA-04).

    En el flujo anterior el backend guardaba el gap; en el nuevo solo decide y
    Moodle lo guarda con este embedding, que sirve para agruparlo.
    """

    is_gap: bool
    max_similarity: Optional[float] = None
    chunks_retrieved: int = 0
    embedding: Optional[List[float]] = None


class ChatResponse(BaseModel):
    # None cuando la conversación vive en Moodle (pedido con `history`).
    session_id: Optional[UUID] = None
    answer: str
    messages: List[MessageOut] = Field(default_factory=list)
    # Tokens consumidos en ESTA respuesta (útiles para monitoreo en tiempo real).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Solo en el flujo nuevo (pedido con `history`), DATA-04.
    metrics: Optional[ChatMetrics] = None
    gap: Optional[GapSignal] = None
    budget: Optional[dict[str, Any]] = None
    usage: Optional[List[dict[str, Any]]] = None
