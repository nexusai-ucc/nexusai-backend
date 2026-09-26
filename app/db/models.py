import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.shared.config import get_settings


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_course_id", "course_id"),
        Index("ix_documents_course_id_section", "course_id", "section"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    uploader_id: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # Número de sección/unidad del curso Moodle (BUS-05). Opcional: el docente
    # puede no asignarla al subir material.
    section: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    storage_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document",
        lazy="selectin",
        order_by="Chunk.chunk_index",
        cascade="all, delete-orphan",  # marca hijos como "deleted" cuando el padre se borra
        passive_deletes=True,  # no emite SQL para los hijos; confía en ON DELETE CASCADE
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id_chunk_index", "document_id", "chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    content_tsv: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', content)", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_user_id_course_id", "user_id", "course_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", lazy="selectin", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_session_id_created_at", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Token counts — solo mensajes role='assistant' los populan (migración 003).
    # NULL en mensajes de usuario y en mensajes anteriores a la migración.
    token_count_prompt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_count_completion: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class ForumPostEmbedding(Base):
    """Embedding de un post de foro para detección de duplicados (Épica 06)."""

    __tablename__ = "forum_post_embeddings"
    __table_args__ = (
        Index("ix_forum_post_embeddings_course_id", "course_id"),
        Index("ix_forum_post_embeddings_discussion_id", "discussion_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    forum_post_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    discussion_id: Mapped[int] = mapped_column(Integer, nullable=False)
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class InteractionLog(Base):
    """Registro anonimizado de cada interacción con el asistente (DOC-D01).

    Alimenta el dashboard de analytics para docentes. No almacena contenido
    de mensajes ni user_id directo — solo un hash SHA-256 del user_id para
    poder contar usuarios únicos sin exponer identidad.
    """

    __tablename__ = "interaction_logs"
    __table_args__ = (
        Index("ix_interaction_logs_course_id_created_at", "course_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    question_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    has_relevant_context: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    is_multicourse: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(10), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UnansweredQuestion(Base):
    """Preguntas del alumno que el material del curso no pudo responder bien.

    Se registran cuando el retrieval no devuelve chunks o devuelve chunks con
    similarity baja. El docente consulta esta tabla para descubrir qué temas
    le faltan al material (Feature G — detección de gaps).
    """

    __tablename__ = "unanswered_questions"
    __table_args__ = (
        Index(
            "ix_unanswered_questions_course_id_created_at", "course_id", "created_at"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # Mejor similarity entre los chunks recuperados (0..1). NULL si chunks=0.
    max_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Cantidad de chunks que se llegaron a recuperar (0 = nada matcheó).
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Embedding de `question`, para agrupar gaps por similitud semántica en
    # vez de texto normalizado (DOC-D06, issue #313). Nullable: filas viejas
    # (de antes de esta feature) o casos donde el embed falló no lo tienen —
    # el router cae de vuelta a agrupación por texto para esas filas.
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # NULL = gap activo. El docente lo archiva desde el panel cuando ya lo
    # resolvió (agregó material, o decidió que no es relevante) — DOC-D08,
    # issue #383. Si un alumno vuelve a preguntar algo equivalente después,
    # se inserta una fila nueva con archived_at=NULL, así que el gap
    # resurge solo sin lógica extra (ver app/gaps/router.py).
    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # NULL = activo en el plan de estudio del alumno. SP-13 (#323): distinto
    # de `archived_at` (archivado del DOCENTE, DOC-D08) — descartar un tema
    # del propio plan no debe afectar lo que ve el docente en Gaps/Analytics.
    # Si el alumno vuelve a preguntar algo equivalente, la fila nueva entra
    # sin dismiss y el tema reaparece solo (ver app/quiz/router.py::study_plan).
    student_dismissed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class QuizAttempt(Base):
    """Intento completado de quiz (SP-09 + ANALYTICS-01).

    Registra cada sesión de quiz que el alumno finaliza. Sirve para:
    - SP-09: historial de práctica del alumno (question_type, difficulty, topic).
    - ANALYTICS-01: histograma de puntajes por curso para el dashboard docente (score).
    """

    __tablename__ = "quiz_attempts"
    __table_args__ = (
        Index(
            "ix_quiz_attempts_user_id_course_id_created_at",
            "user_id",
            "course_id",
            "created_at",
        ),
        Index("ix_quiz_attempts_course_id_created_at", "course_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # Nullable: PRIV-01 anonimiza (en vez de borrar) los intentos de un alumno
    # que pide eliminar sus datos, para no romper el histograma de puntajes
    # de Analytics (app/admin/router.py::_get_quiz_score_distribution lee
    # course_id/created_at/score, nunca user_id — el score sobrevive intacto).
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    question_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    difficulty: Mapped[str] = mapped_column(
        String(10), nullable=False, default="medium"
    )
    topic: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_answers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # NULL = intento activo del alumno. Timestamp = fue anonimizado a pedido
    # del alumno (PRIV-01) — user_id ya es NULL, la fila se excluye del
    # export/vista personal pero sigue contando en los agregados del curso.
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class QuizError(Base):
    """Pregunta de quiz que el alumno respondió mal (SP-10 — repaso basado en errores).

    Antes esto vivía solo en localStorage del navegador (efímero, no
    cross-device, se perdía al superar el tope de 100). Persistir server-side
    habilita historial real y agregación por tema para sugerir qué repasar.
    """

    __tablename__ = "quiz_errors"
    __table_args__ = (
        Index(
            "ix_quiz_errors_user_id_course_id_created_at",
            "user_id",
            "course_id",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    question_type: Mapped[str] = mapped_column(String(20), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    source_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Texto suelto, no FK: el pipeline de quiz/generate ya arrastra un bug de
    # tipos (UUID de Document.id tratado como int en el schema de respuesta),
    # así que este valor no es confiable para joins — solo se guarda best-effort.
    source_document_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    options: Mapped[Optional[List[Any]]] = mapped_column(JSONB, nullable=True)
    correct_index: Mapped[int] = mapped_column(Integer, nullable=False, default=-1)
    user_selected_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # NULL = activo en el plan de estudio del alumno (SP-13, #323). El alumno
    # marca "ya lo entendí" desde el Plan de estudio; una fila nueva sobre el
    # mismo tema vuelve a entrar sin dismiss (ver app/quiz/router.py::study_plan).
    dismissed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Flashcard(Base):
    """Flashcard generada por el generador de quiz y persistida para repaso (SP-11, #315).

    Antes las flashcards (question_type='flashcard' en /quiz/generate) eran
    100% efímeras — un lote nuevo por LLM en cada request, sin ID ni tabla
    propia. Para poder aplicar repetición espaciada hace falta identidad
    estable: cada flashcard generada se upsertea acá por (course_id,
    content_hash), así regenerar el mismo contenido no duplica filas.
    """

    __tablename__ = "flashcards"
    __table_args__ = (
        UniqueConstraint(
            "course_id", "content_hash", name="uq_flashcards_course_content_hash"
        ),
        Index("ix_flashcards_course_id", "course_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    source_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Texto suelto, no FK — mismo criterio que QuizError.source_document_id.
    source_document_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FlashcardReview(Base):
    """Estado de repetición espaciada (SM-2) de una flashcard para un alumno (SP-11, #315).

    Fórmula SM-2 simplificada aplicada en app/quiz/router.py — ver comentario
    junto a `_apply_sm2`. NULL en next_review_at = nunca repasada = "toca hoy".
    """

    __tablename__ = "flashcard_reviews"
    __table_args__ = (
        UniqueConstraint(
            "flashcard_id", "user_id", name="uq_flashcard_reviews_flashcard_user"
        ),
        Index(
            "ix_flashcard_reviews_user_id_next_review_at", "user_id", "next_review_at"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flashcard_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("flashcards.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Nullable: PRIV-01 anonimiza (no borra) igual que QuizAttempt.user_id —
    # ver app/privacy/router.py::delete_personal_data.
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ease_factor: Mapped[float] = mapped_column(Float, nullable=False, default=2.5)
    interval_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repetitions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_review_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class MessageFeedback(Base):
    """Voto 👍/👎 del alumno sobre una respuesta del chat (ASIST-01, #321).

    Anónimo por diseño, mismo criterio que InteractionLog: no guarda
    user_id, solo un hash SHA-256 usado únicamente para permitir que el
    alumno cambie de voto (upsert por message_id+user_id_hash) — nunca
    expuesto al docente. `course_id` va denormalizado porque `message_id`
    puede quedar NULL si el alumno borra su historial (PRIV-01 hard-deletea
    messages/chat_sessions) y el agregado del curso debe sobrevivir a eso.
    """

    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint(
            "message_id", "user_id_hash", name="uq_message_feedback_message_user"
        ),
        Index("ix_message_feedback_course_id_created_at", "course_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    is_helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class CalendarAlert(Base):
    """Alerta de evento de calendario configurada por el alumno (CAL-02)."""

    __tablename__ = "calendar_alerts"
    __table_args__ = (
        UniqueConstraint("user_id", "event_id", name="uq_calendar_alerts_user_event"),
        Index("ix_calendar_alerts_user_course", "user_id", "course_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_name: Mapped[str] = mapped_column(String(200), nullable=False)
    event_timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    days_before: Mapped[int] = mapped_column(Integer, nullable=False)
    notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ForumWebhookConfig(Base):
    """URL de webhook (Slack/Discord/Teams) configurada por el docente para
    recibir el digest semanal del foro (FOR-07, #378). Una config por curso."""

    __tablename__ = "forum_webhook_configs"
    __table_args__ = (
        UniqueConstraint("course_id", name="uq_forum_webhook_configs_course"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    webhook_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LlmUsage(Base):
    """Una fila por cada llamada a un proveedor de IA (COST-01, issue #519).

    Cubre LLM, embeddings y transcripción de voz, más los bloqueos de
    moderación y los intentos fallidos de cada eslabón de la cadena de
    fallback. Lo escribe app/shared/usage_ledger.py desde la capa de
    proveedores, así que no depende de que cada endpoint se acuerde de
    registrar su consumo.
    """

    __tablename__ = "llm_usage"
    __table_args__ = (
        Index("ix_llm_usage_created_at", "created_at"),
        Index("ix_llm_usage_course_id_created_at", "course_id", "created_at"),
        Index("ix_llm_usage_feature_created_at", "feature", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # llm | embedding | transcription | moderation
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    # Ruta que originó la llamada, p. ej. "chat.stream" o "quiz.generate".
    feature: Mapped[str] = mapped_column(String(80), nullable=False)
    provider: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # ok | error | quota | blocked
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    embedding_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    audio_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # NULL = modelo sin precio cargado en model_prices.
    cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 8), nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    saved_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # True si el proveedor no informó los tokens y se estimaron con tiktoken.
    estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    course_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # student | teacher | system | unknown
    role: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class ModelPrice(Base):
    """Precio por modelo, en USD, con fecha de vigencia (COST-01, issue #519).

    El costo de cada fila de llm_usage se calcula con el precio vigente al
    momento de la llamada, así que un cambio de precio no reescribe el
    historial. Se carga a mano con scripts/model_prices.py.
    """

    __tablename__ = "model_prices"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model",
            "valid_from",
            name="uq_model_prices_provider_model_from",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    input_per_mtok: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=0
    )
    output_per_mtok: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=0
    )
    # NULL = los tokens cacheados se cobran como entrada normal.
    cached_input_per_mtok: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 6), nullable=True
    )
    embedding_per_mtok: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=0
    )
    audio_per_minute: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, default=0
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LlmUsageDaily(Base):
    """Resumen diario de llm_usage (COST-01, issue #519).

    scripts/rollup_llm_usage.py pasa acá el detalle más viejo que
    USAGE_LEDGER_DETAIL_DAYS y lo borra de llm_usage. Las columnas de la
    clave no admiten NULL (course_id 0, strings vacíos) para que la
    restricción única funcione como clave del upsert.
    """

    __tablename__ = "llm_usage_daily"
    __table_args__ = (
        UniqueConstraint(
            "day",
            "client_id",
            "course_id",
            "role",
            "feature",
            "kind",
            "provider",
            "model",
            "status",
            name="uq_llm_usage_daily_key",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    client_id: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    course_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    role: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    feature: Mapped[str] = mapped_column(String(80), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    provider: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_prompt_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    embedding_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    audio_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 8), nullable=False, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    saved_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class DocumentSummary(Base):
    """Resumen ya generado, guardado hasta que cambie lo que lo originó (COST-02, issue #521).

    Reemplaza a la caché de Redis de 24 h (PERF-02): el resumen de un documento
    se genera una vez y lo leen todos los alumnos, sin volver a pagarlo.

    `cache_key` es un hash de todo lo que define el resultado: el documento y
    la huella de su archivo (o, para el resumen pre-examen, la lista de
    documentos con sus huellas), el modelo configurado y las versiones del
    prompt. Si cambia cualquiera, la clave es otra y la fila vieja deja de
    servirse; la de un documento se borra en cascada con él.
    """

    __tablename__ = "document_summaries"
    __table_args__ = (Index("ix_document_summaries_course_id", "course_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    # document | pre_exam
    kind: Mapped[str] = mapped_column(String(12), nullable=False)
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # NULL para el pre-examen, que combina varios documentos.
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=True,
    )
    prompt_version: Mapped[str] = mapped_column(String(40), nullable=False)
    # Modelo y proveedor que respondieron de verdad (puede ser un fallback).
    model: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Tokens que costó generarlo: es lo que se ahorra en cada acierto.
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
