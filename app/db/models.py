import uuid
from datetime import datetime
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Computed, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.shared.config import get_settings


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_course_id", "course_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    uploader_id: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    storage_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document",
        lazy="selectin",
        order_by="Chunk.chunk_index",
        cascade="all, delete-orphan",  # marca hijos como "deleted" cuando el padre se borra
        passive_deletes=True,          # no emite SQL para los hijos; confía en ON DELETE CASCADE
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
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
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
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
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
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Token counts — solo mensajes role='assistant' los populan (migración 003).
    # NULL en mensajes de usuario y en mensajes anteriores a la migración.
    token_count_prompt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_count_completion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
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
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
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
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    question_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    has_relevant_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
        Index("ix_unanswered_questions_course_id_created_at", "course_id", "created_at"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class QuizAttempt(Base):
    """Intento completado de quiz (SP-09 — historial por alumno).

    Registra cada sesión de quiz que el alumno finaliza: tipo de pregunta,
    dificultad, puntaje y fecha. Permite mostrar el historial de práctica
    y detectar tendencias de estudio a lo largo del tiempo.
    """
    __tablename__ = "quiz_attempts"
    __table_args__ = (
        Index("ix_quiz_attempts_user_id_course_id_created_at", "user_id", "course_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    question_type: Mapped[str] = mapped_column(String(20), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(10), nullable=False, default="medium")
    topic: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class QuizError(Base):
    """Pregunta de quiz que el alumno respondió mal (SP-10 — repaso basado en errores).

    Antes esto vivía solo en localStorage del navegador (efímero, no
    cross-device, se perdía al superar el tope de 100). Persistir server-side
    habilita historial real y agregación por tema para sugerir qué repasar.
    """
    __tablename__ = "quiz_errors"
    __table_args__ = (
        Index("ix_quiz_errors_user_id_course_id_created_at", "user_id", "course_id", "created_at"),
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