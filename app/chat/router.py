"""
Chat endpoints — Sprint 2 + Sprint 3 (BACK-11, BACK-12, BACK-13, BACK-14).

Sprint 3 additions:
  - BACK-11: retry con backoff ya está en LLMProvider/EmbeddingProvider.
             Si LLM falla tras retries → 503. Si embeddings falla → chat sin RAG.
  - BACK-12: persiste token counts (prompt + completion) en cada mensaje del LLM.
             Devuelve los tokens en ChatResponse para monitoreo en tiempo real.
  - BACK-13: validación explícita de material indexado en el sistema prompt.
             El retrieval filtra por course_id (aislamiento multi-curso verificado).
  - BACK-14: rate limiting por user_id (20 req/min), logging estructurado JSON
             con request_id + course_id + user_id + tokens + latencia.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.hmac import verify_hmac
from app.chat.schemas import ChatRequest, ChatResponse, MessageOut
from app.db.models import ChatSession, Message
from app.db.session import get_db
from app.documents.retriever import format_context_for_prompt, retrieve_context
from app.infrastructure.redis_client import get_redis
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider
from app.shared.config import get_settings
from app.shared.rate_limit import check_rate_limit

import redis.asyncio as redis_async

router = APIRouter()
logger = logging.getLogger("nexusai.chat")


# ============================================================
# Helpers internos
# ============================================================

class EchoRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    course_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)


class EchoResponse(BaseModel):
    echo: str
    course_id: int
    user_id: int
    note: str = "HMAC verificado correctamente. Esto es un mock — Sprint 2 conecta el LLM real."


async def _get_or_create_session(
    db: AsyncSession,
    payload: ChatRequest,
) -> ChatSession:
    if payload.session_id is not None:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == payload.session_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        return session

    # Feature B: si el chat está en modo multi-curso (course_ids con >1 item),
    # la sesión se crea con course_id=0 (sesión global, no atada a una materia).
    session_course_id = (
        0
        if (payload.course_ids and len(payload.course_ids) > 1)
        else payload.course_id
    )
    session = ChatSession(user_id=payload.user_id, course_id=session_course_id)
    db.add(session)
    await db.flush()
    return session


def _build_system_prompt(retrieved_context: str, is_multicourse: bool = False) -> str:
    """Arma el system prompt con (o sin) contexto del material del curso.

    Si hay chunks relevantes, los inyecta y le pide al LLM que cite la fuente.
    Si NO hay (curso sin material indexado, o pregunta no relacionada), le
    indicamos al LLM que sea honesto en lugar de inventar una respuesta.

    Cuando is_multicourse=True, los fragmentos vienen de varios cursos a la
    vez y el LLM debe citar también la materia.
    """
    base_instructions = (
        "Sos un asistente académico de NexusAI. Respondé en el mismo idioma "
        "que el alumno (español o inglés)."
    )

    if retrieved_context:
        source_label = "de tus materias" if is_multicourse else "del curso del alumno"
        multicourse_hint = (
            "Cuando el fragmento viene de una materia específica, mencionala "
            '(ej: "en Cálculo I, según apunte.pdf..."). '
            if is_multicourse
            else ""
        )
        return (
            base_instructions
            + "\n\n"
            + f"Tenés acceso a fragmentos del material {source_label}. "
            "Usá esos fragmentos como tu fuente principal de información. "
            "Cuando los uses, citá explícitamente el archivo del que vienen "
            '(ej: "según apunte-derivadas.pdf..."). '
            + multicourse_hint
            + "Si la pregunta NO se puede responder con los fragmentos disponibles, "
            "decilo explícitamente — no inventes."
            + "\n\n--- MATERIAL DEL CURSO ---\n\n"
            + retrieved_context
            + "\n\n--- FIN DEL MATERIAL ---"
        )

    return (
        base_instructions
        + "\n\n"
        + "El curso del alumno todavía no tiene material indexado en NexusAI. "
        "Si la pregunta requiere conocimiento específico del curso (contenido de "
        "clases, apuntes, trabajos prácticos), decile al alumno que su docente "
        "todavía no subió el material y que puede contactarlo para pedírselo. "
        "Para preguntas generales (saludo, cómo usar el asistente, conceptos "
        "amplios que no dependen del material del curso), respondé normalmente."
    )


# ============================================================
# Endpoint principal: POST /messages
# ============================================================

@router.post("/messages", response_model=ChatResponse)
async def messages(
    request: Request,
    payload: ChatRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
    redis: redis_async.Redis = Depends(get_redis),
) -> ChatResponse:
    settings = get_settings()
    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", "-")

    # ----- BACK-14: Rate limiting por user_id -----
    await check_rate_limit(
        user_id=payload.user_id,
        redis=redis,
        limit=settings.rate_limit_per_user_minute,
        window_sec=60,
    )

    session = await _get_or_create_session(db, payload)

    user_message = Message(session_id=session.id, role="user", content=payload.question)
    db.add(user_message)
    await db.flush()
    await db.commit()

    # ----- BACK-11 + BACK-13: Retrieval RAG (Feature B: multi-curso opcional) -----
    # Si falla (embeddings caídos, cuota agotada, etc.), continúa SIN contexto.
    # El chat sigue siendo útil aunque pierda calidad RAG.
    is_multicourse = bool(payload.course_ids and len(payload.course_ids) > 1)
    # Parsear course_names: el payload trae {str(id): nombre}, lo pasamos a {int: str}.
    course_names_int: dict[int, str] = {}
    if payload.course_names:
        for k, v in payload.course_names.items():
            try:
                course_names_int[int(k)] = str(v)
            except (ValueError, TypeError):
                pass

    try:
        retrieved_chunks = await retrieve_context(
            question=payload.question,
            course_id=payload.course_id,
            course_ids=payload.course_ids,
            db=db,
            embeddings=embeddings,
            top_k=5,
            min_similarity=0.3,
        )
        context_text = format_context_for_prompt(
            retrieved_chunks,
            course_names=course_names_int if is_multicourse else None,
        )
    except Exception as exc:
        import traceback
        print(
            f"[NexusAI] Retrieval failed (continuing without context): "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        retrieved_chunks = []
        context_text = ""

    # ----- Historial y construcción del prompt -----
    history_result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(desc(Message.created_at), desc(Message.id))
        .limit(10)
    )
    recent_messages = list(reversed(history_result.scalars().all()))

    llm_messages = [
        {"role": "system", "content": _build_system_prompt(context_text, is_multicourse=is_multicourse)},
    ]
    for message in recent_messages:
        if message.id == user_message.id:
            continue
        llm_messages.append({"role": message.role, "content": message.content})
    llm_messages.append({"role": "user", "content": payload.question})

    # ----- BACK-11: Llamada al LLM (con retry interno en LLMProvider) -----
    try:
        result = await llm.chat_completion(llm_messages)
    except Exception as exc:
        import traceback
        print(f"[NexusAI] LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El asistente no está disponible temporalmente",
        ) from exc

    # ----- BACK-12: Persistir mensaje del asistente con token counts -----
    assistant_message = Message(
        session_id=session.id,
        role="assistant",
        content=result.text,
        token_count_prompt=result.prompt_tokens,
        token_count_completion=result.completion_tokens,
    )
    db.add(assistant_message)
    await db.commit()

    messages_result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(Message.created_at, Message.id)
    )
    session_messages = messages_result.scalars().all()

    # ----- BACK-14: Logging estructurado JSON -----
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
    logger.info(
        json.dumps(
            {
                "event": "chat_message",
                "request_id": request_id,
                "course_id": payload.course_id,
                "user_id": payload.user_id,
                "session_id": str(session.id),
                "chunks_retrieved": len(retrieved_chunks),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": latency_ms,
            },
            ensure_ascii=False,
        )
    )

    return ChatResponse(
        session_id=session.id,
        answer=result.text,
        messages=[MessageOut.model_validate(m) for m in session_messages],
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )


# ============================================================
# Endpoint de smoke test: POST /echo
# ============================================================

@router.post("/echo", response_model=EchoResponse)
async def echo(
    payload: EchoRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
) -> EchoResponse:
    """Endpoint mock que valida HMAC y devuelve eco.

    Útil para smoke test del HMAC desde el cliente PHP de Moodle.
    Si la firma no valida, FastAPI corta la request con 401 antes de llegar acá.
    """
    return EchoResponse(
        echo=payload.question,
        course_id=payload.course_id,
        user_id=payload.user_id,
    )
