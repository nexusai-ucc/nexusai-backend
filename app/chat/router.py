"""
Chat endpoints.

En esta etapa (Sprint 1) solo tiene `/echo` — un endpoint mock que verifica
el HMAC del cliente y devuelve eco de la pregunta. Sirve para que Marcos pueda
probar el flujo end-to-end Moodle → PHP → FastAPI → respuesta sin esperar a
que estén listos los providers de LLM y el pipeline RAG.

Sprint 2 va a agregar:
  - POST /messages    →  chat real con RAG + LLM (Gemini/OpenAI)
  - GET  /sessions    →  historial de conversaciones del usuario
  - DELETE /sessions/{id} → borrar conversación
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.auth.hmac import verify_hmac
from app.chat.schemas import ChatRequest, ChatResponse, MessageOut
from app.db.models import ChatSession, Message
from app.db.session import get_db
from app.documents.retriever import format_context_for_prompt, retrieve_context
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider

router = APIRouter()


class EchoRequest(BaseModel):
    """Payload que envía el plugin Moodle (firmado con HMAC)."""

    question: str = Field(..., min_length=1, max_length=2000)
    course_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)


class EchoResponse(BaseModel):
    """Respuesta mock — devuelve los datos como llegaron, validados."""

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

    session = ChatSession(user_id=payload.user_id, course_id=payload.course_id)
    db.add(session)
    await db.flush()
    return session


def _build_system_prompt(retrieved_context: str) -> str:
    """Arma el system prompt con (o sin) contexto del material del curso.

    Si hay chunks relevantes, los inyecta y le pide al LLM que cite la fuente.
    Si NO hay (curso sin material indexado, o pregunta no relacionada con el
    material), le decimos al LLM que sea honesto: "no tengo esa info en el
    material" en lugar de inventar una respuesta.
    """
    base_instructions = (
        "Sos un asistente académico de NexusAI. Respondé en el mismo idioma "
        "que el alumno (español o inglés)."
    )

    if retrieved_context:
        return (
            base_instructions
            + "\n\n"
            + "Tenés acceso a fragmentos del material del curso del alumno. "
            "Usá esos fragmentos como tu fuente principal de información. "
            "Cuando los uses, citá explícitamente el archivo del que vienen "
            '(ej: "según apunte-derivadas.pdf..."). '
            "Si la pregunta NO se puede responder con los fragmentos disponibles, "
            "decilo explícitamente — no inventes."
            + "\n\n--- MATERIAL DEL CURSO ---\n\n"
            + retrieved_context
            + "\n\n--- FIN DEL MATERIAL ---"
        )

    return (
        base_instructions
        + "\n\n"
        + "El curso del alumno todavía no tiene material indexado. "
        "Si la pregunta requiere conocimiento específico del curso, decile "
        "que su docente todavía no subió el material y que pueden contactarlo. "
        "Para preguntas generales (saludo, ayuda sobre cómo usar el asistente, "
        "definiciones de conceptos amplios), respondé normalmente."
    )


@router.post("/messages", response_model=ChatResponse)
async def messages(
    payload: ChatRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> ChatResponse:
    session = await _get_or_create_session(db, payload)

    user_message = Message(session_id=session.id, role="user", content=payload.question)
    db.add(user_message)
    await db.flush()
    await db.commit()

    # ----- RETRIEVAL RAG -----
    # Buscar los top-5 chunks del material del curso que matcheen con la pregunta.
    # Si falla (Redis caído, embeddings cuota, etc.), seguimos sin contexto en
    # lugar de tirar 503: el chat sigue siendo útil, solo pierde la calidad RAG.
    try:
        retrieved_chunks = await retrieve_context(
            question=payload.question,
            course_id=payload.course_id,
            db=db,
            embeddings=embeddings,
            top_k=5,
            min_similarity=0.3,
        )
        context_text = format_context_for_prompt(retrieved_chunks)
    except Exception as exc:
        import traceback
        print(f"[NexusAI] Retrieval failed (continuing without context): {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        retrieved_chunks = []
        context_text = ""

    # ----- HISTORIAL Y PROMPT -----
    history_result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(desc(Message.created_at), desc(Message.id))
        .limit(10)
    )
    recent_messages = list(reversed(history_result.scalars().all()))

    llm_messages = [
        {"role": "system", "content": _build_system_prompt(context_text)},
    ]

    for message in recent_messages:
        if message.id == user_message.id:
            continue
        llm_messages.append({"role": message.role, "content": message.content})

    llm_messages.append({"role": "user", "content": payload.question})

    try:
        answer = await llm.chat_completion(llm_messages)
    except Exception as exc:
        # Log el error real para debugging — sin esto, el 503 oculta la causa.
        import traceback
        print(f"[NexusAI] LLM call failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El asistente no está disponible",
        ) from exc

    assistant_message = Message(session_id=session.id, role="assistant", content=answer)
    db.add(assistant_message)
    await db.commit()

    messages_result = await db.execute(
        select(Message)
        .where(Message.session_id == session.id)
        .order_by(Message.created_at, Message.id)
    )
    session_messages = messages_result.scalars().all()

    return ChatResponse(
        session_id=session.id,
        answer=answer,
        messages=[MessageOut.model_validate(message) for message in session_messages],
    )


@router.post("/echo", response_model=EchoResponse)
async def echo(
    payload: EchoRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
) -> EchoResponse:
    """
    Endpoint mock que valida HMAC y devuelve eco.

    Útil para:
      - Smoke test del HMAC desde el cliente PHP de Moodle.
      - Probar la cadena completa React → Moodle PHP → FastAPI sin LLM.
      - Diagnosticar drift de clock entre el server PHP y el server Python
        (si tira "Request expired", hay un problema de NTP).

    El parámetro `_body` no se usa — está acá solo para activar el
    Depends(verify_hmac). Si la firma no valida, FastAPI corta la request
    con 401 antes de llegar a esta función.
    """
    return EchoResponse(
        echo=payload.question,
        course_id=payload.course_id,
        user_id=payload.user_id,
    )
