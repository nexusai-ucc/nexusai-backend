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

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth.hmac import verify_hmac

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
