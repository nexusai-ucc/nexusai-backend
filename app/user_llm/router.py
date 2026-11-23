"""
Prueba de la API key propia de un usuario (BYOK) antes de guardarla.

El plugin llama a este endpoint desde la pantalla "Mi API key": manda la key en
el cuerpo firmado (`user_llm`, igual que en cualquier otro pedido) y el backend
hace una llamada mínima solo contra ese eslabón. Si el proveedor responde, el
plugin la guarda cifrada; si no, le muestra al usuario por qué (key rechazada,
sin saldo, modelo inexistente...). El backend no guarda la key.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.auth.hmac import verify_hmac
from app.providers.llm import LLMProvider, get_llm_provider
from app.providers.user_llm import user_llm_from_body

router = APIRouter()

# Pedido mínimo: un par de tokens de ida y de vuelta. Sin max_tokens a
# propósito: varios modelos nuevos lo rechazan con 400 (piden
# max_completion_tokens) y eso haría fallar la prueba de una key que anda.
_PROBE_MESSAGES = [
    {"role": "user", "content": "Reply with the single word: ok"},
]


class UserLLMTestRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    course_id: int = Field(default=0, ge=0)


class UserLLMTestResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    latency_ms: float


@router.post("/test", response_model=UserLLMTestResponse)
async def test_user_llm(
    request: Request,
    payload: UserLLMTestRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    llm: LLMProvider = Depends(get_llm_provider),
) -> UserLLMTestResponse:
    """Prueba la key del usuario contra su proveedor. 424 si el proveedor falla."""
    config = user_llm_from_body(_body)
    if config is None or not llm.has_user_llm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Falta user_llm o tiene un formato inválido",
        )
    started = time.perf_counter()
    # Errores de la key → UserLLMError → 424 con el motivo (handler de main.py).
    result = await llm.user_only().chat_completion(_PROBE_MESSAGES)
    return UserLLMTestResponse(
        ok=True,
        provider=config.provider,
        model=result.model,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
