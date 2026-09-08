"""
Voz — transcripción de audio a texto para el chat (VOICE-01, issue #314).

  POST /api/v1/voice/transcribe
    Recibe un audio corto (pregunta hablada, grabada con MediaRecorder en el
    navegador) en base64, lo transcribe con Groq Whisper, y devuelve el texto
    plano. El alumno lo ve en el composer del chat para confirmar/editar
    antes de enviarlo — este endpoint NUNCA dispara un mensaje de chat, solo
    transcribe.

Mismo patrón base64+JSON que app/documents/router.py (sin multipart/
UploadFile, consistente con el resto del plugin). Sin GROQ_API_KEY
configurada, devuelve 503 explícito en vez de fallar al intentar transcribir.
"""

from __future__ import annotations

import base64
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth.hmac import verify_hmac
from app.providers.transcription import TranscriptionProvider, get_transcription_provider

router = APIRouter()

_logger = logging.getLogger(__name__)

# Preguntas habladas cortas — de sobra con 10 MB (varios minutos de audio
# comprimido). Mucho más chico que el límite de documentos (20 MB).
MAX_AUDIO_BYTES = 10 * 1024 * 1024

# Formatos que MediaRecorder produce típicamente según navegador/SO.
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-m4a",
}


class TranscribeRequest(BaseModel):
    content_b64: str = Field(min_length=1)
    mime_type: str = Field(min_length=1, max_length=100)
    language: str = Field(default="es", min_length=2, max_length=5)


class TranscribeResponse(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    payload: TranscribeRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    provider: TranscriptionProvider | None = Depends(get_transcription_provider),
) -> TranscribeResponse:
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcripción de voz no configurada",
        )

    if payload.mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Tipo de audio no soportado: {payload.mime_type!r}. "
                f"Tipos aceptados: {sorted(SUPPORTED_AUDIO_MIME_TYPES)}"
            ),
        )

    try:
        audio_bytes = base64.b64decode(payload.content_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 content: {exc}",
        )

    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio is empty")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio too large: {len(audio_bytes)} bytes (max {MAX_AUDIO_BYTES})",
        )

    ext = payload.mime_type.split("/")[-1]
    try:
        text = await provider.transcribe(
            audio_bytes, filename=f"question.{ext}", mime_type=payload.mime_type, language=payload.language
        )
    except Exception as exc:
        _logger.warning("VOICE-01: fallo al transcribir audio", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo transcribir el audio",
        ) from exc

    return TranscribeResponse(text=text)
