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
import binascii
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from openai import BadRequestError
from pydantic import BaseModel, Field

from app.auth.hmac import verify_hmac
from app.providers.transcription import (
    TranscriptionProvider,
    get_transcription_provider,
)

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


# Extensión del archivo que recibe Whisper: Groq deduce el formato por la
# extensión, y "x-m4a" (el subtipo tal cual) no es una que acepte.
_EXTENSION_BY_MIME_TYPE = {"audio/x-m4a": "m4a"}


def _normalize_mime_type(mime_type: str) -> str:
    """Tipo base en minúsculas, sin parámetros.

    MediaRecorder informa el codec como parámetro ("audio/webm;codecs=opus" en
    Chrome/Edge, "audio/ogg;codecs=opus" en Firefox): comparar el string
    entero contra SUPPORTED_AUDIO_MIME_TYPES rechazaba toda grabación real.
    """
    return mime_type.split(";", 1)[0].strip().lower()


class TranscribeRequest(BaseModel):
    content_b64: str = Field(min_length=1)
    mime_type: str = Field(min_length=1, max_length=100)
    # None = the model detects the spoken language. The plugin does not send it,
    # and forcing one language breaks questions spoken in any other.
    language: Optional[str] = Field(default=None, min_length=2, max_length=5)


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

    mime_type = _normalize_mime_type(payload.mime_type)
    if mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Tipo de audio no soportado: {payload.mime_type!r}. "
                f"Tipos aceptados: {sorted(SUPPORTED_AUDIO_MIME_TYPES)}"
            ),
        )

    try:
        audio_bytes = base64.b64decode(payload.content_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 content: {exc}",
        )

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Audio is empty"
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio too large: {len(audio_bytes)} bytes (max {MAX_AUDIO_BYTES})",
        )

    ext = _EXTENSION_BY_MIME_TYPE.get(mime_type, mime_type.split("/")[-1])
    try:
        text = await provider.transcribe(
            audio_bytes,
            filename=f"question.{ext}",
            mime_type=mime_type,
            language=payload.language,
        )
    except BadRequestError as exc:
        # Groq rechazó el audio en sí (demasiado corto, ilegible): es un error
        # del pedido, no una caída — un 503 acá sumaría a la alerta de 5xx.
        _logger.info("VOICE-01: Groq rechazó el audio: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El audio no se pudo procesar (¿demasiado corto?)",
        ) from exc
    except Exception as exc:
        _logger.warning("VOICE-01: fallo al transcribir audio", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo transcribir el audio",
        ) from exc

    return TranscribeResponse(text=text)
