"""
TranscriptionProvider — voz a texto para el chat (VOICE-01, issue #314).

Groq expone un endpoint de transcripción (`whisper-large-v3`) compatible con
el SDK de OpenAI, mismo SDK que ya es dependencia de este proyecto (ver
app/providers/llm.py) — solo cambia `base_url`. A diferencia de
`LLMProvider`, no hay retry/fallback: es una llamada simple, sin la
complejidad de una cadena de proveedores.

Opcional: si `GROQ_API_KEY` no está seteada, `get_transcription_provider()`
devuelve `None` en vez de instanciar un client roto — el router de voz
(app/voice/router.py) responde 503 explícito antes de intentar nada.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class TranscriptionProvider:
    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model: str = model or settings.groq_stt_model
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=_GROQ_BASE_URL,
            timeout=30.0,
            max_retries=0,
        )

    async def transcribe(
        self, audio_bytes: bytes, filename: str, mime_type: str, language: str = "es"
    ) -> str:
        """Transcribe un audio corto (pregunta hablada) a texto.

        `file` se pasa como tupla (nombre, bytes, mime_type) — formato que
        acepta el SDK de OpenAI para uploads en memoria, sin tocar disco.
        """
        result = await self.client.audio.transcriptions.create(
            model=self.model,
            file=(filename, io.BytesIO(audio_bytes), mime_type),
            language=language,
        )
        return result.text.strip()


@lru_cache(maxsize=1)
def _cached_provider() -> Optional[TranscriptionProvider]:
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    return TranscriptionProvider(api_key=settings.groq_api_key)


def get_transcription_provider() -> Optional[TranscriptionProvider]:
    """FastAPI Dependency. `None` si no hay `GROQ_API_KEY` configurada."""
    return _cached_provider()
