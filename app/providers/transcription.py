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
import time
from functools import lru_cache
from typing import Any, Optional

from openai import AsyncOpenAI

from app.shared.config import get_settings
from app.shared.usage_ledger import UsageRecord, record_usage, status_for_exception

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
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: str,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe un audio corto (pregunta hablada) a texto.

        `file` se pasa como tupla (nombre, bytes, mime_type) — formato que
        acepta el SDK de OpenAI para uploads en memoria, sin tocar disco.

        Sin `language` el modelo detecta el idioma hablado; forzar uno rompe las
        preguntas dichas en otro.

        `verbose_json` devuelve además la duración del audio, que es lo que
        cobra Groq por transcripción (registro de consumo, COST-01).
        """
        file = (filename, io.BytesIO(audio_bytes), mime_type)
        kwargs: dict[str, Any] = {"model": self.model, "file": file}
        if language:
            kwargs["language"] = language
        started = time.perf_counter()
        try:
            result = await self.client.audio.transcriptions.create(
                response_format="verbose_json", **kwargs
            )
        except Exception as exc:
            await record_usage(
                UsageRecord(
                    kind="transcription",
                    status=status_for_exception(exc),
                    provider="groq",
                    model=self.model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            )
            raise
        duration = getattr(result, "duration", None)
        await record_usage(
            UsageRecord(
                kind="transcription",
                provider="groq",
                model=self.model,
                audio_seconds=(
                    float(duration) if isinstance(duration, (int, float)) else None
                ),
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
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
