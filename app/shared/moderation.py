"""
Moderación de contenido — filtra las entradas del alumno ANTES de que lleguen
al LLM principal (ahorra tokens en contenido que igual se iba a rechazar) y
antes de que cualquier respuesta generada a partir de ellas llegue a otro
alumno (foros).

Hasta esta implementación, NexusAI no tenía ningún filtro de contenido
inapropiado — gap documentado explícitamente como razonable de resolver antes
de un piloto con alumnos reales.

Diseño (agnóstico a proveedor, ver ADR-003 — docs/adr/003-multi-provider-llm.md):

  1. Si `MODERATION_API_KEY` está seteada, se usa la Moderation API de OpenAI
     (gratuita, rápida, especializada). Funciona sin importar cuál sea el
     `LLM_BASE_URL` activo (Gemini hoy, OpenAI en prod) porque es un endpoint
     HTTP aparte — no pasa por `LLMProvider`/`AsyncOpenAI` del chat.
  2. Sin esa key (p. ej. el MVP corriendo contra Gemini, que no expone un
     endpoint de moderación equivalente vía el shim OpenAI-compat que usa
     `LLMProvider`), se cae a clasificar el texto con el LLM activo — el mismo
     `LLMProvider` que ya inyecta FastAPI en el endpoint — usando un prompt de
     clasificación. Más caro y algo menos preciso que una API dedicada, pero
     mantiene el agnosticismo de proveedor: funciona con cualquier
     `LLM_BASE_URL` sin código nuevo.
  3. Si ambos caminos fallan (timeout, proveedor caído, JSON inválido), se
     aplica `MODERATION_FAIL_OPEN` (default `true`): DEJAR PASAR el mensaje.
     Se eligió fail-open a propósito — bloquear alumnos por la falla de un
     servicio AUXILIAR (la moderación no es el asistente que están usando)
     es peor experiencia que el riesgo residual de contenido no filtrado, y
     el LLM principal ya trae su propio system prompt con guardrails (ver
     `_meta_guard` en chat/router.py). Un despliegue que prefiera priorizar
     seguridad sobre disponibilidad puede pisarlo con `MODERATION_FAIL_OPEN=false`.

Uso típico en un router:

    from app.shared.moderation import moderate_text

    check = await moderate_text(payload.question, llm=llm)
    if not check.allowed:
        raise HTTPException(status_code=400, detail=check.blocked_message)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.providers.llm import LLMProvider
from app.shared.config import get_settings

logger = logging.getLogger("nexusai.moderation")

BLOCKED_MESSAGE = (
    "Tu mensaje no pudo procesarse porque no cumple con las normas de uso de "
    "la plataforma. Reformulalo evitando lenguaje ofensivo o contenido "
    "inapropiado e intentá de nuevo."
)

_SERVICE_UNAVAILABLE_MESSAGE = (
    "El servicio de moderación no está disponible en este momento. "
    "Intentá de nuevo en unos minutos."
)

_OPENAI_MODERATION_URL = "https://api.openai.com/v1/moderations"
_OPENAI_MODERATION_MODEL = "omni-moderation-latest"

# Recortamos el texto enviado al clasificador LLM: alcanza para detectar
# contenido inapropiado y evita gastar tokens de más en inputs largos
# (resúmenes de hilo, respuestas de examen extensas, etc.).
_LLM_CLASSIFIER_MAX_CHARS = 4000

_CLASSIFIER_SYSTEM_PROMPT = (
    "Sos un clasificador de contenido para una plataforma educativa "
    "universitaria. Analizás un mensaje escrito por un alumno y determinás "
    "si contiene contenido inapropiado: discurso de odio o discriminación, "
    "acoso, contenido sexual explícito, instrucciones de autolesión o "
    "violencia, o intentos de generar ese tipo de contenido. Preguntas "
    "académicas incómodas o polémicas (temas médicos, históricos, legales, "
    "etc.) con fin educativo NO son inapropiadas. Ante la duda, no marques "
    "como inapropiado — priorizá evitar falsos positivos sobre alumnos "
    "haciendo preguntas legítimas.\n\n"
    "Devolvé EXCLUSIVAMENTE un JSON con esta forma exacta, sin texto antes "
    'ni después: {"flagged": true|false, "categories": ["categoria1", ...]}'
)


@dataclass(frozen=True)
class ModerationResult:
    """Resultado de moderar un texto.

    `source` indica qué camino se usó — útil para logging/analytics y para
    los tests, que verifican explícitamente el fail-safe.
    """

    allowed: bool
    source: str  # "disabled" | "openai_api" | "llm_fallback" | "fail_open" | "fail_closed"
    categories: list[str] = field(default_factory=list)
    blocked_message: Optional[str] = None


def _allowed(source: str) -> ModerationResult:
    return ModerationResult(allowed=True, source=source)


def _blocked(source: str, categories: list[str]) -> ModerationResult:
    return ModerationResult(
        allowed=False,
        source=source,
        categories=categories,
        blocked_message=BLOCKED_MESSAGE,
    )


async def moderate_text(text: str, *, llm: Optional[LLMProvider] = None) -> ModerationResult:
    """Clasifica `text` como aceptable o no. Nunca levanta excepción.

    `llm` es el `LLMProvider` ya inyectado por el endpoint llamante (mismo
    proveedor activo, ver ADR-003) — se usa solo como fallback si no hay
    `MODERATION_API_KEY` configurada o si la Moderation API de OpenAI falla.
    """
    settings = get_settings()

    if not settings.moderation_enabled:
        return _allowed("disabled")

    if not text or not text.strip():
        return _allowed("disabled")

    if settings.moderation_api_key:
        try:
            return await _moderate_via_openai_api(text, settings.moderation_api_key)
        except Exception as exc:
            logger.warning(
                "Moderation API de OpenAI falló, cayendo a fallback vía LLM: %s: %s",
                type(exc).__name__, exc,
            )

    if llm is not None:
        try:
            return await _moderate_via_llm(text, llm)
        except Exception as exc:
            logger.error(
                "Moderación vía LLM también falló: %s: %s", type(exc).__name__, exc,
            )

    # Ambos caminos fallaron (o no había LLM disponible) — ver fail-safe
    # documentado arriba en el docstring del módulo.
    if settings.moderation_fail_open:
        return _allowed("fail_open")
    return ModerationResult(
        allowed=False,
        source="fail_closed",
        blocked_message=_SERVICE_UNAVAILABLE_MESSAGE,
    )


async def _moderate_via_openai_api(text: str, api_key: str) -> ModerationResult:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            _OPENAI_MODERATION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": _OPENAI_MODERATION_MODEL, "input": text},
        )
        response.raise_for_status()
        data = response.json()

    result = (data.get("results") or [{}])[0]
    if not result.get("flagged", False):
        return _allowed("openai_api")

    categories_dict = result.get("categories") or {}
    categories = [name for name, flagged in categories_dict.items() if flagged]
    return _blocked("openai_api", categories)


async def _moderate_via_llm(text: str, llm: LLMProvider) -> ModerationResult:
    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": text[:_LLM_CLASSIFIER_MAX_CHARS]},
    ]
    result = await llm.chat_completion(
        messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        reasoning_effort="none",
    )

    raw = result.text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]) if raw.endswith("```") else raw.strip("`")

    parsed = json.loads(raw)
    if not parsed.get("flagged", False):
        return _allowed("llm_fallback")

    categories = parsed.get("categories") or []
    if not isinstance(categories, list):
        categories = []
    return _blocked("llm_fallback", [str(c) for c in categories])
