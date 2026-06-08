"""
Recorder de gaps del docente (Feature G).

Persiste preguntas del alumno cuando el material indexado del curso no las
pudo responder. Combina dos señales:

  Señal 1 — retrieval: chunks == 0 o max(similarity) < 0.4.
            Falso negativo posible cuando el retrieval trae chunks de
            similarity alta pero IRRELEVANTES (matches semánticos espurios
            por términos comunes).

  Señal 2 — respuesta del LLM: si el modelo explícitamente dijo "no encontré
            en el material", "no puedo responder", etc., es un gap aunque el
            retrieval haya devuelto chunks. Esta señal compensa los falsos
            negativos de la señal 1.

Cualquiera de las dos señales activa el registro.

El llamado es fire-and-forget: si falla la persistencia NO debe afectar el
chat del alumno (es telemetría secundaria, no funcionalidad crítica).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UnansweredQuestion

logger = logging.getLogger("nexusai.gaps")

# Umbral debajo del cual consideramos un match "débil".
WEAK_MATCH_THRESHOLD = 0.4

# Patrones de respuesta del LLM que indican que no pudo responder con el material.
# Cubrimos español + inglés (current_language en el front decide la respuesta).
# Patrones genéricos enough para no requerir matching exacto.
LLM_NO_ANSWER_PATTERNS = re.compile(
    r"("
    r"no\s+(se\s+)?encuentr[ao]\s+(esa\s+)?información\s+(en\s+)?(el|los)\s+(material|fragmento|apunte)"
    r"|no\s+puedo\s+responder"
    r"|no\s+tengo\s+información"
    r"|no\s+aparec[ae]\s+en\s+(el|los)\s+(material|fragmento|apunte)"
    r"|fuera\s+del\s+alcance\s+del\s+material"
    r"|el\s+material\s+(del\s+curso\s+)?no\s+(lo\s+)?cubre"
    r"|the\s+(course\s+)?material\s+does\s+not\s+(cover|contain|mention)"
    r"|i\s+don[‘’]?t\s+have\s+(that\s+)?information"
    r"|i\s+cannot\s+answer"
    r"|not\s+(found|covered)\s+in\s+the\s+material"
    # Respuestas de meta-guard (capability probing / prompt injection)
    r"|solo\s+puedo\s+ayudarte\s+con\s+consultas\s+sobre\s+el\s+material"
    r"|only\s+(help|assist)\s+you\s+with\s+(questions|queries)\s+about.*?(course\s+)?material"
    # Variantes adicionales de "no está en el material"
    r"|no\s+est[aá]\s+disponible\s+en\s+el\s+material"
    r"|no\s+forma\s+parte\s+del\s+material"
    r"|no\s+est[aá]\s+cubierta?\s+en\s+(el|los)\s+(material|fragmento|apunte)"
    r")",
    re.IGNORECASE,
)


def llm_indicated_no_answer(text: Optional[str]) -> bool:
    """True si la respuesta del LLM señala que no pudo responder con el material."""
    if not text:
        return False
    return bool(LLM_NO_ANSWER_PATTERNS.search(text))


async def record_gap_if_needed(
    db: AsyncSession,
    *,
    course_id: int,
    user_id: int,
    question: str,
    chunks_count: int,
    max_similarity: Optional[float],
    llm_answer: Optional[str] = None,
) -> bool:
    """Persiste un gap si el material no respondió bien.

    No commitea — el caller debe hacer commit. Esto mantiene consistencia con
    el flujo del mensaje del alumno (mismo commit).

    Devuelve True si se registró un gap (útil para tests y telemetría).
    Si falla, swallowea la excepción y loguea — un gap no registrado es
    preferible a romper el chat.
    """
    try:
        no_chunks = chunks_count <= 0
        weak_match = max_similarity is not None and max_similarity < WEAK_MATCH_THRESHOLD
        llm_said_no = llm_indicated_no_answer(llm_answer)

        if not (no_chunks or weak_match or llm_said_no):
            return False

        gap = UnansweredQuestion(
            course_id=course_id,
            user_id=user_id,
            question=question.strip()[:2000],
            max_similarity=max_similarity,
            chunks_retrieved=chunks_count,
        )
        db.add(gap)
        logger.info(
            "gap_recorded course_id=%d user_id=%d chunks=%d sim=%s llm_said_no=%s",
            course_id, user_id, chunks_count, max_similarity, llm_said_no,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to record gap: %s: %s", type(exc).__name__, exc)
        return False
