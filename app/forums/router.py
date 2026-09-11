"""
Foros con IA — Épica 06.

Endpoints:

  POST /api/v1/forums/index-post
    Indexa (o re-indexa) un post de foro. Llamado por el observer PHP
    cada vez que un post se crea o edita en Moodle.
    Solo re-embeddea si el content_hash cambió (evita trabajo innecesario).

  DELETE /api/v1/forums/index-post/{post_id}
    Elimina el embedding de un post. Llamado por el observer PHP cuando
    el post se borra en Moodle.

  POST /api/v1/forums/similar-posts
    Recibe el texto de un post en redacción y devuelve los posts existentes
    en el mismo curso que sean semánticamente similares.
    Usado por el frontend para avisar al alumno antes de publicar.

  POST /api/v1/forums/summarize-thread
    Recibe los posts de una discusión (enviados por PHP desde Moodle DB) y
    devuelve un resumen estructurado generado por el LLM. No usa pgvector —
    es generación pura a partir del texto del hilo. Detecta si la pregunta
    original fue resuelta en el hilo.

  POST /api/v1/forums/suggest-reply
    Recibe el hilo completo + el post específico al que se responde y genera
    una sugerencia de respuesta con RAG: primero recupera chunks relevantes
    del material del curso (pgvector), luego el LLM sintetiza la respuesta
    combinando el contexto del hilo con el material académico.

Todos los endpoints llevan verify_hmac — contrato de seguridad invariante.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re as _re
import uuid
from typing import Annotated, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.logger import log_moderation_block
from app.auth.hmac import verify_hmac
from app.db.models import ForumPostEmbedding, ForumWebhookConfig
from app.db.session import get_db
from app.documents.retriever import format_context_for_prompt, retrieve_context
from app.providers.embeddings import EmbeddingProvider, get_embedding_provider
from app.providers.llm import LLMProvider, get_llm_provider
from app.shared.config import get_settings
from app.shared.moderation import moderate_text

_logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_SIMILARITY_THRESHOLD = 0.75
_DEFAULT_TOP_K = 5


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────

class IndexPostRequest(BaseModel):
    post_id: int = Field(gt=0, description="ID de mdl_forum_posts")
    discussion_id: int = Field(gt=0, description="ID de mdl_forum_discussions")
    course_id: int = Field(gt=0)
    content: str = Field(min_length=1, max_length=10_000)


class IndexPostResponse(BaseModel):
    post_id: int
    status: str  # "indexed" | "skipped" (sin cambios en contenido)


class SimilarPostsRequest(BaseModel):
    course_id: int = Field(gt=0)
    text: str = Field(min_length=10, max_length=5_000, description="Texto del post en redacción")
    exclude_post_id: Optional[int] = Field(
        default=None,
        description="Post a excluir de los resultados (útil al editar un post existente)",
    )
    threshold: float = Field(
        default=_DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similitud mínima para considerar duplicado",
    )
    top_k: int = Field(default=_DEFAULT_TOP_K, ge=1, le=10)


class SimilarPost(BaseModel):
    forum_post_id: int
    discussion_id: int
    similarity: float
    preview: str  # primeros 200 chars del contenido


class SimilarPostsResponse(BaseModel):
    similar_posts: List[SimilarPost]
    threshold_used: float


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/index-post", response_model=IndexPostResponse, status_code=status.HTTP_200_OK)
async def index_post(
    payload: IndexPostRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> IndexPostResponse:
    """Indexa o actualiza el embedding de un post de foro.

    Si el post ya existe con el mismo contenido (igual content_hash),
    devuelve status='skipped' sin hacer nada. Si el contenido cambió,
    re-embeddea y actualiza.
    """
    content_hash = _sha256(payload.content)

    # Buscar si ya existe
    result = await db.execute(
        select(ForumPostEmbedding).where(
            ForumPostEmbedding.forum_post_id == payload.post_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None and existing.content_hash == content_hash:
        return IndexPostResponse(post_id=payload.post_id, status="skipped")

    try:
        vector = await embeddings.embed(payload.content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo generar el embedding del post",
        ) from exc

    if existing is None:
        record = ForumPostEmbedding(
            id=uuid.uuid4(),
            forum_post_id=payload.post_id,
            discussion_id=payload.discussion_id,
            course_id=payload.course_id,
            content_hash=content_hash,
            content=payload.content,
            embedding=vector,
        )
        db.add(record)
    else:
        existing.content_hash = content_hash
        existing.content = payload.content
        existing.embedding = vector

    await db.commit()
    return IndexPostResponse(post_id=payload.post_id, status="indexed")


@router.delete("/index-post/{post_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_post_embedding(
    _body: Annotated[bytes, Depends(verify_hmac)],
    post_id: int = Path(gt=0),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Elimina el embedding de un post cuando se borra en Moodle."""
    await db.execute(
        delete(ForumPostEmbedding).where(
            ForumPostEmbedding.forum_post_id == post_id
        )
    )
    await db.commit()


@router.post("/similar-posts", response_model=SimilarPostsResponse)
async def similar_posts(
    payload: SimilarPostsRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
) -> SimilarPostsResponse:
    """Detecta posts existentes similares al texto que el alumno está escribiendo.

    Usa similitud coseno sobre pgvector. Solo busca dentro del mismo curso.
    Devuelve lista vacía si no hay nada por encima del threshold.
    """
    try:
        vector = await embeddings.embed(payload.text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo vectorizar la consulta",
        ) from exc

    # Usar SQLAlchemy ORM con pgvector (igual que retrieve_context) para evitar
    # problemas de tipos con asyncpg al pasar embeddings como raw SQL params.
    distance_expr = ForumPostEmbedding.embedding.cosine_distance(vector)
    similarity_expr = (1 - distance_expr).label("similarity")

    stmt = (
        select(
            ForumPostEmbedding.forum_post_id,
            ForumPostEmbedding.discussion_id,
            ForumPostEmbedding.content,
            similarity_expr,
        )
        .where(ForumPostEmbedding.course_id == payload.course_id)
        .where(ForumPostEmbedding.embedding.is_not(None))
        .where((1 - distance_expr) >= payload.threshold)
        .order_by(distance_expr)
        .limit(payload.top_k)
    )
    if payload.exclude_post_id is not None:
        stmt = stmt.where(ForumPostEmbedding.forum_post_id != payload.exclude_post_id)

    try:
        result = await db.execute(stmt)
        rows = result.all()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error al buscar posts similares",
        ) from exc

    similar = [
        SimilarPost(
            forum_post_id=row.forum_post_id,
            discussion_id=row.discussion_id,
            similarity=round(float(row.similarity), 3),
            preview=row.content[:200].strip(),
        )
        for row in rows
    ]

    return SimilarPostsResponse(
        similar_posts=similar,
        threshold_used=payload.threshold,
    )


# ─────────────────────────────────────────────────────────────
# F-04 — Schemas
# ─────────────────────────────────────────────────────────────

# Máximo de posts que se incluyen en el prompt del LLM.
# Los que exceden se truncan para no superar el context window.
_MAX_POSTS_IN_PROMPT = 30
# Máximo de caracteres por post incluido en el prompt.
_MAX_CHARS_PER_POST = 1_000


class ThreadPost(BaseModel):
    post_id: int = Field(gt=0)
    author: str = Field(max_length=200)
    content: str = Field(min_length=1, max_length=10_000)


class SummarizeThreadRequest(BaseModel):
    discussion_id: int = Field(gt=0)
    course_id: int = Field(gt=0)
    posts: List[ThreadPost] = Field(min_length=1, max_length=50)


class SummarizeThreadResponse(BaseModel):
    summary: str
    key_points: List[str]
    resolved: bool
    posts_used: int
    posts_truncated: bool


# ─────────────────────────────────────────────────────────────
# F-04 — Helpers
# ─────────────────────────────────────────────────────────────

def _build_summarize_prompt(posts: List[ThreadPost]) -> tuple[str, int, bool]:
    """Construye el bloque de texto del hilo para el prompt del LLM.

    Trunca a _MAX_POSTS_IN_PROMPT posts y a _MAX_CHARS_PER_POST chars por post
    para no superar el context window del LLM. Devuelve (texto, posts_usados, truncado).
    """
    selected = posts[:_MAX_POSTS_IN_PROMPT]
    truncated = len(posts) > _MAX_POSTS_IN_PROMPT

    lines: list[str] = []
    for i, p in enumerate(selected, 1):
        content = p.content[:_MAX_CHARS_PER_POST]
        if len(p.content) > _MAX_CHARS_PER_POST:
            content += "…"
            truncated = True
        lines.append(f"[Post {i} — {p.author}]\n{content}")

    return "\n\n".join(lines), len(selected), truncated


_SUMMARIZE_SYSTEM = """\
Sos un asistente académico que resume discusiones de foro universitario.
Respondé siempre en el mismo idioma que los posts del hilo (español o inglés).
Sé conciso y objetivo — no agregues información que no esté en los posts.
"""

_SUMMARIZE_USER_TMPL = """\
Resumí el siguiente hilo de foro académico.

HILO:
{thread_text}

Respondé con un JSON válido con exactamente estas claves (sin texto antes ni después):
{{
  "summary": "<resumen del hilo en 2-4 oraciones>",
  "key_points": ["<punto clave 1>", "<punto clave 2>"],
  "resolved": <true si la pregunta principal quedó respondida, false si no>
}}
"""


# ─────────────────────────────────────────────────────────────
# F-04 — Endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/summarize-thread", response_model=SummarizeThreadResponse)
async def summarize_thread(
    payload: SummarizeThreadRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    llm: LLMProvider = Depends(get_llm_provider),
) -> SummarizeThreadResponse:
    """Resume un hilo de foro usando el LLM.

    PHP envía los posts ya leídos desde Moodle DB. El endpoint construye
    un prompt estructurado, llama al LLM y parsea la respuesta JSON.
    No escribe en DB ni usa pgvector — es stateless.
    """
    thread_text, posts_used, posts_truncated = _build_summarize_prompt(payload.posts)

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user",   "content": _SUMMARIZE_USER_TMPL.format(thread_text=thread_text)},
    ]

    try:
        result = await llm.chat_completion(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El LLM no está disponible temporalmente",
        ) from exc

    # El LLM devuelve JSON puro según el prompt. Lo parseamos con tolerancia:
    # si falla el parse, devolvemos el texto crudo como summary.
    raw = result.text.strip()
    # Extraer bloque JSON aunque el LLM añada markdown (```json ... ```)
    json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
    parsed: dict = {}
    if json_match:
        try:
            parsed = _json.loads(json_match.group())
        except _json.JSONDecodeError:
            pass

    summary    = str(parsed.get("summary",    raw))
    key_points = parsed.get("key_points", [])
    resolved   = bool(parsed.get("resolved",  False))

    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(kp) for kp in key_points if kp]

    return SummarizeThreadResponse(
        summary=summary,
        key_points=key_points,
        resolved=resolved,
        posts_used=posts_used,
        posts_truncated=posts_truncated,
    )


# ─────────────────────────────────────────────────────────────
# F-05 — Schemas
# ─────────────────────────────────────────────────────────────

_MAX_CONTEXT_CHUNKS = 8
_MAX_CHARS_QUESTION = 2_000


class SuggestReplyRequest(BaseModel):
    discussion_id: int = Field(gt=0)
    course_id: int = Field(gt=0)
    posts: List[ThreadPost] = Field(min_length=1, max_length=50)
    question: str = Field(
        min_length=1,
        max_length=_MAX_CHARS_QUESTION,
        description="Texto del post al que se responde (para RAG y contexto del LLM)",
    )


class SuggestReplyResponse(BaseModel):
    suggested_reply: str
    has_course_material: bool
    sources_used: int


# ─────────────────────────────────────────────────────────────
# F-05 — Prompts
# ─────────────────────────────────────────────────────────────

_SUGGEST_SYSTEM = """\
Sos un asistente académico que ayuda a responder preguntas en foros universitarios.
Generá una respuesta clara y útil basándote en el contexto del hilo y, si se provee,
en el material del curso. Respondé en el mismo idioma que la pregunta (español o inglés).
No inventes información que no esté en el hilo o en el material del curso.
Si usás información del material, citá el nombre del archivo entre paréntesis.
"""

_SUGGEST_USER_TMPL = """\
Tengo que responder a este mensaje en un foro académico:

MENSAJE A RESPONDER:
{question}

CONTEXTO DEL HILO (cronológico, para entender la discusión):
{thread_text}
{material_section}
Escribí una respuesta útil y concisa para ese mensaje.
Devolvé solo el texto de la respuesta, sin JSON ni encabezados."""

_MATERIAL_SECTION_TMPL = """\

MATERIAL DEL CURSO (fragmentos relevantes recuperados por búsqueda semántica):
{context}
"""


# ─────────────────────────────────────────────────────────────
# F-05 — Endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/suggest-reply", response_model=SuggestReplyResponse)
async def suggest_reply(
    payload: SuggestReplyRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
) -> SuggestReplyResponse:
    """Genera una sugerencia de respuesta para un post de foro usando RAG + LLM.

    Flujo:
      1. Recupera chunks del material del curso relevantes a la pregunta (pgvector).
      2. Construye el prompt con el hilo + material recuperado.
      3. El LLM genera la respuesta sugerida.
    """
    # ----- Moderación de contenido — antes de gastar tokens en RAG/LLM y
    # antes de que una respuesta generada a partir de este post llegue a
    # otro alumno del foro. Ver app/shared/moderation.py. -----
    moderation = await moderate_text(payload.question, llm=llm)
    if not moderation.allowed:
        log_moderation_block(
            endpoint="forums.suggest_reply",
            course_id=payload.course_id,
            user_id=None,  # el post no trae user_id — PHP no lo envía en este endpoint
            source=moderation.source,
            categories=moderation.categories,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=moderation.blocked_message)

    # 1. RAG: buscar material del curso relevante a la pregunta.
    try:
        chunks = await retrieve_context(
            question=payload.question,
            course_id=payload.course_id,
            db=db,
            embeddings=embeddings,
            top_k=_MAX_CONTEXT_CHUNKS,
            min_similarity=0.35,
        )
    except Exception:
        chunks = []

    has_material = bool(chunks)
    sources_used = len(chunks)

    # 2. Formatear el contexto del hilo.
    thread_text, _, _ = _build_summarize_prompt(payload.posts)

    # 3. Formatear el material del curso (si lo hay).
    material_section = ""
    if chunks:
        context_str = format_context_for_prompt(chunks, max_chars_per_chunk=1600)
        material_section = _MATERIAL_SECTION_TMPL.format(context=context_str)

    messages = [
        {"role": "system", "content": _SUGGEST_SYSTEM},
        {
            "role": "user",
            "content": _SUGGEST_USER_TMPL.format(
                question=payload.question[:_MAX_CHARS_QUESTION],
                thread_text=thread_text,
                material_section=material_section,
            ),
        },
    ]

    try:
        result = await llm.chat_completion(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El LLM no está disponible temporalmente",
        ) from exc

    return SuggestReplyResponse(
        suggested_reply=result.text.strip(),
        has_course_material=has_material,
        sources_used=sources_used,
    )


# ─────────────────────────────────────────────────────────────
# FOR-05 (#366) — Heurística de urgencia/frustración
# ─────────────────────────────────────────────────────────────

_URGENCY_KEYWORDS = [
    "urgente", "urgencia", "urgent",
    "ayuda", "help",
    "no entiendo", "i don't understand", "i dont understand",
    "no puedo", "i can't", "i cant",
    "desesperad", "desperate",
    "frustrad", "frustrat",
    "perdido", "perdida", "lost",
    "confundid", "confused",
    "por favor", "please help",
]


def detect_urgency(text: str) -> bool:
    """Heurística simple de urgencia/frustración — sin LLM, determinística y
    fácil de verificar con casos de prueba reales (el issue permite
    explícitamente "una llamada LLM o una heurística simple"; una heurística
    es más barata y no depende de la disponibilidad del proveedor).

    Marca "urgente" solo si se combinan AL MENOS 2 señales de las 3 — evita
    falsos positivos por una sola palabra clave suelta o un simple signo de
    pregunta en un post neutral:
      1. Palabra clave de urgencia/frustración/pedido de ayuda.
      2. Densidad alta de "!"/"?" (3 o más en el post).
      3. Un tramo largo en mayúsculas sostenidas ("grito", 15+ caracteres).
    """
    if not text or not text.strip():
        return False

    lower = text.lower()
    signals = 0

    if any(kw in lower for kw in _URGENCY_KEYWORDS):
        signals += 1

    if (text.count("!") + text.count("?")) >= 3:
        signals += 1

    if _re.search(r"[A-ZÁÉÍÓÚÑ]{15,}", text):
        signals += 1

    return signals >= 2


# ─────────────────────────────────────────────────────────────
# FOR-06 (#367) — Digest semanal del foro
# ─────────────────────────────────────────────────────────────
#
# Un solo endpoint combinado con FOR-05: ambas issues comparten la misma
# ventana de datos (últimos N días) y el mismo momento de revisión del
# docente ("no tener que entrar hilo por hilo"), así que la señal de
# urgencia por hilo viaja junto con el resumen general en la misma
# respuesta, sin duplicar el fetch de discusiones.
#
# Este backend NO tiene copia de foros/hilos/posts (solo embeddings para
# duplicados, ForumPostEmbedding) — PHP arma la lista de discusiones con
# actividad reciente consultando directo las tablas nativas de Moodle
# (que sí tienen timestamps) y la manda acá. Stateless, no escribe en DB.

_MAX_DISCUSSIONS_IN_DIGEST = 15
_MAX_POSTS_PER_DISCUSSION_IN_DIGEST = 20


class DigestDiscussion(BaseModel):
    discussion_id: int = Field(gt=0)
    discussion_name: str = Field(max_length=300)
    forum_name: str = Field(max_length=300)
    posts: List[ThreadPost] = Field(min_length=1, max_length=_MAX_POSTS_PER_DISCUSSION_IN_DIGEST)


class WeeklyDigestRequest(BaseModel):
    course_id: int = Field(gt=0)
    days: int = Field(default=7, ge=1, le=30)
    discussions: List[DigestDiscussion] = Field(default_factory=list, max_length=_MAX_DISCUSSIONS_IN_DIGEST)


class DigestDiscussionResult(BaseModel):
    discussion_id: int
    discussion_name: str
    forum_name: str
    post_count: int
    urgent: bool  # FOR-05: al menos un post del hilo disparó la heurística


class WeeklyDigestResponse(BaseModel):
    course_id: int
    period_days: int
    discussion_count: int
    discussions: List[DigestDiscussionResult]
    summary: Optional[str] = None  # None si no hubo actividad en la ventana


def _build_digest_prompt(discussions: List[DigestDiscussion]) -> str:
    blocks = []
    for d in discussions:
        thread_text, _, _ = _build_summarize_prompt(d.posts)
        blocks.append(f'=== Hilo: "{d.discussion_name}" (foro: {d.forum_name}) ===\n{thread_text}')
    return "\n\n".join(blocks)


_DIGEST_SYSTEM = """\
Sos un asistente académico que arma un resumen semanal de la actividad de \
los foros de un curso, para que el docente no tenga que revisar cada hilo \
por separado. Respondé en español salvo que la mayoría de los posts estén \
en inglés. Sé conciso y priorizá lo accionable: qué necesita la atención \
del docente, qué se resolvió solo entre los alumnos.
"""

_DIGEST_USER_TMPL = """\
Estos son los hilos de foro con actividad nueva en los últimos {days} días.

{threads_text}

Escribí un resumen de 3-6 oraciones de la actividad de la semana, agrupando \
temas relacionados si los hay y señalando qué hilos parecen necesitar una \
respuesta del docente. Devolvé solo el texto del resumen, sin JSON ni \
encabezados."""


@router.post("/weekly-digest", response_model=WeeklyDigestResponse)
async def weekly_digest(
    payload: WeeklyDigestRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    llm: LLMProvider = Depends(get_llm_provider),
    db: AsyncSession = Depends(get_db),
) -> WeeklyDigestResponse:
    """Resumen semanal del foro (FOR-06) + señal de urgencia por hilo (FOR-05).

    Sin discusiones con actividad → respuesta vacía sin llamar al LLM
    (nada que resumir, no tiene sentido gastar una llamada).
    """
    if not payload.discussions:
        return WeeklyDigestResponse(
            course_id=payload.course_id,
            period_days=payload.days,
            discussion_count=0,
            discussions=[],
            summary=None,
        )

    discussion_results = [
        DigestDiscussionResult(
            discussion_id=d.discussion_id,
            discussion_name=d.discussion_name,
            forum_name=d.forum_name,
            post_count=len(d.posts),
            urgent=any(detect_urgency(p.content) for p in d.posts),
        )
        for d in payload.discussions
    ]

    threads_text = _build_digest_prompt(payload.discussions)
    messages = [
        {"role": "system", "content": _DIGEST_SYSTEM},
        {"role": "user", "content": _DIGEST_USER_TMPL.format(days=payload.days, threads_text=threads_text)},
    ]

    try:
        result = await llm.chat_completion(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El LLM no está disponible temporalmente",
        ) from exc

    summary = result.text.strip()

    # FOR-07 (#378): si el docente configuró un webhook para este curso,
    # reenviar el resumen ahí además de mostrarlo en el panel. Un fallo acá
    # NUNCA debe romper la respuesta al frontend — se loguea y se ignora.
    await _notify_webhook_if_configured(db, payload.course_id, summary)

    return WeeklyDigestResponse(
        course_id=payload.course_id,
        period_days=payload.days,
        discussion_count=len(payload.discussions),
        discussions=discussion_results,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────
# FOR-07 (#378) — Webhook externo para el digest semanal
# ─────────────────────────────────────────────────────────────
#
# El plugin Moodle no tiene ninguna tabla de configuración por-curso
# propia (ver migración 022_forum_webhook_configs) — la URL vive acá, y
# el POST saliente a Slack/Discord/Teams se dispara desde este mismo
# backend (httpx ya es dependencia, sin librería nueva).

_WEBHOOK_POST_TIMEOUT = 8.0


class WebhookConfigSaveRequest(BaseModel):
    course_id: int = Field(gt=0)
    webhook_url: str = Field(default="", max_length=2000)  # "" = borrar config


class WebhookConfigSaveResponse(BaseModel):
    webhook_url: Optional[str]


class WebhookConfigGetRequest(BaseModel):
    course_id: int = Field(gt=0)


class WebhookConfigGetResponse(BaseModel):
    webhook_url: Optional[str]


async def _notify_webhook_if_configured(db: AsyncSession, course_id: int, summary: Optional[str]) -> None:
    if not summary:
        return  # nada que notificar (mismo criterio que "sin discusiones, sin LLM")

    row = await db.execute(
        select(ForumWebhookConfig.webhook_url).where(ForumWebhookConfig.course_id == course_id)
    )
    webhook_url = row.scalar_one_or_none()
    if not webhook_url:
        return

    try:
        async with httpx.AsyncClient(timeout=_WEBHOOK_POST_TIMEOUT) as client:
            await client.post(webhook_url, json={"text": summary})
    except Exception:
        _logger.warning("FOR-07: fallo al notificar el webhook del curso %s", course_id, exc_info=True)


@router.post("/webhook-config/save", response_model=WebhookConfigSaveResponse)
async def save_webhook_config(
    payload: WebhookConfigSaveRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> WebhookConfigSaveResponse:
    """Guarda (o borra, con webhook_url="") la URL de webhook del curso."""
    if not payload.webhook_url:
        await db.execute(delete(ForumWebhookConfig).where(ForumWebhookConfig.course_id == payload.course_id))
        await db.commit()
        return WebhookConfigSaveResponse(webhook_url=None)

    if not _re.match(r"^https?://", payload.webhook_url):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="webhook_url debe ser http(s)")

    stmt = (
        pg_insert(ForumWebhookConfig)
        .values(id=uuid.uuid4(), course_id=payload.course_id, webhook_url=payload.webhook_url)
        .on_conflict_do_update(
            constraint="uq_forum_webhook_configs_course",
            set_={"webhook_url": payload.webhook_url},
        )
        .returning(ForumWebhookConfig.webhook_url)
    )
    result = await db.execute(stmt)
    await db.commit()
    row = result.fetchone()
    return WebhookConfigSaveResponse(webhook_url=row.webhook_url)


@router.post("/webhook-config/get", response_model=WebhookConfigGetResponse)
async def get_webhook_config(
    payload: WebhookConfigGetRequest,
    _body: Annotated[bytes, Depends(verify_hmac)],
    db: AsyncSession = Depends(get_db),
) -> WebhookConfigGetResponse:
    row = await db.execute(
        select(ForumWebhookConfig.webhook_url).where(ForumWebhookConfig.course_id == payload.course_id)
    )
    return WebhookConfigGetResponse(webhook_url=row.scalar_one_or_none())
