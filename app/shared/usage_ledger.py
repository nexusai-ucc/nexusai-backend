"""
Registro de consumo por llamada a proveedor de IA (COST-01, issue #519).

Cada llamada al LLM, a embeddings o a transcripción de voz deja una fila en
`llm_usage` con los tokens, el modelo que respondió de verdad (después del
fallback), el costo, el rol de quien la originó y el resultado. El registro
se hace desde la capa de proveedores (app/providers/*), así que ningún
endpoint tiene que acordarse de registrar su consumo.

Contexto por pedido: `verify_hmac` (app/auth/hmac.py) llama a
`set_usage_context()` con la ruta, el curso, el usuario y el rol. Se guarda
en un ContextVar, que asyncio copia a las tareas creadas durante el pedido
(por ejemplo, la indexación de documentos en background).

Fail-open: registrar nunca rompe el pedido del alumno. Si la escritura
falla, se loguea y se sigue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import urlparse

from app.shared.config import get_settings

logger = logging.getLogger(__name__)

ROLES = frozenset({"student", "teacher", "system"})

_KNOWN_HOSTS = {
    "generativelanguage.googleapis.com": "google",
    "api.openai.com": "openai",
    "api.groq.com": "groq",
}
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "host.docker.internal", "ollama"})

# Precio por (proveedor, modelo) cacheado en memoria para no consultar
# model_prices en cada llamada. Un precio nuevo se nota a los 10 minutos.
_PRICE_TTL_SEC = 600
_MILLION = Decimal(1_000_000)
_COST_QUANTUM = Decimal("0.00000001")

# Alerta de modelo sin precio: una por modelo por día.
_MISSING_PRICE_ALERT_TTL_SEC = 86_400


@dataclass(frozen=True)
class UsageContext:
    """Datos del pedido que originó las llamadas a proveedores."""

    feature: str = "unknown"
    course_id: Optional[int] = None
    user_id: Optional[int] = None
    role: str = "unknown"
    request_id: Optional[str] = None
    client_id: Optional[str] = None


@dataclass(frozen=True)
class UsageRecord:
    """Una llamada a un proveedor, tal como la ve la capa de proveedores.

    `feature`, `course_id` y `user_id` pisan los del contexto del pedido
    cuando el caller sabe algo más preciso (por ejemplo, un bloqueo de
    moderación informado desde un endpoint).
    """

    kind: str
    status: str = "ok"
    provider: Optional[str] = None
    model: Optional[str] = None
    fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    embedding_tokens: int = 0
    audio_seconds: Optional[float] = None
    latency_ms: Optional[float] = None
    cache_hit: bool = False
    saved_tokens: int = 0
    # True cuando el proveedor no informó los tokens y se estimaron con
    # tiktoken (p. ej. los embeddings de Gemini vía su API compatible).
    estimated: bool = False
    feature: Optional[str] = None
    course_id: Optional[int] = None
    user_id: Optional[int] = None


@dataclass(frozen=True)
class Price:
    """Precios en USD vigentes para un modelo."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cached_input_per_mtok: Optional[Decimal]
    embedding_per_mtok: Decimal
    audio_per_minute: Decimal


_context: ContextVar[UsageContext] = ContextVar(
    "nexusai_usage_context", default=UsageContext()
)
_price_cache: dict[tuple[str, str], tuple[float, Optional[Price]]] = {}
_background_tasks: set[asyncio.Task] = set()


# ============================================================
# Contexto del pedido
# ============================================================


def get_usage_context() -> UsageContext:
    return _context.get()


def set_usage_context(ctx: UsageContext) -> None:
    _context.set(ctx)


@contextmanager
def usage_scope(step: str) -> Iterator[None]:
    """Marca las llamadas de un paso interno dentro de la función del pedido,
    para poder separarlas en el registro: "chat.stream" pasa a
    "chat.stream:moderation" mientras dura el bloque."""
    ctx = _context.get()
    token = _context.set(replace(ctx, feature=f"{ctx.feature}:{step}"[:80]))
    try:
        yield
    finally:
        _context.reset(token)


def feature_from_route_path(path: str) -> str:
    """ "/api/v1/documents/{document_id}/preview" -> "documents.preview"."""
    trimmed = path
    for prefix in ("/api/v1/", "/api/"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix) :]
            break
    parts = [p for p in trimmed.strip("/").split("/") if p and not p.startswith("{")]
    return (".".join(parts) or "unknown")[:80]


def provider_from_base_url(base_url: Optional[str]) -> Optional[str]:
    """Nombre corto del proveedor a partir de su base_url."""
    host = (urlparse(base_url or "").hostname or "").lower()
    if host in _KNOWN_HOSTS:
        return _KNOWN_HOSTS[host]
    if host in _LOCAL_HOSTS:
        return "local"
    return host[:40] or None


def resolve_role(header_value: Optional[str], is_teacher: Any = None) -> str:
    """Rol de quien originó el pedido.

    El plugin lo manda en X-NexusAI-Role, calculado en PHP con
    has_capability(); nunca viene del navegador. Si falta, se usa el
    is_teacher del cuerpo (lo manda el chat desde el presupuesto de tokens).
    """
    value = (header_value or "").strip().lower()
    if value in ROLES:
        return value
    if is_teacher is True:
        return "teacher"
    if is_teacher is False:
        return "student"
    return "unknown"


def client_id_for(api_key: str) -> str:
    """Identifica la instalación de Moodle sin exponer su API key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def context_from_request(
    *,
    route_path: str,
    body: bytes,
    query: Mapping[str, str],
    role_header: Optional[str],
    request_id: Optional[str],
    api_key: str,
) -> UsageContext:
    """Arma el contexto a partir de un pedido ya validado por verify_hmac."""
    data: dict[str, Any] = {}
    if body:
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            data = parsed

    def pick(key: str) -> Optional[int]:
        return _positive_int(data.get(key, query.get(key)))

    return UsageContext(
        feature=feature_from_route_path(route_path),
        course_id=pick("course_id"),
        user_id=pick("user_id"),
        role=resolve_role(role_header, data.get("is_teacher")),
        request_id=request_id,
        client_id=client_id_for(api_key),
    )


# ============================================================
# Números de uso y costo
# ============================================================


def as_int(value: Any) -> int:
    """Entero no negativo, o 0 si el SDK no mandó un número."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def llm_usage_numbers(usage: Any) -> tuple[int, int, int]:
    """(prompt, completion, cached) de un `usage` del SDK de OpenAI."""
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = as_int(getattr(details, "cached_tokens", 0)) if details is not None else 0
    return (
        as_int(getattr(usage, "prompt_tokens", 0)),
        as_int(getattr(usage, "completion_tokens", 0)),
        cached,
    )


def embedding_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    return as_int(getattr(usage, "prompt_tokens", 0)) or as_int(
        getattr(usage, "total_tokens", 0)
    )


def status_for_exception(exc: BaseException) -> str:
    """ "quota" para cuota agotada (429); "error" para cualquier otra falla."""
    import openai

    return "quota" if isinstance(exc, openai.RateLimitError) else "error"


def _consumed_nothing(record: UsageRecord) -> bool:
    return (
        record.prompt_tokens == 0
        and record.completion_tokens == 0
        and record.embedding_tokens == 0
        and not record.audio_seconds
    )


def compute_cost(price: Optional[Price], record: UsageRecord) -> Optional[Decimal]:
    """Costo en USD de una llamada. None si hubo consumo y no hay precio."""
    if _consumed_nothing(record):
        return Decimal(0)
    if price is None:
        return None
    cached = min(record.cached_prompt_tokens, record.prompt_tokens)
    uncached = record.prompt_tokens - cached
    cached_rate = (
        price.cached_input_per_mtok
        if price.cached_input_per_mtok is not None
        else price.input_per_mtok
    )
    cost = (
        Decimal(uncached) * price.input_per_mtok
        + Decimal(cached) * cached_rate
        + Decimal(record.completion_tokens) * price.output_per_mtok
        + Decimal(record.embedding_tokens) * price.embedding_per_mtok
    ) / _MILLION
    if record.audio_seconds:
        cost += (
            Decimal(str(record.audio_seconds)) / Decimal(60) * price.audio_per_minute
        )
    return cost.quantize(_COST_QUANTUM)


def clear_price_cache() -> None:
    _price_cache.clear()


# ============================================================
# Registro
# ============================================================


def build_row(record: UsageRecord, ctx: UsageContext) -> dict[str, Any]:
    """Columnas de llm_usage (sin el costo, que se calcula al persistir)."""
    values = asdict(record)
    for key in ("feature", "course_id", "user_id"):
        values.pop(key)
    return {
        **values,
        "feature": (record.feature or ctx.feature)[:80],
        "course_id": record.course_id
        if record.course_id is not None
        else ctx.course_id,
        "user_id": record.user_id if record.user_id is not None else ctx.user_id,
        "role": ctx.role,
        "request_id": ctx.request_id,
        "client_id": ctx.client_id,
        "created_at": datetime.now(timezone.utc),
    }


async def record_usage(record: UsageRecord) -> None:
    """Escribe una fila en llm_usage. Nunca propaga excepciones."""
    if not get_settings().usage_ledger_enabled:
        return
    try:
        row = build_row(record, get_usage_context())
        await _persist(row, record)
    except Exception as exc:
        logger.warning(
            "No se pudo registrar el consumo (%s/%s): %s: %s",
            record.kind,
            record.model,
            type(exc).__name__,
            exc,
        )


def schedule_usage(record: UsageRecord) -> None:
    """Como record_usage, pero sin esperar: para código sincrónico o para
    registrar desde un stream que se está cancelando."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(record_usage(record))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _persist(row: dict[str, Any], record: UsageRecord) -> None:
    from app.db.models import LlmUsage
    from app.db.session import get_session_factory

    missing_price: Optional[tuple[str, str]] = None
    async with get_session_factory()() as session:
        cost: Optional[Decimal] = Decimal(0)
        if not _consumed_nothing(record):
            price = None
            if record.provider and record.model:
                price = await _lookup_price(session, record.provider, record.model)
            cost = compute_cost(price, record)
            if cost is None and record.provider and record.model:
                missing_price = (record.provider, record.model)
        session.add(LlmUsage(**row, cost_usd=cost))
        await session.commit()

    if missing_price:
        await _alert_missing_price(*missing_price)


async def _lookup_price(session: Any, provider: str, model: str) -> Optional[Price]:
    from sqlalchemy import select

    from app.db.models import ModelPrice

    key = (provider, model)
    now = time.monotonic()
    cached = _price_cache.get(key)
    if cached is not None and now - cached[0] < _PRICE_TTL_SEC:
        return cached[1]

    stmt = (
        select(ModelPrice)
        .where(
            ModelPrice.provider == provider,
            ModelPrice.model == model,
            ModelPrice.valid_from <= datetime.now(timezone.utc),
        )
        .order_by(ModelPrice.valid_from.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    price = (
        Price(
            input_per_mtok=Decimal(row.input_per_mtok),
            output_per_mtok=Decimal(row.output_per_mtok),
            cached_input_per_mtok=(
                Decimal(row.cached_input_per_mtok)
                if row.cached_input_per_mtok is not None
                else None
            ),
            embedding_per_mtok=Decimal(row.embedding_per_mtok),
            audio_per_minute=Decimal(row.audio_per_minute),
        )
        if row is not None
        else None
    )
    _price_cache[key] = (now, price)
    return price


async def _alert_missing_price(provider: str, model: str) -> None:
    from app.infrastructure.redis_client import get_redis
    from app.shared.alerting import send_alert

    try:
        redis = await get_redis()
        first_today = await redis.set(
            f"nexusai:alert:model_price_missing:{provider}:{model}",
            "1",
            nx=True,
            ex=_MISSING_PRICE_ALERT_TTL_SEC,
        )
    except Exception as exc:
        logger.warning("No se pudo chequear la alerta de precio faltante: %s", exc)
        return
    if not first_today:
        return
    logger.warning(
        "Modelo sin precio en model_prices: %s/%s (cost_usd queda en NULL).",
        provider,
        model,
    )
    task = asyncio.create_task(
        send_alert(
            f"NexusAI: modelo sin precio ({provider}/{model})",
            f"El modelo {model} de {provider} se está usando y no tiene precio "
            "en model_prices, así que su costo queda vacío en llm_usage. "
            "Cargalo con scripts/model_prices.py.",
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
