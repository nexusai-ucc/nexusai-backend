"""
API key propia del usuario (BYOK) como último eslabón de la cadena de modelos.

Un alumno o docente puede cargar en Moodle la key de su propia cuenta en un
proveedor compatible con el SDK de OpenAI. El plugin la guarda cifrada y la
manda en el cuerpo firmado de cada pedido que usa el LLM, bajo `user_llm`:

    {"user_llm": {"provider": "anthropic", "model": "...", "api_key": "..."}}

El backend no la persiste (ni en la base, ni en Redis, ni en logs): arma un
client con ella y lo agrega al final de la cadena (ver
LLMProvider._fallback_chain). Ese client se reutiliza entre pedidos del mismo
usuario, así que la key vive en la memoria del proceso mientras el client esté
en el caché (ver client_for). Se usa en dos casos:

  1. La cadena de la facultad (primario → intermedios → secundario) se quedó
     sin cuota o está caída: la cadena llega sola al eslabón del usuario.
  2. El usuario llegó a su presupuesto de tokens del chat: el chat responde
     solo con su key (LLMProvider.user_only) en vez de cortar con 429.

La lista de proveedores es cerrada a propósito: el usuario elige un nombre y la
URL la pone el backend. Si la URL viniera del pedido, cualquiera podría hacer
que el servidor le pegue a una dirección interna (SSRF).

La key no se loguea, no se guarda en llm_usage y no aparece en el repr de
UserLLMConfig. El consumo con la key del usuario queda en llm_usage con
`byok=True` y costo 0: lo paga el usuario, no la facultad.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Proveedores que aceptan el formato de chat completions del SDK de OpenAI.
# Mantener sincronizada con local_nexusai\local\user_llm::PROVIDERS (plugin).
PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("OpenAI", "https://api.openai.com/v1"),
    "anthropic": ("Anthropic", "https://api.anthropic.com/v1/"),
    "gemini": (
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "groq": ("Groq", "https://api.groq.com/openai/v1"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1"),
    "mistral": ("Mistral", "https://api.mistral.ai/v1"),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1"),
}

# Nombres de modelo reales: "gpt-4.1-mini", "models/gemini-x", "org/modelo:free".
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,119}$")
_KEY_MIN, _KEY_MAX = 8, 512

# Clients reutilizados entre pedidos del mismo usuario: cada AsyncOpenAI abre
# su propio pool HTTP, y crear uno por pedido dejaría conexiones colgadas.
_CLIENT_CACHE_SIZE = 256
# Un client que sale del caché se cierra recién después de este tiempo, por si
# un pedido todavía lo está usando (el más largo, el chat en streaming, lo
# corta el plugin a los 300 s).
_CLOSE_DELAY_SEC = 330.0


@dataclass(frozen=True)
class UserLLMConfig:
    """La key del usuario para un pedido. `api_key` queda fuera del repr."""

    provider: str
    model: str
    api_key: str = field(repr=False)

    @property
    def base_url(self) -> str:
        return PROVIDERS[self.provider][1]

    @property
    def label(self) -> str:
        return PROVIDERS[self.provider][0]


def parse_user_llm(data: Any) -> Optional[UserLLMConfig]:
    """Valida el objeto `user_llm` del cuerpo. None si falta o es inválido.

    Un `user_llm` mal formado no rompe el pedido: se ignora (el usuario sigue
    con la cadena de la facultad) y se loguea sin la key.
    """
    if data is None:
        return None
    if not isinstance(data, dict):
        logger.warning("user_llm ignorado: no es un objeto")
        return None
    provider = data.get("provider")
    model = data.get("model")
    api_key = data.get("api_key")
    if not isinstance(provider, str) or provider not in PROVIDERS:
        logger.warning("user_llm ignorado: proveedor desconocido")
        return None
    if not isinstance(model, str) or not _MODEL_RE.match(model.strip()):
        logger.warning("user_llm ignorado: nombre de modelo inválido (%s)", provider)
        return None
    if (
        not isinstance(api_key, str)
        or not _KEY_MIN <= len(api_key.strip()) <= _KEY_MAX
        or any(c.isspace() for c in api_key.strip())
    ):
        logger.warning("user_llm ignorado: key con formato inválido (%s)", provider)
        return None
    return UserLLMConfig(
        provider=provider, model=model.strip(), api_key=api_key.strip()
    )


def user_llm_from_body(body: bytes) -> Optional[UserLLMConfig]:
    """Lee `user_llm` del cuerpo JSON ya verificado por HMAC."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return parse_user_llm(data.get("user_llm"))


# (proveedor, sha256 de la key) → client. La clave del caché no es la key en
# claro; la key sí está adentro del client, que la necesita para autenticarse.
_clients: "OrderedDict[tuple[str, str], AsyncOpenAI]" = OrderedDict()
_closing: set[asyncio.Task] = set()


def client_for(provider: str, api_key: str) -> AsyncOpenAI:
    """Client del proveedor del usuario, reutilizado entre sus pedidos.

    Caché acotado (LRU de _CLIENT_CACHE_SIZE): el client que sale se cierra
    después de _CLOSE_DELAY_SEC, así no quedan pools HTTP abiertos ni keys de
    usuarios que ya no la usan en memoria indefinidamente.
    """
    cache_key = (provider, hashlib.sha256(api_key.encode("utf-8")).hexdigest())
    client = _clients.get(cache_key)
    if client is not None:
        _clients.move_to_end(cache_key)
        return client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=PROVIDERS[provider][1],
        timeout=120.0,
        max_retries=0,  # Retries manejados por async_retry, igual que el primario.
    )
    _clients[cache_key] = client
    while len(_clients) > _CLIENT_CACHE_SIZE:
        _, evicted = _clients.popitem(last=False)
        _close_later(evicted)
    return client


def _close_later(client: AsyncOpenAI) -> None:
    """Cierra un client que salió del caché cuando ya no lo use nadie."""
    delay = _CLOSE_DELAY_SEC
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # Fuera de un event loop (scripts, tests sincrónicos).

    async def _close() -> None:
        await asyncio.sleep(delay)
        try:
            await client.close()
        except Exception as exc:  # cerrar nunca debe romper nada
            logger.debug("No se pudo cerrar un client de usuario: %s", exc)

    task = loop.create_task(_close())
    _closing.add(task)
    task.add_done_callback(_closing.discard)


def clear_clients() -> None:
    """Vacía el caché de clients (tests)."""
    _clients.clear()


# ============================================================
# Fallas del eslabón del usuario
# ============================================================

# Motivos que el plugin sabe mostrar (ver errors.js y la pantalla Mi API key).
REASON_REJECTED = "rejected"  # 401/403: key inválida, vencida o borrada.
REASON_QUOTA = "quota"  # 429: la cuenta del usuario no tiene cuota o saldo.
REASON_MODEL = "model"  # 404: el modelo no existe en esa cuenta.
REASON_REQUEST = "request"  # 400/422: el modelo no aceptó el pedido.
REASON_UNAVAILABLE = "unavailable"  # timeout, conexión, 5xx del proveedor.

USER_LLM_ERROR = "user_llm_failed"


def reason_for(exc: BaseException) -> str:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return REASON_REJECTED
    if isinstance(exc, openai.RateLimitError):
        return REASON_QUOTA
    if isinstance(exc, openai.NotFoundError):
        return REASON_MODEL
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        return REASON_REQUEST
    return REASON_UNAVAILABLE


class UserLLMError(Exception):
    """El eslabón del usuario falló y era el último de la cadena.

    Al usuario le llega con un mensaje que nombra a su proveedor (handler de
    main.py). Si antes había fallado la facultad, esa falla viaja en
    `institutional_error` y es la que alimenta las alertas.
    """

    def __init__(
        self,
        reason: str,
        provider: str,
        model: str,
        institutional_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(f"user_llm {provider}/{model}: {reason}")
        self.reason = reason
        self.provider = provider
        self.model = model
        # La falla de la facultad que llevó la cadena hasta la key del usuario
        # (None si se usó solo su key, p. ej. por presupuesto agotado). Con
        # ella los routers siguen disparando la alerta de cuota/caída.
        self.institutional_error = institutional_error

    @property
    def label(self) -> str:
        return PROVIDERS.get(self.provider, (self.provider, ""))[0]

    def message(self, lang: Optional[str]) -> str:
        label = self.label
        if lang == "en":
            return {
                REASON_REJECTED: f"{label} rejected your API key. It may be expired, deleted or out of credit. Check it or add another one in My API key.",
                REASON_QUOTA: f"Your {label} account has no quota or credit left. Check it in your {label} account or add another key in My API key.",
                REASON_MODEL: f"{label} could not find the model {self.model}. Check its exact name in My API key.",
                REASON_REQUEST: f"The model {self.model} of {label} did not accept the request. Try another model in My API key.",
                REASON_UNAVAILABLE: f"{label} is not responding right now. Try again in a few minutes.",
            }[self.reason]
        return {
            REASON_REJECTED: f"{label} rechazó tu API key. Puede estar vencida, borrada o sin saldo. Revisala o cargá otra en Mi API key.",
            REASON_QUOTA: f"Tu cuenta de {label} no tiene cuota o saldo. Revisala en {label} o cargá otra key en Mi API key.",
            REASON_MODEL: f"{label} no encontró el modelo {self.model}. Revisá el nombre exacto en Mi API key.",
            REASON_REQUEST: f"El modelo {self.model} de {label} no aceptó el pedido. Probá con otro modelo en Mi API key.",
            REASON_UNAVAILABLE: f"{label} no está respondiendo en este momento. Probá de nuevo en unos minutos.",
        }[self.reason]

    def detail(self, lang: Optional[str]) -> dict[str, str]:
        """Cuerpo `detail` de la respuesta: plano, para que lo lea errors.js."""
        return {
            "error": USER_LLM_ERROR,
            "reason": self.reason,
            "provider": self.provider,
            "model": self.model,
            "message": self.message(lang),
        }


def find_user_llm_error(exc: Optional[BaseException]) -> Optional[UserLLMError]:
    """Busca un UserLLMError en la cadena de causas de una excepción.

    Los routers envuelven cualquier falla del LLM en un HTTPException 503
    genérico (`raise HTTPException(...) from exc`). En vez de tocar cada
    router, el handler de main.py mira la causa y, si fue la key del usuario,
    responde con su mensaje propio.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, UserLLMError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None
