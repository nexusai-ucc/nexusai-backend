"""
LLMProvider — abstracción para chat completions con retry automático.

Soporta cualquier proveedor compatible con el SDK de OpenAI cambiando
`LLM_BASE_URL` en el .env:

  - Gemini (default MVP):     https://generativelanguage.googleapis.com/v1beta/openai/
  - OpenAI (prod):            https://api.openai.com/v1
  - Ollama local (dev):       http://localhost:11434/v1
  - Groq:                     https://api.groq.com/openai/v1

Control del thinking (PERF-01): los flash 2.5+ de Gemini razonan internamente
antes de emitir el primer token, y ese razonamiento invisible domina la
latencia percibida. `LLM_REASONING_EFFORT` (default "none") se manda en todas
las llamadas; cada call-site lo puede pisar con `reasoning_effort=` — hoy solo
lo hace la generación de quiz/examen, con `LLM_REASONING_EFFORT_GENERATION`.
Con el valor "default"/"auto"/"" no se manda el parámetro y el comportamiento
vuelve a ser el previo a PERF-01.

Retry: 3 intentos con backoff 1s → 2s para errores transitorios
(RateLimitError, Timeout, ConnectionError, InternalServerError).
Los errores definitivos (AuthenticationError, BadRequestError) propagan de
inmediato — sin reintentos. Ojo: no reintentar y no ceder el paso al siguiente
eslabón son cosas distintas; ver _FALLBACK_TRIGGERS más abajo.

Fallback automático (INFRA-01 / issue #307): si el proveedor primario agota
sus 3 reintentos por cuota (429 RateLimitError) o caída del servidor (503
InternalServerError), y hay un proveedor secundario configurado vía
LLM_FALLBACK_API_KEY/LLM_FALLBACK_BASE_URL/LLM_FALLBACK_MODEL, se reintenta
automáticamente contra ese secundario antes de propagar el error al caller.
Sin esas 3 env vars, el fallback queda deshabilitado (comportamiento idéntico
al anterior).

Cadena de modelos intermedios (INFRA-03 / issue #343): Google AI Studio da
cuota gratuita POR MODELO, no por cuenta — agotar `gemini-3.5-flash` no
agota `gemini-3.1-flash-lite` ni `gemini-3.8-flash`. `LLM_INTERMEDIATE_MODELS`
(opcional, coma-separado) prueba esos modelos extra usando el MISMO client
del primario (misma API key, mismo base_url — no son otro proveedor, solo
otro modelo) antes de recién ahí pasar al proveedor secundario configurado.
Cadena resultante: primario → intermedios (en orden) → secundario. Vacío =
comportamiento idéntico a antes de INFRA-03.

Key propia del usuario (BYOK): si el pedido trae `user_llm` (ver
app/providers/user_llm.py), get_llm_provider() devuelve una copia del provider
con un eslabón más al final, el del usuario. Cadena resultante: primario →
intermedios → secundario → usuario. Con `user_only()` la cadena es solo el
eslabón del usuario (el chat lo usa cuando el usuario agotó su presupuesto).

Ver ADR-003 (decisión multi-provider) y ADR-004 (Gemini MVP / OpenAI prod).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, Optional, Union

import openai
from fastapi import Request
from openai import AsyncOpenAI

from app.shared.config import get_settings
from app.providers.user_llm import (
    UserLLMConfig,
    UserLLMError,
    client_for,
    reason_for,
    user_llm_from_body,
)
from app.shared.retry import async_retry
from app.shared.usage_ledger import (
    UsageRecord,
    as_int,
    llm_usage_numbers,
    provider_from_base_url,
    record_usage,
    schedule_usage,
    status_for_exception,
)

logger = logging.getLogger(__name__)

# Errores que disparan el fallback al siguiente eslabón de la cadena.
# Deliberadamente más angosto que _RETRYABLE_OPENAI de retry.py — timeouts y
# errores de conexión NO disparan fallback porque probablemente afecten a
# todos los eslabones por igual.
#
#   RateLimitError (429)       → cuota agotada (RESOURCE_EXHAUSTED en Gemini).
#   InternalServerError (503)  → modelo saturado o caído.
#   NotFoundError (404)        → el modelo ya no existe. Google retira modelos
#       viejos sin previo aviso: al 2026-09, gemini-2.0-flash y
#       gemini-2.5-flash-lite —ambos sugeridos por el .env.example previo a
#       PERF-01— devuelven 404. Sin este trigger, un eslabón intermedio muerto
#       no cedía el paso al siguiente: cortaba la cadena entera y le propagaba
#       un 503 al alumno.
#   BadRequestError (400)      → el modelo rechaza un parámetro de la request
#       (p. ej. `reasoning_effort`, que no todos los flash-lite aceptan). Es
#       "este modelo no puede servir esta request", así que corresponde probar
#       el siguiente. Contrapartida asumida: una request genuinamente mal
#       formada ahora se reintenta una vez por eslabón antes de fallar. Con la
#       config default (cadena de un solo eslabón) no cambia nada.
_FALLBACK_TRIGGERS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.NotFoundError,
    openai.BadRequestError,
)


class _UserLinkFailed(Exception):
    """Interno: falló el eslabón del usuario dentro de _run_completion_chain
    (método estático, sin acceso al provider); chat_completion lo convierte
    en UserLLMError. `institutional` es la falla de la facultad que hizo
    llegar la cadena hasta la key del usuario (None si no la hubo)."""

    def __init__(
        self, original: BaseException, institutional: Optional[BaseException]
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.institutional = institutional


# Reintentos por eslabón cuando la cadena tiene la key del usuario (BYOK):
#   - Eslabones de la facultad: sin reintentar timeouts ni conexión caída. Un
#     timeout ya tardó lo que el client espera (120 s): reintentarlo 3 veces
#     dejaba al pedido sin tiempo para llegar a la key del usuario antes de que
#     el plugin corte. Pasa directo a la key (ver _next_index).
#   - Eslabón del usuario: solo el 5xx. Ni el 429 (una cuenta sin saldo no se
#     arregla esperando unos segundos) ni timeouts/conexión (mismo motivo de
#     arriba; APITimeoutError es subclase de APIConnectionError).
_RETRY_INSTITUTION_WITH_USER: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
)
_RETRY_USER_LINK: tuple[type[BaseException], ...] = (openai.InternalServerError,)

_BAD_REQUEST = (openai.BadRequestError, openai.UnprocessableEntityError)


# Valores de reasoning_effort que significan "no mandes el parámetro, que el
# modelo decida" — o sea, el comportamiento previo a PERF-01.
_EFFORT_UNSET = frozenset({"", "default", "auto"})


@dataclass
class CompletionResult:
    """Resultado de una chat completion con texto y métricas de tokens.

    `model`, `provider` y `fallback` dicen qué eslabón de la cadena respondió
    de verdad (COST-01): con fallback, no es el modelo configurado primero.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    model: str = ""
    provider: str = ""
    fallback: bool = False
    # True si respondió el eslabón del usuario (BYOK): no cuenta contra su
    # presupuesto ni contra el costo de la facultad.
    byok: bool = False


@dataclass(frozen=True)
class StreamToken:
    """Un chunk de texto del LLM en modo streaming."""

    text: str


@dataclass(frozen=True)
class StreamUsage:
    """Conteo final de tokens del stream (último chunk cuando include_usage=True)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0
    model: str = ""
    provider: str = ""
    fallback: bool = False
    byok: bool = False


@dataclass(frozen=True)
class StreamLink:
    """Qué eslabón abrió el stream; se informa por `on_open` antes del primer token."""

    model: str
    provider: str
    fallback: bool
    byok: bool


StreamChunk = Union[StreamToken, StreamUsage]


class LLMProvider:
    """
    Wrapper async sobre el SDK de OpenAI configurado para el proveedor activo.

    No es un singleton estricto — `get_llm_provider()` (la dependency de FastAPI)
    cachea una instancia con `lru_cache`. Para tests, instanciar directo es ok.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model: str = model or settings.llm_model

        # Default de thinking para todas las llamadas — ver config.py::
        # llm_reasoning_effort. Cada call-site lo puede pisar pasando
        # `reasoning_effort=` a chat_completion/chat_stream.
        self.reasoning_effort: str = (
            reasoning_effort
            if reasoning_effort is not None
            else settings.llm_reasoning_effort
        )
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=base_url or settings.llm_base_url,
            timeout=120.0,
            max_retries=0,  # Retries manejados por async_retry, no por el SDK.
        )
        self.provider_name: str = (
            provider_from_base_url(base_url or settings.llm_base_url) or ""
        )

        # Proveedor secundario opcional — ver _FALLBACK_TRIGGERS arriba.
        self.fallback_model: Optional[str] = settings.llm_fallback_model
        self.fallback_client: Optional[AsyncOpenAI] = None
        if (
            settings.llm_fallback_api_key
            and settings.llm_fallback_base_url
            and settings.llm_fallback_model
        ):
            self.fallback_client = AsyncOpenAI(
                api_key=settings.llm_fallback_api_key,
                base_url=settings.llm_fallback_base_url,
                timeout=120.0,
                max_retries=0,
            )
        self.fallback_provider_name: str = (
            provider_from_base_url(settings.llm_fallback_base_url) or ""
            if self.fallback_client
            else ""
        )

        # Modelos intermedios opcionales (INFRA-03) — mismo client que el
        # primario, ver _fallback_chain().
        self.intermediate_models: list[str] = [
            m.strip()
            for m in (settings.llm_intermediate_models or "").split(",")
            if m.strip()
        ]

        # Eslabón del usuario (BYOK) — vacío en la instancia compartida; solo
        # lo tienen las copias por pedido de with_user_llm().
        self.user_client: Optional[AsyncOpenAI] = None
        self.user_model: str = ""
        self.user_provider_name: str = ""
        self._only_user: bool = False

    # ------------------------------------------------------------
    # Key propia del usuario (BYOK) — ver app/providers/user_llm.py
    # ------------------------------------------------------------

    def with_user_llm(self, config: UserLLMConfig) -> "LLMProvider":
        """Copia de este provider con el eslabón del usuario al final.

        Copia superficial: comparte los clients de la facultad con la
        instancia cacheada (no abre conexiones nuevas) y nunca la modifica.
        """
        clone = copy.copy(self)
        clone.user_client = client_for(config.provider, config.api_key)
        clone.user_model = config.model
        clone.user_provider_name = config.provider
        clone._only_user = False
        return clone

    @property
    def has_user_llm(self) -> bool:
        return self.user_client is not None

    @property
    def is_user_only(self) -> bool:
        return self._only_user and self.user_client is not None

    def user_only(self) -> "LLMProvider":
        """Copia cuya cadena es solo el eslabón del usuario.

        La usa el chat cuando el usuario agotó su presupuesto de tokens: en
        vez de cortar con 429, sigue respondiendo con su propia key.
        """
        if self.user_client is None:
            raise RuntimeError("user_only() sin key del usuario")
        clone = copy.copy(self)
        clone._only_user = True
        return clone

    def institutional(self) -> "LLMProvider":
        """Copia sin el eslabón del usuario, para lo que no debe depender de
        un modelo que eligió el alumno (la moderación de contenido)."""
        if self.user_client is None:
            return self
        clone = copy.copy(self)
        clone.user_client = None
        clone.user_model = ""
        clone.user_provider_name = ""
        clone._only_user = False
        return clone

    def _is_user_client(self, client: AsyncOpenAI) -> bool:
        return self.user_client is not None and client is self.user_client

    @staticmethod
    def _user_link_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Parámetros para el eslabón del usuario.

        `reasoning_effort` está pensado para los modelos de la facultad (ver
        PERF-01) y muchos modelos de otros proveedores lo rechazan con 400:
        al eslabón del usuario no se le manda.
        """
        extra_body = kwargs.get("extra_body")
        if not extra_body or "reasoning_effort" not in extra_body:
            return kwargs
        trimmed = {k: v for k, v in extra_body.items() if k != "reasoning_effort"}
        out = {k: v for k, v in kwargs.items() if k != "extra_body"}
        if trimmed:
            out["extra_body"] = trimmed
        return out

    def _with_reasoning_effort(
        self,
        kwargs: dict[str, Any],
        override: Optional[str] = None,
    ) -> dict[str, Any]:
        """Agrega `reasoning_effort` al body de la request (PERF-01).

        Va dentro de `extra_body` y no como kwarg nombrado porque el SDK de
        OpenAI pineado en requirements.txt (1.51.0) todavía no lo expone como
        parámetro tipado — `extra_body` lo manda igual, tal cual, en el JSON.

        Respeta lo que ya venga puesto: si un call-site arma su propio
        `extra_body` con un `reasoning_effort` adentro, no se lo pisa.
        """
        effort = override if override is not None else self.reasoning_effort
        if not effort or effort.strip().lower() in _EFFORT_UNSET:
            return kwargs

        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("reasoning_effort", effort.strip().lower())
        return {**kwargs, "extra_body": extra_body}

    def _fallback_chain(self) -> list[tuple[AsyncOpenAI, str]]:
        """Cadena completa a intentar en orden (INFRA-03):

          primario → modelos intermedios (mismo client, ver __init__) →
          proveedor secundario final (otro client, si está configurado).

        Sin LLM_INTERMEDIATE_MODELS ni fallback configurado, devuelve una
        lista de un solo elemento — mismo comportamiento que antes de
        INFRA-01/INFRA-03 (un solo intento, sin fallback).
        """
        if self._only_user and self.user_client is not None:
            return [(self.user_client, self.user_model)]
        chain = [(self.client, self.model)]
        chain.extend((self.client, m) for m in self.intermediate_models)
        if self.fallback_client:
            # Invariante de __init__: fallback_client solo se crea si
            # llm_fallback_model también estaba seteado.
            assert self.fallback_model is not None
            chain.append((self.fallback_client, self.fallback_model))
        if self.user_client is not None:
            # BYOK: siempre el último — se usa solo si la facultad no pudo.
            chain.append((self.user_client, self.user_model))
        return chain

    def _provider_of(self, client: AsyncOpenAI) -> str:
        """Proveedor de un eslabón: los intermedios comparten client con el primario."""
        if self._is_user_client(client):
            return self.user_provider_name
        if self.fallback_client is not None and client is self.fallback_client:
            return self.fallback_provider_name
        return self.provider_name

    def _user_index(self, chain: list[tuple[AsyncOpenAI, str]]) -> Optional[int]:
        """Posición del eslabón del usuario en la cadena (siempre el último)."""
        if chain and self._is_user_client(chain[-1][0]):
            return len(chain) - 1
        return None

    async def _record_link_failure(
        self,
        chain: list[tuple[AsyncOpenAI, str]],
        index: int,
        exc: BaseException,
    ) -> None:
        """Registra el intento fallido de un eslabón (COST-01): así queda
        medido cuántas veces entra el fallback y por qué."""
        client, model = chain[index]
        await record_usage(
            UsageRecord(
                kind="llm",
                status=status_for_exception(exc),
                provider=self._provider_of(client),
                model=model,
                fallback=index > 0,
                byok=self._is_user_client(client),
            )
        )

    def _user_failure(
        self,
        chain: list[tuple[AsyncOpenAI, str]],
        exc: BaseException,
        institutional: Optional[BaseException] = None,
    ) -> BaseException:
        """Qué excepción sube cuando falló el eslabón del usuario.

        Normalmente un UserLLMError con el motivo que verá el usuario, que
        además guarda la falla de la facultad (`institutional_error`) para que
        la alerta de cuota/caída de la facultad se siga disparando.

        Si la facultad Y el usuario rechazaron el mismo pedido como inválido
        (400/422), el problema es el pedido, no la key: sube el error de la
        facultad, como antes de BYOK, en vez de decirle al usuario que cambie
        de modelo.
        """
        if isinstance(exc, _BAD_REQUEST) and isinstance(institutional, _BAD_REQUEST):
            return institutional
        return UserLLMError(
            reason_for(exc),
            self.user_provider_name,
            chain[-1][1],
            institutional_error=institutional,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Devuelve la respuesta completa del LLM junto con el conteo de tokens.

        Reintenta automáticamente hasta 3 veces en errores transitorios
        (rate limit, timeout, servidor caído). Propaga inmediatamente en errores
        definitivos (auth inválida, bad request, etc.).

        Args:
            messages: lista en formato ChatML
                [{"role": "system|user|assistant", "content": "..."}, ...]
            reasoning_effort: pisa el default de `LLM_REASONING_EFFORT` solo
                para esta llamada ("none" | "low" | "medium" | "high").
            **kwargs: parámetros adicionales del SDK (temperature, max_tokens, etc.)

        Returns:
            CompletionResult con el texto de respuesta y los token counts del LLM.

        Si un eslabón de la cadena (ver _fallback_chain) agota sus reintentos
        por cuota/servidor caído (ver _FALLBACK_TRIGGERS), pasa automáticamente
        al siguiente eslabón antes de propagar.

        Raises:
            openai.* — errores no-retryables, o el del último eslabón de la
            cadena si todos fallan.
            UserLLMError — si la cadena llegó al eslabón del usuario y ese
            también falló.
        """
        chain = self._fallback_chain()
        started = time.perf_counter()

        async def on_link_failure(index: int, exc: BaseException) -> None:
            await self._record_link_failure(chain, index, exc)

        user_index = self._user_index(chain)
        try:
            response, index = await self._run_completion_chain(
                chain,
                messages,
                on_link_failure=on_link_failure,
                user_index=user_index,
                **self._with_reasoning_effort(kwargs, reasoning_effort),
            )
        except _UserLinkFailed as failed:
            raise self._user_failure(
                chain, failed.original, failed.institutional
            ) from failed.original
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt, completion, cached = llm_usage_numbers(usage)
        client, model = chain[index]
        provider = self._provider_of(client)
        byok = self._is_user_client(client)
        await record_usage(
            UsageRecord(
                kind="llm",
                provider=provider,
                model=model,
                fallback=index > 0,
                byok=byok,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_prompt_tokens=cached,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        )
        return CompletionResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cached_prompt_tokens=cached,
            model=model,
            provider=provider,
            fallback=index > 0,
            byok=byok,
        )

    @staticmethod
    async def _create_completion(
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, str]],
        *,
        retryable: Optional[tuple[type[BaseException], ...]] = None,
        **kwargs: Any,
    ) -> Any:
        """Chat completion no-streaming con retry, contra el client/model dados.

        `retryable`: qué errores reintentar; None = los de siempre (retry.py).
        """
        retry_kwargs: dict[str, Any] = {}
        if retryable is not None:
            retry_kwargs["retryable"] = retryable
        return await async_retry(
            # El SDK espera TypedDicts con "role" literal; dicts planos
            # funcionan bien en runtime (se serializan igual) pero mypy no
            # los matchea estructuralmente.
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                stream=False,
                **kwargs,
            ),
            **retry_kwargs,
        )

    @staticmethod
    def _retryable_for(
        i: int, user_index: Optional[int]
    ) -> Optional[tuple[type[BaseException], ...]]:
        """Errores a reintentar en el eslabón `i` (ver _RETRY_INSTITUTION_WITH_USER)."""
        if user_index is None:
            return None
        if i == user_index:
            return _RETRY_USER_LINK
        return _RETRY_INSTITUTION_WITH_USER

    @staticmethod
    def _next_index(
        i: int, exc: BaseException, chain_len: int, user_index: Optional[int]
    ) -> Optional[int]:
        """A qué eslabón pasar después de que falló el `i`. None = propagar.

        Sin key del usuario es el comportamiento de siempre: solo los
        _FALLBACK_TRIGGERS pasan al siguiente, y el último propaga.

        Con key del usuario, una falla de la facultad que no es un trigger
        (timeout, conexión caída, key de la facultad rechazada) salta directo
        al eslabón del usuario: los eslabones intermedios comparten client con
        el primario y probablemente fallen igual, pero el del usuario es otro
        proveedor y otra cuenta. Errores que no son del SDK (bugs) propagan.
        """
        if user_index is not None and i == user_index:
            return None
        if isinstance(exc, _FALLBACK_TRIGGERS):
            return i + 1 if i < chain_len - 1 else None
        if user_index is not None and isinstance(exc, openai.APIError):
            return user_index
        return None

    @staticmethod
    async def _run_completion_chain(
        chain: list[tuple[AsyncOpenAI, str]],
        messages: list[dict[str, str]],
        *,
        on_link_failure: Optional[Any] = None,
        user_index: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """Recorre la cadena en orden (ver _fallback_chain). Cada eslabón ya
        tiene sus propios 3 reintentos vía _create_completion/async_retry —
        recién si esos 3 fallan por _FALLBACK_TRIGGERS pasa al siguiente
        eslabón. Con un solo eslabón (caso default sin fallback configurado),
        el comportamiento es idéntico al de antes de INFRA-01/INFRA-03.

        Si CUALQUIER eslabón agota cuota (RateLimitError) antes de llegar al
        final de la cadena, esa excepción se guarda y se prioriza sobre la
        del último eslabón (ver ADR-012): sin esto, un primario que agota
        cuota pero un eslabón intermedio que ya no existe (NotFoundError) o
        un secundario caído (InternalServerError) hace que la excepción que
        llega al caller no sea RateLimitError — y record_llm_failure_and_maybe_alert
        (app/shared/error_monitoring.py) nunca dispara la alerta específica
        de cuota agotada, la más urgente/accionable de las 4.

        `user_index`: posición del eslabón del usuario (BYOK), o None. Si ese
        eslabón falla, se levanta _UserLinkFailed con la falla del usuario y
        la de la facultad que llevó hasta ahí (la cuota agotada si la hubo).

        Devuelve la response y el índice del eslabón que respondió.
        `on_link_failure(index, exc)` se awaitea por cada eslabón que falla
        con un error que corta o pasa al siguiente (registro de consumo)."""
        quota_exhausted: Optional[openai.RateLimitError] = None
        institutional: Optional[BaseException] = None
        i = 0
        while i < len(chain):
            client, model = chain[i]
            link_kwargs = (
                LLMProvider._user_link_kwargs(kwargs) if i == user_index else kwargs
            )
            try:
                response = await LLMProvider._create_completion(
                    client,
                    model,
                    messages,
                    retryable=LLMProvider._retryable_for(i, user_index),
                    **link_kwargs,
                )
                return response, i
            except Exception as exc:
                if isinstance(exc, openai.RateLimitError):
                    quota_exhausted = quota_exhausted or exc
                if on_link_failure is not None:
                    await on_link_failure(i, exc)
                if i == user_index:
                    raise _UserLinkFailed(exc, institutional) from exc
                nxt = LLMProvider._next_index(i, exc, len(chain), user_index)
                if nxt is not None and nxt == user_index:
                    institutional = quota_exhausted or exc
                if nxt is None:
                    if (
                        isinstance(exc, _FALLBACK_TRIGGERS)
                        and not isinstance(exc, openai.RateLimitError)
                        and quota_exhausted is not None
                    ):
                        raise quota_exhausted from exc
                    raise
                logger.warning(
                    "LLM fallback activado: %s agotado (%s: %s). Pasando a %s.",
                    model,
                    type(exc).__name__,
                    exc,
                    chain[nxt][1] if nxt != user_index else "la key del usuario",
                )
                i = nxt
        raise RuntimeError("Cadena de modelos vacía")

    async def _create_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Abre un stream recorriendo la cadena de fallback en orden (ver
        _fallback_chain), pasando al siguiente eslabón si la apertura falla
        por cuota/servidor caído.

        Nota: el fallback solo cubre la apertura del stream. Si la conexión se
        corta a mitad de la iteración (después de ya haber yieldeado texto),
        no se reintenta — mismo comportamiento que antes de INFRA-01, porque
        ya se envió output parcial al caller y no se puede "deshacer".

        Devuelve el stream y el índice del eslabón que lo abrió. Cada
        apertura fallida queda registrada (COST-01). Si falla el eslabón del
        usuario, levanta UserLLMError.
        """
        chain = self._fallback_chain()
        user_index = self._user_index(chain)
        quota_exhausted: Optional[openai.RateLimitError] = None
        institutional: Optional[BaseException] = None
        i = 0
        while i < len(chain):
            client, model = chain[i]
            link_kwargs = self._user_link_kwargs(kwargs) if i == user_index else kwargs
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    stream=True,
                    **link_kwargs,
                )
                return stream, i
            except Exception as exc:
                if isinstance(exc, openai.RateLimitError):
                    quota_exhausted = quota_exhausted or exc
                await self._record_link_failure(chain, i, exc)
                if i == user_index:
                    raise self._user_failure(chain, exc, institutional) from exc
                nxt = self._next_index(i, exc, len(chain), user_index)
                if nxt is not None and nxt == user_index:
                    institutional = quota_exhausted or exc
                if nxt is None:
                    # Ver el comentario equivalente en _run_completion_chain:
                    # priorizar la cuota agotada de un eslabón anterior sobre
                    # la falla puntual del último eslabón.
                    if (
                        isinstance(exc, _FALLBACK_TRIGGERS)
                        and not isinstance(exc, openai.RateLimitError)
                        and quota_exhausted is not None
                    ):
                        raise quota_exhausted from exc
                    raise
                logger.warning(
                    "LLM fallback activado (stream): %s agotado (%s: %s). Pasando a %s.",
                    model,
                    type(exc).__name__,
                    exc,
                    chain[nxt][1] if nxt != user_index else "la key del usuario",
                )
                i = nxt
        raise RuntimeError("Cadena de modelos vacía")

    async def _record_stream(
        self,
        index: int,
        started: float,
        status: str,
        usage: Optional[StreamUsage],
        cancelled: bool,
    ) -> None:
        """Registra un stream terminado, cortado o fallido (COST-01)."""
        client, model = self._fallback_chain()[index]
        record = UsageRecord(
            kind="llm",
            status=status,
            provider=self._provider_of(client),
            model=model,
            fallback=index > 0,
            byok=self._is_user_client(client),
            prompt_tokens=as_int(usage.prompt_tokens) if usage else 0,
            completion_tokens=as_int(usage.completion_tokens) if usage else 0,
            cached_prompt_tokens=as_int(usage.cached_prompt_tokens) if usage else 0,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        if cancelled:
            # El stream se está cerrando porque el cliente cortó: no se
            # puede esperar dentro de la cancelación, se registra aparte.
            schedule_usage(record)
        else:
            await record_usage(record)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming de chunks de texto (útil para SSE hacia el frontend).

        El streaming no se reintenta automáticamente: si la conexión se corta
        a mitad del stream, el error propaga al caller. Para el chat MVP
        (no-streaming), usar `chat_completion`.

        Yields:
            Strings con incrementos de texto. Algunos chunks pueden ser "" —
            el caller debe ignorarlos al armar SSE.
        """
        started = time.perf_counter()
        stream, index = await self._create_stream(
            messages, **self._with_reasoning_effort(kwargs, reasoning_effort)
        )
        chain = self._fallback_chain()
        byok = self._is_user_client(chain[index][0])

        status, cancelled = "ok", False
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True
            raise
        except Exception as exc:
            status = "error"
            if byok:
                raise self._user_failure(chain, exc) from exc
            raise
        finally:
            await self._record_stream(index, started, status, None, cancelled)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        *,
        reasoning_effort: Optional[str] = None,
        on_open: Optional[Callable[[StreamLink], None]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming con conteo de tokens al final.

        Yieldea StreamToken por cada chunk de texto. Cuando el provider lo soporta
        (Gemini compat OpenAI / OpenAI nativo), al final del stream yieldea un
        único StreamUsage con los token counts del prompt + completion. Útil para
        persistir métricas en la DB después del streaming.

        `on_open(link)` se llama apenas se abre el stream, con el eslabón que
        lo abrió: el chat lo necesita para saber si respondió la key del
        usuario aunque el stream se corte antes del StreamUsage final.
        """
        started = time.perf_counter()
        stream, index = await self._create_stream(
            messages,
            stream_options={"include_usage": True},
            **self._with_reasoning_effort(kwargs, reasoning_effort),
        )
        chain = self._fallback_chain()
        client, model = chain[index]
        provider = self._provider_of(client)
        byok = self._is_user_client(client)
        if on_open is not None:
            on_open(
                StreamLink(
                    model=model, provider=provider, fallback=index > 0, byok=byok
                )
            )

        usage_seen: Optional[StreamUsage] = None
        status, cancelled = "ok", False
        try:
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield StreamToken(text=delta.content)
                # En el último chunk con include_usage=True, viene el usage poblado.
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    _, _, cached = llm_usage_numbers(u)
                    usage_seen = StreamUsage(
                        prompt_tokens=u.prompt_tokens or 0,
                        completion_tokens=u.completion_tokens or 0,
                        total_tokens=u.total_tokens or 0,
                        cached_prompt_tokens=cached,
                        model=model,
                        provider=provider,
                        fallback=index > 0,
                        byok=byok,
                    )
                    yield usage_seen
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True
            raise
        except Exception as exc:
            status = "error"
            if byok:
                raise self._user_failure(chain, exc) from exc
            raise
        finally:
            await self._record_stream(index, started, status, usage_seen, cancelled)


# ============================================================
# FastAPI Dependency
# ============================================================


@lru_cache(maxsize=1)
def _cached_provider() -> LLMProvider:
    return LLMProvider()


async def get_llm_provider(request: Request) -> LLMProvider:
    """
    FastAPI Dependency. Inyecta el LLMProvider en endpoints y servicios.

    Si el pedido trae la key propia del usuario (`user_llm` en el cuerpo
    firmado, ver app/providers/user_llm.py), devuelve una copia del provider
    con ese eslabón al final de la cadena. Sin key, la instancia compartida
    de siempre. El cuerpo ya lo leyó verify_hmac: Starlette lo tiene en memoria.

    Uso:
        from fastapi import Depends
        from app.providers.llm import LLMProvider, get_llm_provider

        @router.post("/chat")
        async def chat(
            payload: ChatRequest,
            llm: LLMProvider = Depends(get_llm_provider),
        ):
            result = await llm.chat_completion([
                {"role": "user", "content": payload.question}
            ])
            return {"answer": result.text, "tokens": result.total_tokens}
    """
    provider = _cached_provider()
    if request.method != "POST" or not get_settings().user_llm_enabled:
        return provider
    config = user_llm_from_body(await request.body())
    return provider.with_user_llm(config) if config is not None else provider
