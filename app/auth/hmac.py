"""
Verificación HMAC SHA-256 para requests del plugin Moodle.

Patrón Hybrid PHP Proxy:
  Browser (React) → Moodle PHP → FastAPI → OpenAI/Gemini
                       ↑
                  Acá firmamos con HMAC

Ver investigacion/05-backend-fastapi/autenticacion-hmac.md para el contexto
arquitectónico completo y el código equivalente del lado PHP.

3 capas de defensa:
  1. Bearer API key (Authorization header)         → identidad del cliente
  2. HMAC SHA-256 de (timestamp + nonce + body)    → integridad del payload
  3. Anti-replay con nonce store en Redis          → prevenir replay attacks

Headers que el cliente PHP debe enviar:
  Authorization: Bearer <NEXUSAI_API_KEY>
  X-Timestamp:   <unix epoch en segundos>
  X-Nonce:       <UUID v4 único por request>
  X-Signature:   <hex_hmac_sha256(timestamp + nonce + body, SHARED_SECRET)>
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
import redis.asyncio as redis_async

from app.infrastructure.redis_client import get_redis
from app.shared.config import get_settings

# Prefijo de las keys de Redis para evitar colisión con otras features
# (rate limiting, cache de respuestas LLM, etc.).
_NONCE_KEY_PREFIX = "nexusai:hmac:nonce:"


async def verify_hmac(
    request: Request,
    authorization: Annotated[str, Header()],
    x_timestamp: Annotated[str, Header()],
    x_nonce: Annotated[str, Header()],
    x_signature: Annotated[str, Header()],
    redis: Annotated[redis_async.Redis, Depends(get_redis)],
) -> bytes:
    """
    FastAPI Dependency que valida una request firmada por el plugin Moodle.

    Uso:
        from fastapi import Depends
        from app.auth.hmac import verify_hmac

        @router.post("/api/v1/chat/echo")
        async def echo(_body: bytes = Depends(verify_hmac)):
            ...

    Returns:
        bytes: el body raw de la request, ya validado.
                Útil si el endpoint necesita el JSON antes de Pydantic
                (ej. logging, audit). FastAPI igual hace re-parse para los
                argumentos tipados del endpoint, así que esto es solo bonus.

    Raises:
        HTTPException 401: si cualquiera de las 3 capas falla.
    """
    settings = get_settings()

    # ----- Capa 1: Bearer API key -----
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )
    api_key = authorization[len("Bearer "):]
    if not hmac_lib.compare_digest(api_key, settings.nexusai_api_key):
        # `compare_digest` evita timing attacks (no leakea info por el largo
        # del prefix coincidente).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # ----- Capa 2a: timestamp dentro de la ventana -----
    try:
        ts = int(x_timestamp)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Timestamp",
        )

    now = int(time.time())
    if abs(now - ts) > settings.hmac_replay_window_sec:
        # Si el clock del server PHP de Moodle está muy desincronizado con
        # el server Python, esto explota. Ventana default: 300s (5 min).
        # En prod debería loggearse para detectar drift de NTP entre nodos.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Request expired or clock skew too large",
        )

    # ----- Capa 2b: firma HMAC del payload -----
    body = await request.body()

    # IMPORTANTE: el orden de concatenación tiene que ser EXACTAMENTE igual
    # que en el cliente PHP (ver backend_client::send_message en plugin/).
    # Cualquier cambio acá rompe TODA la integración. Es contrato.
    signed_string = (x_timestamp + x_nonce).encode("utf-8") + body

    expected_signature = hmac_lib.new(
        settings.nexusai_shared_secret.encode("utf-8"),
        signed_string,
        hashlib.sha256,
    ).hexdigest()

    if not hmac_lib.compare_digest(x_signature.lower(), expected_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )

    # ----- Capa 3: anti-replay con nonce store -----
    # Si el atacante captura una request válida (ej. snifeando red interna)
    # y la re-envía dentro de la ventana de 5 min, las capas 1 y 2 la
    # aprueban — la firma sigue siendo válida. Para prevenir eso, cada
    # request debe traer un nonce único, y nosotros lo guardamos en Redis
    # con TTL = ventana_temporal. Si el nonce ya existe, rechazamos.
    #
    # Usamos `set(NX, EX=...)` que es atómico: o lo crea o devuelve None.
    # Esto evita race conditions con dos requests llegando simultáneamente.
    nonce_key = f"{_NONCE_KEY_PREFIX}{x_nonce}"
    was_set = await redis.set(
        nonce_key,
        str(ts),                # value: timestamp original (útil para debug)
        nx=True,                # NX: solo si no existe
        ex=settings.hmac_replay_window_sec,  # EX: TTL en segundos
    )
    if not was_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Replay detected: nonce already used",
        )

    return body
