"""HTTP middleware: X-Request-ID + logging estructurado JSON por request."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.infrastructure.redis_client import get_redis
from app.shared.error_monitoring import record_5xx_and_maybe_alert
from app.shared.usage_ledger import (
    USAGE_HEADER,
    start_call_collection,
    usage_header_value,
)

logger = logging.getLogger("nexusai.access")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Agrega X-Request-ID a cada response y loguea una línea JSON de acceso.

    El request_id también se almacena en request.state.request_id para que
    los endpoints puedan incluirlo en sus propios logs contextuales
    (ej. chat/router.py logea course_id + user_id + tokens con el mismo id).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        # DATA-04: juntar las llamadas a proveedores de este pedido para el
        # encabezado X-NexusAI-Usage (ver app/shared/usage_ledger.py).
        calls = start_call_collection()

        try:
            response = await call_next(request)
        except Exception as exc:
            # Una excepción que no sea HTTPException (un bug no anticipado)
            # normalmente la resuelve Starlette's ServerErrorMiddleware, que
            # queda AFUERA de este middleware — el 5xx más grave (un crash
            # real) nunca pasaría por el chequeo de abajo ni por el log de
            # acceso (ver ADR-012, hallazgo de audit). Atajarla acá adentro
            # es lo único que garantiza que sí pase por los dos.
            logger.error(
                "Unhandled exception",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "error": str(exc),
                    "type": type(exc).__name__,
                },
                exc_info=True,
            )
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        # En un stream los encabezados salen antes que el consumo: el chat lo
        # manda en su evento `done`.
        is_stream = response.headers.get("content-type", "").startswith(
            "text/event-stream"
        )
        if calls and not is_stream:
            response.headers[USAGE_HEADER] = usage_header_value(calls)

        logger.info(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
                ensure_ascii=False,
            )
        )

        if response.status_code >= 500:
            try:
                redis = await get_redis()
                await record_5xx_and_maybe_alert(
                    redis, status_code=response.status_code, path=request.url.path
                )
            except Exception as exc:
                # No dejar que un fallo de alertas (p. ej. Redis caído) tumbe
                # la response real que ya se generó.
                logger.warning("record_5xx_and_maybe_alert falló: %s", exc)

        return response
