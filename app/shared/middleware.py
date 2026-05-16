"""HTTP middleware: X-Request-ID + logging estructurado JSON por request."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

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

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request_id

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

        return response
