"""
NexusAI — Backend FastAPI entrypoint.

Lifespan:
  - startup:  imprime config básica para verificar que las env vars cargaron OK
  - shutdown: cierra el pool de Redis y el engine de DB para terminar limpio

CORS:
  Por arquitectura (Hybrid PHP Proxy — ver ADR-001 e investigacion/05-backend-fastapi/),
  el navegador NUNCA habla directo con FastAPI. Solo el plugin Moodle PHP nos llama
  vía cURL (server-to-server, no hay CORS en juego). CORS desactivado en producción,
  whitelist mínima en dev.

Middleware (BACK-14):
  RequestIDMiddleware: agrega X-Request-ID a cada response y loguea
  una línea JSON de acceso estructurado por request.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import dispose_engine
from app.infrastructure.redis_client import close_redis
from app.shared.config import get_settings
from app.shared.middleware import RequestIDMiddleware

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"NexusAI API v{settings.app_version} starting...")
    print(f"ENV: {settings.env}")
    print(f"LLM model: {settings.llm_model}")

    yield

    print("NexusAI API shutting down...")
    await close_redis()
    await dispose_engine()


app = FastAPI(
    title="NexusAI API",
    version=settings.app_version,
    description="Academic AI assistant backend for Moodle",
    lifespan=lifespan,
)

# RequestIDMiddleware va primero para que X-Request-ID esté disponible en todos
# los handlers de error de FastAPI también.
app.add_middleware(RequestIDMiddleware)

if settings.env == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",
            "http://localhost:8080",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================
# Routers de feature
# ============================================================

from app.admin.router import router as admin_router          # noqa: E402
from app.chat.router import router as chat_router            # noqa: E402
from app.courses.router import router as courses_router      # noqa: E402
from app.documents.router import router as documents_router  # noqa: E402
from app.search.router import router as search_router        # noqa: E402

app.include_router(chat_router,      prefix="/api/v1/chat",      tags=["chat"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(admin_router,     prefix="/api/v1/admin",     tags=["admin"])
app.include_router(courses_router,   prefix="/api/v1/courses",   tags=["courses"])
app.include_router(search_router,    prefix="/api/v1/search",    tags=["search"])


# ============================================================
# Endpoints de servicio
# ============================================================

@app.get("/health")
async def health():
    """Liveness probe para Docker/Kubernetes."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "env": settings.env,
        "llm_model": settings.llm_model,
    }


@app.get("/")
async def root():
    return {"message": "NexusAI API running"}
