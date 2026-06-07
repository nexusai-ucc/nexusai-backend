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
from sqlalchemy import text

from app.db.session import dispose_engine, get_session_factory
from app.infrastructure.redis_client import close_redis
from app.shared.config import get_settings
from app.shared.middleware import RequestIDMiddleware

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


logger = logging.getLogger("nexusai.startup")


async def _recover_interrupted_documents() -> None:
    """Marca como 'error' los documentos que quedaron en 'pending' o 'indexing'.

    Las background tasks de indexación se pierden cuando el container se reinicia.
    Sin esta limpieza, esos registros bloquean re-subidas (collision check) y
    aparecen como "EN COLA" para siempre en la UI docente.
    """
    try:
        factory = get_session_factory()
        async with factory() as db:
            result = await db.execute(
                text("""
                    UPDATE documents
                    SET status        = 'error',
                        error_message = 'Indexación interrumpida al reiniciar el servidor. Eliminá y volvé a subir el archivo.'
                    WHERE status IN ('pending', 'indexing')
                """)
            )
            await db.commit()
            if result.rowcount:
                logger.info(
                    "Startup recovery: %d documento(s) con indexación interrumpida → error",
                    result.rowcount,
                )
    except Exception as exc:
        logger.warning("Startup recovery failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NexusAI API v%s starting (%s)", settings.app_version, settings.env)
    logger.info("LLM model: %s", settings.llm_model)

    await _recover_interrupted_documents()

    yield

    logger.info("NexusAI API shutting down...")
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
from app.gaps.router import router as gaps_router            # noqa: E402
from app.quiz.router import router as quiz_router            # noqa: E402
from app.search.router import router as search_router        # noqa: E402

app.include_router(chat_router,      prefix="/api/v1/chat",      tags=["chat"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(admin_router,     prefix="/api/v1/admin",     tags=["admin"])
app.include_router(courses_router,   prefix="/api/v1/courses",   tags=["courses"])
app.include_router(search_router,    prefix="/api/v1/search",    tags=["search"])
app.include_router(quiz_router,      prefix="/api/v1/quiz",      tags=["quiz"])
app.include_router(gaps_router,      prefix="/api/v1/gaps",      tags=["gaps"])


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
