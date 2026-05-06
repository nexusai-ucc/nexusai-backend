"""
NexusAI — Backend FastAPI entrypoint.

Lifespan:
  - startup:  imprime config básica para verificar que las env vars cargaron OK
  - shutdown: cierra el pool de Redis y el engine de DB para terminar limpio

CORS:
  Por arquitectura (Hybrid PHP Proxy — ver ADR-001 e investigacion/05-backend-fastapi/),
  el navegador NUNCA habla directo con FastAPI. Solo el plugin Moodle PHP nos llama
  vía cURL (server-to-server, no hay CORS en juego). Por eso CORS queda DESACTIVADO
  para producción y solo permitimos localhost en dev por si alguien quiere testear
  con curl/Postman desde su máquina.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import dispose_engine
from app.infrastructure.redis_client import close_redis
from app.shared.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- Startup -----
    print(f"NexusAI API v{settings.app_version} starting...")
    print(f"ENV: {settings.env}")
    print(f"LLM model: {settings.llm_model}")

    yield

    # ----- Shutdown -----
    # Cerrar conexiones limpias evita warnings tipo "unclosed connection"
    # y previene que docker-compose tarde 10s en parar el container esperando
    # timeouts del pool.
    print("NexusAI API shutting down...")
    await close_redis()
    await dispose_engine()


app = FastAPI(
    title="NexusAI API",
    version=settings.app_version,
    description="Academic AI assistant backend for Moodle",
    lifespan=lifespan,
)

# CORS solo en dev. Ver docstring del módulo para el por qué.
if settings.env == "development":
    app.add_middleware(
        CORSMiddleware,
        # Whitelist explícita en lugar de "*" — incluso en dev.
        allow_origins=[
            "http://localhost:8000",   # Moodle dev (moodle-docker default)
            "http://localhost:8080",   # Moodle dev (alt port del docker-compose raíz)
            "http://localhost:5173",   # Vite dev server (por si alguien lo usa)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================
# Routers de feature
# ============================================================
# Cada feature (chat, documents, analytics) registra su router acá.

from app.chat.router import router as chat_router  # noqa: E402
from app.documents.router import router as documents_router  # noqa: E402

app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])


# ============================================================
# Endpoints de servicio (no van a un router de feature)
# ============================================================

@app.get("/health")
async def health():
    """
    Healthcheck para Docker/Kubernetes. Liveness probe.

    NO valida que Redis o Postgres estén OK — eso sería un readiness probe
    aparte (TODO: agregar /ready cuando metamos la DB real).
    """
    return {
        "status": "ok",
        "version": settings.app_version,
        "env": settings.env,
        "llm_model": settings.llm_model,
    }


@app.get("/")
async def root():
    return {"message": "NexusAI API running"}
