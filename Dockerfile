# NexusAI — Backend FastAPI
#
# Imagen Python slim, multi-stage para minimizar tamaño y attack surface.
# Etapas:
#   1. builder — instala deps en una venv aislada
#   2. runtime — copia solo la venv + el código, sin toolchain de compilación
#
# Resultado: imagen final ~250MB en lugar de ~1GB con la imagen full.
#
# Uso típico (docker-compose lo invoca):
#   docker build -t nexusai/api:dev .
#   docker run -p 8000:8000 --env-file .env nexusai/api:dev

# ============================================================
# Etapa 1 — builder
# ============================================================
FROM python:3.11-slim AS builder

# Variables de entorno para Python en containers:
#   - PYTHONDONTWRITEBYTECODE=1: no escribir .pyc (no aporta nada en container)
#   - PYTHONUNBUFFERED=1: stdout/stderr sin buffer (logs en tiempo real)
#   - PIP_NO_CACHE_DIR=1: no guardar cache de pip (imagen más chica)
#   - PIP_DISABLE_PIP_VERSION_CHECK=1: skip el check de update en cada install
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies de sistema necesarias para compilar wheels que no son puros
# (asyncpg, cryptography, etc. necesitan gcc + headers).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear virtualenv aislada para que la copiemos limpia a la etapa runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar deps. Se copia primero requirements.txt para que Docker cachee
# esta layer mientras no cambien las deps (changes a app/ no invalidan esto).
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ============================================================
# Etapa 2 — runtime
# ============================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Solo lo mínimo para runtime: libpq (cliente Postgres) + curl para healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar la venv ya construida (sin headers ni gcc, pesa mucho menos).
COPY --from=builder /opt/venv /opt/venv

# Crear usuario non-root. Correr como root en producción es un riesgo: si
# alguien compromete el container, escala más fácil.
RUN groupadd --system --gid 1001 nexusai \
    && useradd --system --uid 1001 --gid nexusai --no-create-home nexusai

WORKDIR /app
COPY --chown=nexusai:nexusai app/ ./app/
COPY --chown=nexusai:nexusai migrations/ ./migrations/

USER nexusai

EXPOSE 8000

# Healthcheck que pega contra /health (definido en app/main.py).
# Docker compose lo usa para esperar a que el API esté listo antes de
# considerar el container "healthy".
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn directo, sin gunicorn — para dev/MVP es suficiente.
# En producción evaluar gunicorn + uvicorn workers (4-8 workers) si hay
# suficiente tráfico que justifique más concurrencia.
#
# --proxy-headers: respeta los X-Forwarded-* de Moodle (que está delante)
# --forwarded-allow-ips=*: confía en cualquier proxy en la red interna
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
