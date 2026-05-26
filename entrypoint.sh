#!/bin/sh
# Entrypoint del container — corre migraciones antes de levantar la API.
# Esto garantiza que la DB esté siempre actualizada al arrancar, sin
# necesidad de correr alembic manualmente ni via SSH en CI.
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting API..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips "*"
