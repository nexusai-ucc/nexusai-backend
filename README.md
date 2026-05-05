# services/api — Backend FastAPI de NexusAI

Backend Python que orquesta el pipeline RAG de NexusAI: recibe consultas firmadas
con HMAC desde el plugin Moodle, persiste sesiones de chat, hace retrieval
semántico contra pgvector y genera respuestas con el LLM activo (Gemini Flash en
MVP, OpenAI en producción).

> Para entender la arquitectura completa del sistema (no solo el backend),
> empezá por [`docs/architecture.md`](../../docs/architecture.md).

## Estado al 5 May 2026

| Pieza | Estado |
|---|---|
| FastAPI + lifespan + healthcheck | ✅ |
| Dockerfile multi-stage + non-root user | ✅ |
| Autenticación HMAC SHA-256 (3 capas) | ✅ |
| LLMProvider y EmbeddingProvider multi-vendor | ✅ |
| DB session async (SQLAlchemy 2.0 + asyncpg) | ✅ |
| Redis client async + nonce store HMAC | ✅ |
| Schema pgvector + Alembic con migración inicial | ✅ |
| Endpoint `POST /api/v1/chat/echo` (mock pre-RAG) | ✅ |
| Endpoint `POST /api/v1/chat/messages` con LLM real | ✅ |
| Pipeline RAG (extractor + chunker + indexer) | ✅ módulos · ⏳ endpoint para dispararlo |
| Retrieval semántico en `/messages` | ⏳ Sprint 2 |
| Endpoint `POST /api/v1/documents` (upload + indexación) | ⏳ Sprint 2 |
| Tests pytest (HMAC + providers) | ✅ 18 tests |
| Tests para chunker/extractor/pipeline/modelos | ⏳ Sprint 2 |

## Stack

- **Python** 3.11
- **FastAPI** 0.115 + Uvicorn
- **SQLAlchemy** 2.0 async + **asyncpg** + **Alembic**
- **PostgreSQL** 16 + **pgvector** 0.3.5
- **Redis** 7 (anti-replay HMAC + cache)
- **OpenAI SDK** (compat con Gemini, Groq, Ollama, etc.) — abstracción multi-provider
- **pytest** + **pytest-asyncio** para tests
- **ruff** para lint/format
- **tiktoken** para chunking
- **pdfplumber** para extracción de PDFs

## Estructura

```
services/api/
├── Dockerfile                  # Multi-stage, non-root, healthcheck
├── .dockerignore
├── alembic.ini                 # Config de migraciones (la URL la inyecta env.py)
├── pytest.ini                  # asyncio_mode = auto + markers custom
├── requirements.txt            # 18 deps pinneadas
├── app/
│   ├── main.py                 # FastAPI app + lifespan + CORS dev-only
│   ├── auth/
│   │   └── hmac.py             # Dependency verify_hmac (3 capas)
│   ├── chat/
│   │   ├── router.py           # POST /messages, /echo
│   │   └── schemas.py          # Pydantic ChatRequest/ChatResponse/MessageOut
│   ├── db/
│   │   ├── session.py          # async engine + get_db Dependency + Base
│   │   └── models.py           # 4 modelos ORM con UUIDs
│   ├── documents/
│   │   ├── extractor.py        # pdfplumber wrapper
│   │   ├── chunker.py          # sliding window 512/64 con tiktoken
│   │   └── pipeline.py         # orquestador: extract → chunk → embed → save
│   ├── infrastructure/
│   │   └── redis_client.py     # pool async + lifecycle hooks
│   ├── providers/
│   │   ├── llm.py              # LLMProvider (chat_completion, chat_stream)
│   │   └── embeddings.py       # EmbeddingProvider (embed, embed_many)
│   └── shared/
│       └── config.py           # Pydantic Settings (env vars validadas al startup)
├── migrations/
│   ├── env.py                  # Alembic env (lee DATABASE_URL del .env)
│   └── versions/
│       └── 001_initial_schema.py   # 4 tablas + pgvector + índices
└── tests/
    ├── conftest.py             # env vars de test + fixtures (Redis mock, etc.)
    ├── test_hmac.py            # 9 tests cubriendo todos los caminos de fallo
    └── test_providers.py       # 9 tests con SDK mockeado
```

## Setup local — quick start

El backend está pensado para correr **dentro de Docker Compose**. El stack
completo (Postgres + Redis + API) se levanta con un solo comando.

### 1. Pre-requisitos

- Docker Desktop corriendo
- API key de Gemini gratis: https://aistudio.google.com/apikey

### 2. Configurar `.env` (en la raíz del repo, no en `services/api/`)

```bash
cd ~/Documents/NexusAI/nexusAI/NexusAI
cp .env.example .env

# Generar HMAC secrets
echo "NEXUSAI_API_KEY=$(openssl rand -hex 32)" >> .env.tmp
echo "NEXUSAI_SHARED_SECRET=$(openssl rand -hex 32)" >> .env.tmp
# Copiá esos valores a las líneas correspondientes en .env

# Pegar tu Gemini API key:
# LLM_API_KEY=AIzaSy...
# EMBEDDING_API_KEY=AIzaSy...   (la misma)
```

> ⚠️ **No usar `gemini-2.0-flash` con cuentas Gemini nuevas** — tienen `limit: 0`
> en ese modelo. Usar `gemini-2.5-flash` (default actual). Verificado el 4 May.

### 3. Levantar el stack

```bash
./scripts/setup-e2e.sh
```

Lo que hace el script:

1. Valida que `.env` tenga todos los placeholders reemplazados
2. `docker compose up -d postgres redis api`
3. Espera a que `/health` responda 200 OK (polling cada 2s)
4. Aplica las migraciones de Alembic (`alembic upgrade head`)
5. Imprime las credenciales para configurar en Moodle

Después, `http://localhost:8001/docs` te abre el Swagger UI.

### 4. Verificar end-to-end (sin Moodle)

```bash
# Healthcheck simple
curl http://localhost:8001/health

# Test del endpoint mock /echo con HMAC
TIMESTAMP=$(date +%s)
NONCE=$(openssl rand -hex 16)
SECRET=$(grep '^NEXUSAI_SHARED_SECRET=' .env | cut -d= -f2)
APIKEY=$(grep '^NEXUSAI_API_KEY=' .env | cut -d= -f2)
BODY='{"question":"hola","course_id":1,"user_id":1}'
SIG=$(printf "%s%s%s" "$TIMESTAMP" "$NONCE" "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $NF}')

curl -s -X POST http://localhost:8001/api/v1/chat/echo \
  -H "Authorization: Bearer $APIKEY" \
  -H "X-Timestamp: $TIMESTAMP" \
  -H "X-Nonce: $NONCE" \
  -H "X-Signature: $SIG" \
  -H "Content-Type: application/json" \
  -d "$BODY"
```

Tiene que devolver `{"echo":"hola",...}`.

## Comandos útiles

```bash
# Levantar / parar
./scripts/dev.sh up         # postgres + redis + api
./scripts/dev.sh down       # parar (preserva datos)
./scripts/dev.sh destroy    # parar + borrar volúmenes (¡destructivo!)

# Después de cambiar .env (CRÍTICO — restart no relee .env)
./scripts/dev.sh reload     # equivale a: docker compose up -d --force-recreate

# Logs
./scripts/dev.sh logs api               # solo el backend
docker compose logs api -f --tail=50    # con follow

# Shell adentro del container
./scripts/dev.sh shell:api              # bash en el container
./scripts/dev.sh shell:pg               # psql en la DB

# Migraciones Alembic
docker compose exec api alembic current                # versión actual
docker compose exec api alembic upgrade head           # aplicar todas
docker compose exec api alembic downgrade -1           # bajar 1
docker compose exec api alembic revision --autogenerate -m "..."   # nueva migración

# Inspeccionar DB
docker compose exec postgres psql -U nexusai -d nexusai -c "\dt"
docker compose exec postgres psql -U nexusai -d nexusai -c "SELECT * FROM messages LIMIT 5;"
```

## Tests

```bash
# Correr toda la suite (dentro del container)
docker compose exec api pytest

# Verbose con coverage
docker compose exec api pytest -v --cov=app --cov-report=term-missing

# Solo tests de HMAC
docker compose exec api pytest tests/test_hmac.py -v

# Saltarse los tests marcados como "integration"
docker compose exec api pytest -m "not integration"
```

Convenciones:

- Los tests de HMAC mockean Redis (no requieren Redis real corriendo).
- Los tests de providers mockean el cliente `AsyncOpenAI` (no hacen llamadas reales a Gemini/OpenAI).
- Las env vars de test están seteadas en `tests/conftest.py` con valores dummy.

## Variables de entorno

Todas viven en el `.env` raíz del repo (no acá). Detalle en
[`.env.example`](../../.env.example).

| Variable | Default | Notas |
|---|---|---|
| `ENV` | `development` | Solo activa CORS para localhost en dev |
| `LLM_API_KEY` | — | Tu key de Gemini AI Studio |
| `LLM_BASE_URL` | endpoint OpenAI-compat de Gemini | Cambiar para usar OpenAI/Groq/Ollama |
| `LLM_MODEL` | `gemini-2.5-flash` | NO usar `gemini-2.0-flash` (cuota free 0 en cuentas nuevas) |
| `EMBEDDING_API_KEY` | — | Misma key que LLM en MVP |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | 768 dim |
| `EMBEDDING_DIMENSIONS` | `768` | Si cambia, hay que migrar `chunks.embedding` |
| `DATABASE_URL` | `postgresql://nexusai:...@postgres:5432/nexusai` | El compose lo override automáticamente |
| `REDIS_URL` | `redis://redis:6379/0` | El compose lo override automáticamente |
| `NEXUSAI_API_KEY` | — | Bearer key compartida con el plugin Moodle. Generar con `openssl rand -hex 32` |
| `NEXUSAI_SHARED_SECRET` | — | Secret HMAC. Generar con `openssl rand -hex 32` |
| `HMAC_REPLAY_WINDOW_SEC` | `300` | Ventana de tolerancia para clock skew |
| `RATE_LIMIT_PER_USER_DAILY` | `50` | Implementación pendiente Sprint 2 |
| `API_PORT` | `8001` | Puerto en el host (8000 dentro del container) |

## Endpoints

Swagger UI completo: http://localhost:8001/docs

| Endpoint | Método | Auth | Estado | Descripción |
|---|---|---|---|---|
| `/health` | GET | — | ✅ | Liveness probe |
| `/` | GET | — | ✅ | Banner con versión |
| `/api/v1/chat/echo` | POST | HMAC | ✅ | Mock que valida HMAC y devuelve eco — útil para diagnóstico |
| `/api/v1/chat/messages` | POST | HMAC | ✅ | Chat real con LLM. Persiste sesión, mensajes, llama Gemini |
| `/api/v1/chat/sessions` | GET | HMAC | ⏳ | Historial de conversaciones del usuario (Sprint 2) |
| `/api/v1/chat/sessions/{id}` | DELETE | HMAC | ⏳ | Borrar conversación (Sprint 2) |
| `/api/v1/documents` | POST | HMAC | ⏳ | Upload PDF + dispara pipeline RAG (Sprint 2) |
| `/api/v1/documents/{id}` | GET | HMAC | ⏳ | Estado de indexación (Sprint 2) |

## Troubleshooting frecuente

| Síntoma | Causa | Fix |
|---|---|---|
| `relation "chat_sessions" does not exist` | Migraciones no corrieron | `docker compose exec api alembic upgrade head` |
| Container arranca pero `/health` no responde | Postgres todavía starting | Esperar 10s y reintentar; verificar `docker compose ps` que postgres sea `Healthy` |
| `LLM model: gemini-2.0-flash` aunque cambiaste `.env` | `restart` NO relee `.env` | Usar `./scripts/dev.sh reload` (recrea containers) |
| `No config file 'alembic.ini' found` | Imagen vieja del backend sin alembic.ini | Rebuildear: `docker compose build api && docker compose up -d api` |
| `RateLimitError: limit: 0, model: gemini-2.0-flash` | Cuenta Gemini nueva no tiene cuota free para 2.0 | Cambiar `LLM_MODEL=gemini-2.5-flash` en `.env` y `dev.sh reload` |
| `Invalid signature` en cada request | Secrets de PHP y Python no coinciden | Re-pegar exactamente desde el `.env`. Ojo con sed truncando keys |
| `Request expired or clock skew` | Reloj de Mac y container desincronizados >5 min | Restart Docker Desktop o aumentar `HMAC_REPLAY_WINDOW_SEC` |
| `Replay detected: nonce already used` | El cliente PHP repitió un nonce | Verificar que `random_bytes(16)` esté generando único en cada request |
| OpenAI/Gemini timeout | Conexión lenta o LLM colgado | Default 120s. Si es persistente, bajar `timeout` en `app/providers/llm.py` |

## Linting y formato

```bash
docker compose exec api ruff check app/
docker compose exec api ruff format app/   # formatea in-place
```

## Generar nueva migración tras cambiar modelos

```bash
# Editar app/db/models.py (agregar tabla, columna, índice)
docker compose exec api alembic revision --autogenerate -m "descripcion corta"
# Revisar el archivo generado en migrations/versions/ — Alembic suele acertar 90%
docker compose exec api alembic upgrade head
```

## Próximos pasos (Sprint 2)

- ⏳ **`POST /api/v1/documents`** — upload PDF, dispara `pipeline.index_document()` async
- ⏳ **Retrieval semántico en `/messages`** — embed pregunta → query top-5 chunks por curso → inyectar como contexto en system prompt
- ⏳ **Streaming SSE** — usar `chat_stream()` del provider, emitir `data: {token}\n\n` desde el endpoint
- ⏳ **Endpoints de sesiones** — GET historial, DELETE conversación
- ⏳ **Tests** para `chunker`, `extractor`, `pipeline`, `models`
- ⏳ **Rate limiting** — middleware FastAPI + Redis counter por user/day

## Referencias

- Arquitectura del sistema: [`docs/architecture.md`](../../docs/architecture.md)
- Decisión multi-provider: [ADR-003](../../docs/adr/003-multi-provider-llm.md)
- Decisión Gemini MVP: [ADR-004](../../docs/adr/004-gemini-mvp-openai-prod.md)
- HMAC en detalle: [ADR-005](../../docs/adr/005-hmac-php-python.md)
- Privacy strategy: [ADR-006](../../docs/adr/006-privacy-strategy.md)
- Investigación FastAPI: [`investigacion/05-backend-fastapi/`](../../investigacion/05-backend-fastapi/)
- Investigación RAG: [`investigacion/02-rag/`](../../investigacion/02-rag/)
- Investigación procesamiento docs: [`investigacion/07-procesamiento-docs/`](../../investigacion/07-procesamiento-docs/)

## Licencia

GPL v3 — alineado con la licencia de Moodle.

---

*Última actualización: 5 May 2026 — Equipo NexusAI*
