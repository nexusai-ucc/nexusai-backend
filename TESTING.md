# NexusAI — Plan de Testing y Criterios de Aceptación

**Sprint 4 · Entrega MVP: 1 de junio de 2026**

---

## 1. Criterios de aceptación por feature

### Chat RAG
| Criterio | Valor objetivo |
|----------|---------------|
| Tiempo de primera respuesta (corpus ≤ 20 docs) | < 8 s |
| Respuesta incluye fuente citada | Siempre (cuando hay contexto recuperado) |
| Idioma de respuesta | Español |
| Respuesta honesta cuando no hay contexto | No inventa — indica que no encontró información |

### Indexación de documentos
| Criterio | Valor objetivo |
|----------|---------------|
| PDF de 10 páginas: tiempo total hasta `status=indexed` | < 30 s |
| PDF de 50 páginas | < 120 s |
| Estado final tras indexación exitosa | `status = "indexed"` |
| Estado final tras fallo de indexación | `status = "error"` con `error_message` no nulo |
| Tipos de archivo aceptados | PDF, DOCX, TXT |

### Dedup incremental (CONT-04)
| Criterio | Valor objetivo |
|----------|---------------|
| Segunda subida del mismo archivo (mismo hash) | HTTP 200 en < 1 s, sin nueva indexación |
| Subida de archivo modificado (hash distinto) | HTTP 202, nueva indexación completa |

### Vista de estado (CONT-05)
| Criterio | Valor objetivo |
|----------|---------------|
| Columna "Indexado el" visible tras indexación exitosa | Muestra `dd/mm/yyyy HH:MM` |
| Columna durante estado `pending` o `indexing` | Muestra "—" |
| Columna en estado `error` | Muestra la fecha del último intento |

---

## 2. Alcance del testing por tipo

### Unitario (automatizado — `pytest`)
Módulo cubierto → archivo de test:
- `app/documents/chunker.py` → `tests/test_chunker.py`
- `app/documents/extractor.py` → `tests/test_extractor.py`
- `app/documents/retriever.py` → `tests/test_retriever.py`
- `app/providers/llm.py` + `embeddings.py` → `tests/test_providers.py`
- `app/auth/hmac.py` → `tests/test_hmac.py`
- `app/documents/router.py` → `tests/test_documents_router.py`
- `app/documents/pipeline.py` → `tests/test_pipeline.py`

Correr: `cd services/api && .venv/bin/python -m pytest tests/ -v`

### Integración Moodle (automatizado — PHPUnit vía moodle-plugin-ci)
- `document_list` External Function → `plugin/local/nexusai/tests/document_list_test.php`
- HMAC de `backend_client` → `plugin/local/nexusai/tests/backend_client_test.php`

Correr (en entorno con Moodle instalado):
`vendor/bin/phpunit plugin/local/nexusai/tests/`

### Integración backend-frontend (manual)
Ver sección 4 — Guión de integración.

### Funcional E2E (manual)
Ver sección 5 — Guión E2E.

### Calidad de respuestas (manual)
Ver `tests/golden_set.md`.

### Performance (manual)
Ver sección 6 — Benchmarks.

### Pruebas con usuarios reales (TEST-06)
Coordinadas con compañeros UCC. Ver sección 7.

---

## 3. Definición de "sprint cerrado"

El Sprint 4 se considera cerrado cuando:

1. **CI verde**: todos los tests automatizados pasan en GitHub Actions
   (backend-ci.yml + moodle-ci.yml sin fallos).
2. **Guiones manuales ejecutados**: secciones 4 y 5 con columna "resultado obtenido"
   completada y todos los pasos en OK.
3. **Golden set evaluado**: `tests/golden_set.md` con puntajes completados;
   promedio de "honestidad" ≥ 4/5 en preguntas fuera del material.
4. **Benchmarks medidos**: tabla de sección 6 completada; todos los SLAs en OK.
5. **Sin bugs críticos abiertos**: bugs encontrados en TEST-06 corregidos (TEST-07).

---

## 4. Guión de integración backend-frontend (TEST-03)

**Prerequisito:** backend FastAPI corriendo (`docker compose up`), plugin Moodle
instalado y configurado con la URL y credenciales del backend.

| # | Acción | Resultado esperado | Resultado obtenido | OK/FAIL |
|---|--------|-------------------|-------------------|---------|
| 1 | Iniciar sesión en Moodle como docente. Ir al curso de prueba. Abrir la pestaña NexusAI → Documentos. | Se carga la tabla de documentos (vacía o con docs previos). | | |
| 2 | Subir un PDF de prueba (≥ 5 páginas) usando el uploader. | El documento aparece en la tabla con `status = "En cola"`. No se recarga la página. | | |
| 3 | Esperar hasta que `status` cambie a `"✓ Indexado"`. Verificar la columna "Indexado el". | La columna muestra la fecha en formato `dd/mm/yyyy HH:MM`. Antes del cambio mostraba "—". | | |
| 4 | Subir el **mismo archivo** otra vez. | La tabla muestra inmediatamente el documento ya existente con `status = "✓ Indexado"` (HTTP 200). El documento no aparece duplicado. | | |
| 5 | Subir un **archivo modificado** (mismo nombre, contenido distinto). | Aparece un nuevo documento en `status = "En cola"` y se indexa normalmente (HTTP 202). | | |
| 6 | Ir a la pestaña de chat. Escribir una pregunta sobre el contenido del PDF. | La respuesta llega en < 8 s, en español, y menciona el nombre del archivo como fuente. | | |
| 7 | Verificar en los logs del backend (`docker compose logs api`) que la request de upload llegó firmada con HMAC. | Log muestra línea de acceso con ruta `/api/v1/documents` y HTTP 202 (o 200 en dedup). | | |
| 8 | Borrar el documento desde la tabla. Volver a preguntar sobre el contenido. | La respuesta indica que no hay información disponible (el retriever no recupera chunks borrados). | | |

---

## 5. Guión funcional E2E (TEST-05)

### Flujo 1 — Docente: subida e indexación

**Prerequisito:** usuario con rol `teacher` en el curso de prueba.

| # | Acción | Resultado esperado | Resultado obtenido | OK/FAIL |
|---|--------|-------------------|-------------------|---------|
| 1 | Ir a `http://<moodle>/login` → ingresar credenciales de docente. | Login exitoso, dashboard visible. | | |
| 2 | Hacer clic en el curso de prueba → pestaña NexusAI → subpestaña Documentos. | Vista de gestión de documentos carga correctamente. | | |
| 3 | Arrastrar un PDF al área de upload (o usar "Seleccionar archivo"). | El archivo se muestra en la tabla con `status = "En cola"`. | | |
| 4 | Observar el cambio de estado en tiempo real sin recargar. | Status cambia: `"En cola"` → `"Indexando"` → `"✓ Indexado"`. El polling cada 3 s es imperceptible para el usuario. | | |
| 5 | Verificar la columna "Indexado el". | Muestra fecha y hora actuales en `dd/mm/yyyy HH:MM`. | | |
| 6 | Subir el **mismo archivo** sin modificar. | La tabla muestra el documento original; no aparece duplicado. El feedback es instantáneo (< 1 s). | | |
| 7 | Subir un PDF con un **tipo de archivo no soportado** (ej. `.png` renombrado a `.pdf`). | Aparece un mensaje de error en la tabla (`status = "✕ Error"`). El tooltip muestra el motivo. | | |
| 8 | Hacer clic en "Eliminar" sobre un documento indexado → confirmar. | El documento desaparece de la tabla inmediatamente. | | |

### Flujo 2 — Alumno: consulta RAG

**Prerequisito:** usuario con rol `student` en el mismo curso. El docente ya indexó al menos un documento en el Flujo 1.

| # | Acción | Resultado esperado | Resultado obtenido | OK/FAIL |
|---|--------|-------------------|-------------------|---------|
| 1 | Ir a `http://<moodle>/login` → ingresar credenciales de alumno. | Login exitoso. | | |
| 2 | Entrar al curso → abrir el widget NexusAI (pestaña Chat). | El widget carga; se ve el campo de texto y el historial vacío. | | |
| 3 | Escribir una pregunta cuya respuesta está en el PDF subido por el docente. | Respuesta en español en < 8 s. Menciona el nombre del archivo como fuente. | | |
| 4 | Escribir una pregunta de seguimiento relacionada con la anterior. | El asistente mantiene el contexto de la conversación. | | |
| 5 | Escribir una pregunta cuya respuesta **no está en el material** (ej. "¿Cuál es la capital de Francia?"). | El asistente responde honestamente que no encontró esa información en el material del curso, sin inventar. | | |
| 6 | Recargar la página y volver al chat. | El historial de la sesión se conserva. | | |
| 7 | Verificar que el alumno **no ve** la pestaña de Documentos (gestión). | Solo se muestra la pestaña de Chat. | | |

---

## 6. Benchmarks de rendimiento (TEST-09)

### SLAs definidos

| Escenario | SLA | Medición real | OK/FAIL |
|-----------|-----|--------------|---------|
| Indexación PDF ≤ 10 páginas | < 30 s | | |
| Indexación PDF ≤ 50 páginas | < 120 s | | |
| Query RAG (corpus ≤ 20 docs) — tiempo hasta primera respuesta | < 8 s | | |
| Dedup CONT-04 — segunda subida del mismo archivo | < 1 s | | |
| Carga inicial del widget NexusAI en Moodle | < 3 s | | |
| Respuesta en estado de cold start (primera query del día) | < 15 s | | |

### Cómo medir

**Indexación (tiempo hasta `status=indexed`):**
```bash
# 1. Subir el documento y anotar el document_id del response.
# 2. Polear hasta que status='indexed':
START=$(date +%s); \
while true; do \
  STATUS=$(curl -s -H "Authorization: Bearer <KEY>" \
    http://localhost:8001/api/v1/documents/<DOC_ID> | jq -r '.status'); \
  echo "$(( $(date +%s) - START ))s → $STATUS"; \
  [ "$STATUS" = "indexed" ] && break; \
  sleep 2; \
done
```

**Query RAG (tiempo de respuesta):**
```bash
curl -s -o /dev/null -w "tiempo_total: %{time_total}s\n" \
  -X POST http://localhost:8001/api/v1/chat/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <KEY>" \
  -d '{"question":"¿Qué temas cubre el curso?","course_id":1,"user_id":1}'
```

**Dedup (segunda subida):**
```bash
# Correr dos veces el mismo upload y comparar los tiempos:
time curl -s -X POST http://localhost:8001/api/v1/documents \
  -H "Authorization: Bearer <KEY>" \
  -d @payload.json
```

**Carga del widget:**
Abrir las DevTools del browser (F12) → pestaña Network → recargar la página de Moodle.
Anotar el tiempo en la entrada `nexusai/amd/build/widget.min.js` (columna "Time").

---

## 7. Pruebas con usuarios reales (TEST-06)

**Participantes objetivo:** 3-5 compañeros de la UCC con acceso a la instancia de prueba.

**Escenarios a evaluar:**
1. Subida de un documento propio (apuntes de clase, PDF de parcial).
2. Consulta de al menos 3 preguntas sobre el documento subido.
3. Intento de consulta fuera del material.

**Criterios de satisfacción (escala 1-5 en formulario Google Forms):**
- ¿Cuán útiles fueron las respuestas del asistente?
- ¿Las respuestas citaron el documento correctamente?
- ¿El tiempo de respuesta fue aceptable?
- ¿La interfaz fue fácil de usar?

**Bugs reportados:** registrar en el issue TEST-07 con pasos para reproducir.
