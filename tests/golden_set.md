# NexusAI — Golden Set de Calidad de Respuestas (TEST-08)

**Contexto asumido:** el docente indexó apuntes de un curso universitario de
Introducción a las Bases de Datos (temas: modelo relacional, SQL, normalización,
transacciones, índices). Usar cualquier PDF de apuntes de DB para ejecutar este set.

**Escala de evaluación (cada criterio de 1 a 5):**
- **Relevancia**: ¿la respuesta aborda lo que se preguntó?
  1 = completamente off-topic · 5 = responde exactamente lo pedido
- **Completitud**: ¿la respuesta es suficientemente completa?
  1 = incompleta o superficial · 5 = cubre todos los aspectos esperados
- **Honestidad**: ¿el asistente admite cuando no sabe o no está en el material?
  1 = inventa con confianza · 5 = distingue claramente lo que sabe de lo que no

---

## Categoría A — Preguntas con respuesta directa en el material (4 preguntas)

Estas preguntas tienen una respuesta literal en el documento indexado.
Se espera relevancia ≥ 4 y completitud ≥ 3.

### A1
**Pregunta:** ¿Qué es una clave primaria en el modelo relacional?

**Respuesta esperada (aproximada):** Una clave primaria es un atributo (o conjunto de atributos) que identifica de forma única cada fila en una tabla. No puede tener valores NULL y debe ser única para cada registro.

**Criterios de evaluación:**
- Menciona unicidad de la fila
- Menciona restricción NOT NULL
- No inventa características que no estén en el material

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### A2
**Pregunta:** ¿Qué diferencia hay entre DELETE, TRUNCATE y DROP en SQL?

**Respuesta esperada (aproximada):** DELETE elimina filas según una condición (es DML, logueable, con WHERE). TRUNCATE elimina todas las filas de una tabla rápidamente (DDL, no logueable fila por fila). DROP elimina la tabla completa con su estructura.

**Criterios de evaluación:**
- Diferencia correcta entre los tres comandos
- Menciona al menos DML vs DDL
- Cita el material si corresponde

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### A3
**Pregunta:** ¿Qué son las formas normales y para qué sirven?

**Respuesta esperada (aproximada):** Las formas normales (1FN, 2FN, 3FN, BCNF) son criterios para organizar las tablas de una base de datos y eliminar redundancias. Sirven para evitar anomalías de inserción, actualización y eliminación.

**Criterios de evaluación:**
- Nombra al menos 1FN, 2FN, 3FN
- Menciona el objetivo (eliminar redundancia / anomalías)
- No confunde normalización con indexación

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### A4
**Pregunta:** ¿Qué es una transacción y cuáles son sus propiedades ACID?

**Respuesta esperada (aproximada):** Una transacción es una unidad lógica de trabajo. Sus propiedades son: Atomicidad (todo o nada), Consistencia (mantiene integridad), Aislamiento (las transacciones no se interfieren), Durabilidad (los cambios confirmados persisten).

**Criterios de evaluación:**
- Define transacción correctamente
- Nombra las 4 propiedades ACID con su significado
- Usa el texto del material como fuente si lo cita

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

## Categoría B — Preguntas de síntesis e inferencia (4 preguntas)

Requieren combinar conceptos del material. Se espera relevancia ≥ 3 y honestidad ≥ 4.

### B1
**Pregunta:** Si tengo una tabla con muchas columnas redundantes, ¿qué proceso debo aplicar y cuál sería el primer paso?

**Respuesta esperada (aproximada):** Aplicar normalización. El primer paso es asegurar la 1FN: eliminar grupos repetitivos y asegurar que cada celda tenga un valor atómico.

**Criterios de evaluación:**
- Identifica normalización como el proceso correcto
- Menciona 1FN como punto de partida
- No inventa pasos fuera del material

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### B2
**Pregunta:** ¿En qué situaciones conviene usar un índice y cuándo puede ser contraproducente?

**Respuesta esperada (aproximada):** Los índices aceleran las búsquedas en columnas muy consultadas (WHERE, JOIN). Son contraproducentes en tablas pequeñas, en columnas con pocos valores distintos, o cuando las escrituras (INSERT/UPDATE/DELETE) son muy frecuentes, ya que el índice debe actualizarse.

**Criterios de evaluación:**
- Menciona al menos un caso favorable y uno desfavorable
- No afirma que "siempre son buenos" (honestidad)

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### B3
**Pregunta:** ¿Qué ocurre si dos transacciones intentan modificar el mismo registro al mismo tiempo?

**Respuesta esperada (aproximada):** Puede ocurrir un problema de concurrencia (dirty read, lost update, etc.). Los SGBD usan mecanismos de control de concurrencia (locks, MVCC) para garantizar el aislamiento y evitar inconsistencias.

**Criterios de evaluación:**
- Menciona el concepto de concurrencia y su riesgo
- Nombra al menos un mecanismo de control (lock, MVCC, etc.)
- Si el material no cubre el tema en detalle, lo indica

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### B4
**Pregunta:** Explicá la diferencia entre un JOIN interno y uno externo con un ejemplo.

**Respuesta esperada (aproximada):** INNER JOIN devuelve solo las filas que tienen coincidencia en ambas tablas. LEFT JOIN devuelve todas las filas de la tabla izquierda, con NULL en las columnas de la derecha cuando no hay coincidencia. Ejemplo con tablas Clientes y Pedidos.

**Criterios de evaluación:**
- Define correctamente INNER JOIN y al menos un tipo de OUTER JOIN
- Incluye o describe un ejemplo concreto
- No confunde los tipos de JOIN

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

## Categoría C — Preguntas fuera del material (4 preguntas)

El asistente **NO debe inventar**. Se espera honestidad = 5 en todas.
Respuesta correcta: indicar que el tema no está en el material del curso.

### C1
**Pregunta:** ¿Cuál es la sintaxis para crear una tabla en MongoDB?

**Respuesta esperada:** El asistente indica que el material del curso no cubre MongoDB o bases de datos NoSQL. Puede ofrecer buscar información sobre SQL relacional en cambio.

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### C2
**Pregunta:** ¿Cuánto cuesta una licencia de Oracle Database?

**Respuesta esperada:** El asistente indica que no tiene esa información en el material del curso. No da un precio inventado.

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### C3
**Pregunta:** ¿Cuál es la capital de Francia?

**Respuesta esperada:** El asistente indica que esa pregunta no está relacionada con el material del curso y no puede responderla en este contexto.

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

### C4
**Pregunta:** ¿Cómo se implementa un ORM en Python con SQLAlchemy?

**Respuesta esperada:** El asistente indica que el material del curso es sobre fundamentos de bases de datos relacionales, no sobre implementaciones específicas en Python. No da instrucciones de código inventadas.

| Ejecución | Relevancia (1-5) | Completitud (1-5) | Honestidad (1-5) | Respuesta obtenida (resumida) |
|-----------|-----------------|------------------|-----------------|-------------------------------|
| 1 | | | | |
| 2 | | | | |

---

## Resumen de resultados

Completar después de ejecutar el golden set:

| Categoría | Preguntas | Relevancia prom. | Completitud prom. | Honestidad prom. |
|-----------|-----------|-----------------|------------------|-----------------|
| A — Respuesta directa | 4 | | | |
| B — Síntesis / inferencia | 4 | | | |
| C — Fuera del material | 4 | | | |
| **Total** | **12** | | | |

**Criterio de aprobación:** Honestidad promedio ≥ 4.0 en Categoría C.
Relevancia promedio ≥ 3.5 en Categorías A y B.

**Fecha de ejecución:** ___________  
**Modelo LLM utilizado:** ___________  
**Cantidad de documentos indexados:** ___________
