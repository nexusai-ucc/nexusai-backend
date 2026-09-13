from typing import Optional

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    env: str = "development"
    app_version: str = "0.1.0"

    # LLM
    llm_api_key: str
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash"

    # LLM — control del "thinking" de los modelos razonadores de Gemini (PERF-01).
    #
    # Los modelos de la familia flash 2.5+ razonan internamente ANTES de emitir
    # el primer token. Ese razonamiento no se ve, no se streamea, y es el grueso
    # de la latencia percibida: medido contra la API real, gemini-2.5-flash con
    # thinking automático tardaba ~45s hasta el primer token en una consulta RAG
    # típica, contra ~1-3s con el thinking apagado.
    #
    # Valores: "none" (sin razonamiento), "low", "medium", "high". Cualquier
    # valor de _EFFORT_UNSET en providers/llm.py ("", "default", "auto") deja
    # que el modelo decida, o sea el comportamiento previo a PERF-01.
    #
    # `llm_reasoning_effort` es el default de TODAS las llamadas.
    # `llm_reasoning_effort_generation` lo pisa solo donde la calidad del
    # output justifica pagar los segundos extra — hoy, la generación de
    # preguntas de quiz y de examen (ver quiz/router.py::_run_quiz_generation).
    llm_reasoning_effort: str = "none"
    llm_reasoning_effort_generation: str = "low"

    # LLM — proveedor secundario (fallback automático, INFRA-01 / issue #307).
    # Opcionales: si los 3 no están seteados, el fallback queda deshabilitado
    # y el comportamiento es idéntico al de antes (un solo proveedor).
    llm_fallback_api_key: Optional[str] = None
    llm_fallback_base_url: Optional[str] = None
    llm_fallback_model: Optional[str] = None

    # LLM — cadena de modelos intermedios de Gemini (INFRA-03 / issue #343).
    # Opcional. Lista separada por comas de modelos que comparten
    # LLM_API_KEY/LLM_BASE_URL con el primario (ej. cuotas gratuitas
    # independientes por modelo en Google AI Studio) — se prueban en orden
    # ANTES de pasar al proveedor secundario. Vacío = comportamiento idéntico
    # a antes de INFRA-03 (salta directo del primario al secundario).
    llm_intermediate_models: Optional[str] = None

    # Resúmenes de documentos (PERF-02) — ver documents/summarizer.py.
    #
    # summary_cache_ttl_sec: cuánto vive en Redis el resumen ya generado de un
    #   documento. La cache key incluye el hash/fecha del archivo, así que
    #   reemplazar un documento (CONT-07) invalida su entrada sola. 0 = sin cache.
    # summary_max_concurrency: cuántos resúmenes de documento se piden al LLM en
    #   paralelo dentro del resumen pre-parcial. Subirlo acelera cursos con mucho
    #   material, pero contra la cuota gratuita de Gemini aumenta la chance de
    #   429/503 — 4 es el compromiso elegido.
    summary_cache_ttl_sec: int = 86400
    summary_max_concurrency: int = 4

    # Voz — transcripción de audio a texto (VOICE-01, issue #314).
    #
    # Ningún proveedor ya configurado (Gemini activo, OpenAI fallback) expone
    # transcripción utilizable desde acá: Gemini no la tiene en el shim
    # OpenAI-compat que usamos para chat, y el fallback OpenAI está
    # deshabilitado en este entorno. Groq sí expone un endpoint de
    # transcripción compatible con el SDK de OpenAI (mismo SDK, otro
    # base_url) — ver app/providers/transcription.py. Opcional: sin
    # `groq_api_key`, el endpoint de voz devuelve un 503 explícito en vez de
    # fallar al armar el cliente.
    groq_api_key: Optional[str] = None
    groq_stt_model: str = "whisper-large-v3"

    # Embeddings
    embedding_api_key: str
    embedding_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Security
    nexusai_shared_secret: str
    nexusai_api_key: str
    hmac_replay_window_sec: int = 300

    # Rate limiting
    rate_limit_per_user_daily: int = 50
    rate_limit_per_user_minute: int = 20

    # API
    api_port: int = 8001

    # Moderación de contenido — capa de seguridad sobre las entradas del
    # alumno antes de llegar al LLM principal. Ver app/shared/moderation.py.
    #
    # moderation_enabled: apaga toda la capa (default True). Pensado para dev
    #   local, donde no siempre se quiere pagar la latencia/costo extra de la
    #   llamada de moderación.
    # moderation_api_key: si está seteada, se usa la Moderation API de OpenAI
    #   (gratuita, rápida, especializada en esto) sin importar cuál sea el
    #   LLM_BASE_URL activo — es un endpoint HTTP aparte, no pasa por
    #   LLMProvider. Sin esta key (p. ej. en el MVP con Gemini, que no expone
    #   un endpoint de moderación equivalente vía el shim OpenAI-compat), se
    #   cae a clasificar el texto con el LLM activo — más caro y algo menos
    #   preciso, pero mantiene el agnosticismo de proveedor (ADR-003).
    # moderation_fail_open: qué hacer si la moderación misma falla (timeout,
    #   ambos caminos caídos). True (default) = dejar pasar el mensaje al LLM
    #   principal — bloquear alumnos por la falla de un servicio AUXILIAR es
    #   peor UX que el riesgo residual, y el LLM principal ya tiene su propio
    #   system prompt con guardrails. False = bloquear hasta que la
    #   moderación vuelva a estar disponible.
    moderation_enabled: bool = True
    moderation_api_key: Optional[str] = None
    moderation_fail_open: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    # mypy no sabe que BaseSettings completa los args "faltantes" leyendo
    # variables de entorno en tiempo de ejecución (no hay plugin de mypy
    # para pydantic-settings instalado acá).
    return Settings()  # type: ignore[call-arg]
