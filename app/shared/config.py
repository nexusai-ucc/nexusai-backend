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

    # Alertas mínimas (ver ADR-012 y app/shared/alerting.py).
    #
    # alert_smtp_*/alert_email_to: envío por email vía SMTP (pensado para
    #   Gmail con una App Password — no la contraseña normal de la cuenta,
    #   Gmail bloquea login SMTP directo). Sin `alert_smtp_user` +
    #   `alert_smtp_password` + `alert_email_to` configuradas, las alertas
    #   quedan solo logueadas (WARNING) — no rompe nada, solo pierde
    #   visibilidad automática. Nunca hardcodear la password acá: siempre
    #   por variable de entorno.
    #
    # error_rate_*: ventana fija (mismo patrón que rate_limit.py) para avisar
    #   si hay una ráfaga de respuestas 5xx.
    #
    # llm_failure_*: ídem pero contando fallas del LLM específicamente (el
    #   proveedor agotó su cadena de fallback completa), que llegan al
    #   cliente como 503 pero conviene diferenciar de un 5xx genérico porque
    #   la causa y la acción a tomar son otras (cuota/proveedor caído).
    #
    # llm_slow_*: latencia del LLM. Una respuesta lenta no es un error (sigue
    #   devolviendo 200), así que no la detecta el conteo de 5xx — necesita
    #   su propio umbral.
    alert_smtp_host: str = "smtp.gmail.com"
    alert_smtp_port: int = 465
    alert_smtp_user: Optional[str] = None
    alert_smtp_password: Optional[str] = None
    alert_email_to: Optional[str] = None

    error_rate_window_sec: int = 60
    error_rate_threshold: int = 10

    llm_failure_window_sec: int = 300
    llm_failure_threshold: int = 3

    # Cuota/presupuesto del proveedor de IA: una RateLimitError (429) que se
    # propaga hasta el caller significa que la cadena ENTERA (primario +
    # intermedios + secundario, ver providers/llm.py) devolvió "cuota
    # agotada". Umbral bajo (1) porque, a diferencia de una falla transitoria
    # cualquiera, esto ya implica que ningún eslabón configurado puede
    # responder — vale la pena avisar de inmediato.
    llm_quota_window_sec: int = 600
    llm_quota_threshold: int = 1

    llm_slow_threshold_ms: int = 15_000
    llm_slow_window_sec: int = 300
    llm_slow_threshold_count: int = 5

    # API
    api_port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
