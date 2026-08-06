from typing import Optional

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    env: str = "development"
    app_version: str = "0.1.0"

    # LLM
    llm_api_key: str
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.0-flash"

    # LLM — proveedor secundario (fallback automático, INFRA-01 / issue #307).
    # Opcionales: si los 3 no están seteados, el fallback queda deshabilitado
    # y el comportamiento es idéntico al de antes (un solo proveedor).
    llm_fallback_api_key: Optional[str] = None
    llm_fallback_base_url: Optional[str] = None
    llm_fallback_model: Optional[str] = None

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
