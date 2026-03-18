"""Configuration management using pydantic-settings."""
from __future__ import annotations

import logging

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    z_ai_api_key: SecretStr
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"

    # OpenAI
    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o"

    # Embedding (provider-agnostic)
    embedding_provider: str = "openai"  # "openai" | "gemini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    gemini_api_key: SecretStr | None = None

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    # Reranker
    reranker_backend: str = "openai"  # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None

    # Feature flags
    image_caption_enabled: bool = True

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_json: bool = False


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
