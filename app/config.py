"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Persona
    persona_name: str = "Bhuvanesh"
    github_owner: str = "Bhuvanesh66"

    # GitHub token — used for corpus ingest + embeddings
    github_token: str = ""

    # GitHub Models — OpenAI-compatible endpoint (embeddings only)
    github_models_base_url: str = "https://models.inference.ai.azure.com"

    # Embeddings (GitHub Models — free, only used during ingest)
    embedding_provider: str = "github"
    openai_api_key: str = ""
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 1024

    # LLM models — GitHub Models (gpt-4o-mini)
    chat_model: str = "gpt-4o-mini"
    voice_model: str = "gpt-4o-mini"

    # Vector store
    database_url: str = ""

    # Resume
    resume_path: str = "data/resume.txt"

    # Cal.com scheduling
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0
    calcom_username: str = ""
    calcom_event_slug: str = "30min"

    # Retrieval knobs
    retrieve_k: int = 6
    candidate_k: int = 20


settings = Settings()
