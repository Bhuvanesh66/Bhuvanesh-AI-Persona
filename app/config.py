"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Persona
    persona_name: str = "Bhuvanesh"
    github_owner: str = "Bhuvanesh66"

    # Anthropic
    anthropic_api_key: str = ""
    chat_model: str = "claude-opus-4-8"
    voice_model: str = "claude-haiku-4-5"

    # GitHub
    github_token: str = ""

    # Embeddings
    embedding_provider: str = "voyage"  # "voyage" | "openai"
    voyage_api_key: str = ""
    openai_api_key: str = ""
    embed_dim: int = 1024  # both providers configured to emit 1024-dim vectors

    # Vector store
    database_url: str = ""

    # Resume
    resume_path: str = "data/resume.pdf"

    # Cal.com scheduling
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0   # 30-min interview event type

    # Retrieval knobs
    retrieve_k: int = 6          # final chunks passed to the LLM
    candidate_k: int = 20        # candidates pulled per retriever before fusion


settings = Settings()
