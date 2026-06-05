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

    # GitHub token — used ONLY for corpus ingest (repo scraping + embeddings)
    github_token: str = ""

    # GitHub Models — OpenAI-compatible endpoint (embeddings only)
    github_models_base_url: str = "https://models.inference.ai.azure.com"

    # Embeddings (GitHub Models — free, only used during ingest)
    embedding_provider: str = "github"
    openai_api_key: str = ""
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 1024

    # LLM models (GitHub Models free tier)
    chat_model: str = "gpt-4o"
    voice_model: str = "gpt-4o-mini"

    # Groq (optional alternative — faster, higher rate limit)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Vector store
    database_url: str = ""

    # Resume
    resume_path: str = "data/resume.txt"

    # Cal.com scheduling
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0
    calcom_username: str = ""        # your cal.com username (from cal.com/YOUR-USERNAME)
    calcom_event_slug: str = "30min" # event slug from the booking URL

    # Retrieval knobs
    retrieve_k: int = 6
    candidate_k: int = 20


settings = Settings()
