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

    # GitHub (used for BOTH corpus ingestion AND GitHub Models LLM/embeddings)
    github_token: str = ""

    # GitHub Models — OpenAI-compatible endpoint
    github_models_base_url: str = "https://models.inference.ai.azure.com"
    chat_model: str = "gpt-4o"          # best instruction following on free tier
    voice_model: str = "gpt-4o-mini"    # faster TTFT for voice

    # Embeddings
    embedding_provider: str = "github"   # "github" (default) | "openai"
    openai_api_key: str = ""             # only needed if embedding_provider="openai"
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 1024

    # Vector store
    database_url: str = ""

    # Resume
    resume_path: str = "data/resume.txt"

    # Cal.com scheduling (optional — only needed for booking feature)
    calcom_api_key: str = ""
    calcom_event_type_id: int = 0

    # Retrieval knobs
    retrieve_k: int = 6
    candidate_k: int = 20


settings = Settings()
