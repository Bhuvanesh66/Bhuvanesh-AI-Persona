"""Pluggable embedding provider. All providers emit 1024-dim vectors
so the pgvector schema stays fixed regardless of choice.

  github (default) — text-embedding-3-large via GitHub Models, uses GITHUB_TOKEN
  openai           — text-embedding-3-large via OpenAI, uses OPENAI_API_KEY
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings

_BATCH = 64


def _llm_client():
    from openai import OpenAI
    return OpenAI(
        base_url=settings.github_models_base_url,
        api_key=settings.github_token,
    )


@lru_cache(maxsize=1)
def _github_client():
    return _llm_client()


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str], *, input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []

    provider = settings.embedding_provider.lower()
    client = _github_client() if provider == "github" else _openai_client()
    out: list[list[float]] = []

    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        resp = client.embeddings.create(
            model=settings.embed_model,
            input=batch,
            dimensions=settings.embed_dim,
        )
        out.extend([d.embedding for d in resp.data])

    return out


def embed_query(text: str) -> list[float]:
    return embed([text], input_type="query")[0]
