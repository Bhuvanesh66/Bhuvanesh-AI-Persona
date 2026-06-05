"""Pluggable embedding provider. Both providers emit 1024-dim vectors so the
pgvector schema stays fixed regardless of choice."""
from __future__ import annotations

from functools import lru_cache

from app.config import settings

_BATCH = 64


@lru_cache(maxsize=1)
def _voyage():
    import voyageai

    return voyageai.Client(api_key=settings.voyage_api_key)


@lru_cache(maxsize=1)
def _openai():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str], *, input_type: str = "document") -> list[list[float]]:
    """Embed a list of texts. input_type is 'document' (corpus) or 'query'."""
    if not texts:
        return []

    provider = settings.embedding_provider.lower()
    out: list[list[float]] = []

    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        if provider == "voyage":
            resp = _voyage().embed(batch, model="voyage-3", input_type=input_type)
            out.extend(resp.embeddings)
        elif provider == "openai":
            resp = _openai().embeddings.create(
                model="text-embedding-3-large",
                input=batch,
                dimensions=settings.embed_dim,  # force 1024 to match schema
            )
            out.extend([d.embedding for d in resp.data])
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")

    return out


def embed_query(text: str) -> list[float]:
    return embed([text], input_type="query")[0]
