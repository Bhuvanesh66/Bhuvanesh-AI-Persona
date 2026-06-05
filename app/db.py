"""Postgres + pgvector connection and schema management."""
from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def connect() -> psycopg.Connection:
    """Open a connection with the pgvector type adapter registered."""
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    conn = psycopg.connect(settings.database_url, autocommit=True)
    # Ensure the extension exists before registering the vector type.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,                 -- 'resume' | 'github'
    repo        TEXT,                          -- repo name, or NULL for resume
    file_path   TEXT,                          -- 'README.md' | 'src/x.py' | 'commit:<sha>'
    chunk_type  TEXT NOT NULL,                 -- 'resume'|'readme'|'code'|'commit'|'meta'
    is_fork     BOOLEAN NOT NULL DEFAULT FALSE,
    title       TEXT,
    content     TEXT NOT NULL,
    tokens      INT,
    embedding   vector({dim}),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);",
    "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);",
    "CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo);",
]


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.format(dim=settings.embed_dim))
    for stmt in INDEXES:
        conn.execute(stmt)


def reset(conn: psycopg.Connection) -> None:
    """Drop all rows (used by `run_ingest --fresh`)."""
    conn.execute("TRUNCATE chunks RESTART IDENTITY;")


def delete_source(conn: psycopg.Connection, source: str, repo: str | None = None) -> None:
    if repo is None:
        conn.execute("DELETE FROM chunks WHERE source = %s;", (source,))
    else:
        conn.execute("DELETE FROM chunks WHERE source = %s AND repo = %s;", (source, repo))
