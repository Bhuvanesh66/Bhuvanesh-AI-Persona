"""Hybrid retrieval: vector similarity + Postgres full-text, fused with RRF.

Reciprocal Rank Fusion (RRF) merges the two ranked lists without needing a
trained reranker — robust and dependency-free. A cross-encoder reranker
(e.g. voyage rerank-2) can be slotted in later behind the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass

from app import db
from app.config import settings
from app.embeddings import embed_query

RRF_K = 60


@dataclass
class Chunk:
    id: int
    source: str
    repo: str | None
    file_path: str | None
    chunk_type: str
    is_fork: bool
    title: str
    content: str
    score: float = 0.0


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row[0], source=row[1], repo=row[2], file_path=row[3],
        chunk_type=row[4], is_fork=row[5], title=row[6], content=row[7],
    )


_SELECT = "id, source, repo, file_path, chunk_type, is_fork, title, content"


def _vector_search(conn, qvec, k: int) -> list[Chunk]:
    rows = conn.execute(
        f"SELECT {_SELECT} FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s;",
        (qvec, k),
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def _text_search(conn, query: str, k: int) -> list[Chunk]:
    rows = conn.execute(
        f"""SELECT {_SELECT} FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s;""",
        (query, query, k),
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def retrieve(query: str, k: int | None = None) -> list[Chunk]:
    k = k or settings.retrieve_k
    cand = settings.candidate_k
    conn = db.connect()
    try:
        qvec = embed_query(query)
        vec_hits = _vector_search(conn, qvec, cand)
        txt_hits = _text_search(conn, query, cand)
    finally:
        conn.close()

    # Reciprocal Rank Fusion over the two lists.
    scores: dict[int, float] = {}
    by_id: dict[int, Chunk] = {}
    for ranked in (vec_hits, txt_hits):
        for rank, ch in enumerate(ranked):
            scores[ch.id] = scores.get(ch.id, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id[ch.id] = ch

    fused = sorted(by_id.values(), key=lambda c: scores[c.id], reverse=True)
    for c in fused:
        c.score = scores[c.id]
    return fused[:k]
