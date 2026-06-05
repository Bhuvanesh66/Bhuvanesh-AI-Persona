"""End-to-end ingestion: load corpus -> chunk -> embed -> store in pgvector.

Usage:
    python -m app.ingest.run_ingest            # incremental (replaces resume + each repo)
    python -m app.ingest.run_ingest --fresh    # truncate everything first
"""
from __future__ import annotations

import sys

from app import db
from app.config import settings
from app.embeddings import embed
from app.ingest import chunker
from app.ingest.github_loader import Doc, load_all
from app.ingest.resume_loader import load_resume


def _chunk_doc(doc: Doc) -> list[str]:
    if doc.chunk_type == "code":
        return chunker.chunk_code(doc.text)
    return chunker.chunk_markdown(doc.text)


def run(fresh: bool = False) -> None:
    print("→ connecting to Postgres / pgvector …")
    conn = db.connect()
    db.init_schema(conn)
    if fresh:
        print("→ --fresh: truncating chunks table")
        db.reset(conn)

    print("→ loading resume …")
    docs: list[Doc] = load_resume()
    print("→ loading GitHub corpus …")
    docs += load_all()

    # Expand docs into (metadata, text) chunk rows.
    rows: list[tuple] = []
    texts: list[str] = []
    for doc in docs:
        for piece in _chunk_doc(doc):
            piece = piece.strip()
            if not piece:
                continue
            rows.append((doc.source, doc.repo, doc.file_path, doc.chunk_type,
                         doc.is_fork, doc.title, piece, chunker.est_tokens(piece)))
            texts.append(piece)

    if not rows:
        print("✗ nothing to ingest — check resume path and GitHub token.")
        return

    print(f"→ embedding {len(texts)} chunks via '{settings.embedding_provider}' …")
    vectors = embed(texts, input_type="document")

    # Clean re-ingest: clear the sources we're about to write.
    if not fresh:
        sources = {(r[0], r[1]) for r in rows}
        for source, repo in sources:
            db.delete_source(conn, source, repo)

    print("→ writing to pgvector …")
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks
               (source, repo, file_path, chunk_type, is_fork, title, content, tokens, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(*row, vec) for row, vec in zip(rows, vectors)],
        )

    total = conn.execute("SELECT count(*) FROM chunks;").fetchone()[0]
    forks = conn.execute("SELECT count(*) FROM chunks WHERE is_fork;").fetchone()[0]
    print(f"✓ done. {len(rows)} chunks written this run; {total} total in DB ({forks} from forks).")
    conn.close()


if __name__ == "__main__":
    run(fresh="--fresh" in sys.argv)
