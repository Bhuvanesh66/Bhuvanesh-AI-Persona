"""Quick terminal test of the RAG core (no HTTP server needed).

    python -m app.ask "Why are you a good fit for an AI engineer role?"
    python -m app.ask                # interactive loop
"""
from __future__ import annotations

import sys

from app.rag.answer import answer


def _ask_once(q: str) -> None:
    res = answer(q)
    print("\n" + res["answer"] + "\n")
    print("— sources —")
    for s in res["sources"]:
        tag = "FORK" if s["is_fork"] else "orig"
        print(f"  [{s['n']}] {s['title']} ({s['type']}, {tag})")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        _ask_once(" ".join(sys.argv[1:]))
        return
    print("Ask the persona (Ctrl-C to quit):")
    try:
        while True:
            q = input("> ").strip()
            if q:
                _ask_once(q)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
