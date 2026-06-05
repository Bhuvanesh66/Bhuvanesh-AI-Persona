"""Lightweight, structure-aware chunking.

Token sizing uses a char-based heuristic (~4 chars/token) — this is only for
splitting, not billing, so an approximation is fine and avoids a tokenizer dep.
"""
from __future__ import annotations

import re

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 400
OVERLAP_TOKENS = 60

_TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
_OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN


def est_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_oversized(block: str) -> list[str]:
    """Sliding-window split for a block longer than the target."""
    out: list[str] = []
    start = 0
    n = len(block)
    while start < n:
        end = min(start + _TARGET_CHARS, n)
        out.append(block[start:end])
        if end >= n:
            break
        start = end - _OVERLAP_CHARS
    return out


def chunk_markdown(text: str) -> list[str]:
    """Split markdown on headings, then pack sections up to the target size."""
    if not text or not text.strip():
        return []

    # Split on markdown headings while keeping the heading with its body.
    parts = re.split(r"(?m)^(?=#{1,6}\s)", text)
    sections = [p.strip() for p in parts if p.strip()]
    if not sections:
        sections = [text.strip()]

    chunks: list[str] = []
    buf = ""
    for sec in sections:
        if len(sec) > _TARGET_CHARS:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_oversized(sec))
            continue
        if len(buf) + len(sec) + 2 <= _TARGET_CHARS:
            buf = f"{buf}\n\n{sec}" if buf else sec
        else:
            if buf:
                chunks.append(buf)
            buf = sec
    if buf:
        chunks.append(buf)
    return chunks


def chunk_code(text: str) -> list[str]:
    """Code: plain sliding window (keeps things simple and language-agnostic)."""
    if not text or not text.strip():
        return []
    if len(text) <= _TARGET_CHARS:
        return [text]
    return _split_oversized(text)
