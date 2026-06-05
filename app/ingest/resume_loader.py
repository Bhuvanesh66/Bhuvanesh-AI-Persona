"""Load the resume from PDF or plain text into a single document."""
from __future__ import annotations

import os

from app.config import settings
from app.ingest.github_loader import Doc


def load_resume() -> list[Doc]:
    path = settings.resume_path
    if not os.path.exists(path):
        # allow a .txt fallback next to the configured path
        alt = os.path.splitext(path)[0] + ".txt"
        if os.path.exists(alt):
            path = alt
        else:
            print(f"  ! resume not found at {settings.resume_path} (or .txt fallback) — skipping")
            return []

    if path.lower().endswith(".pdf"):
        text = _read_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    if not text or not text.strip():
        print(f"  ! resume at {path} produced no text — skipping")
        return []

    print(f"  resume loaded from {path} ({len(text)} chars)")
    return [Doc("resume", None, "resume", "resume", False, "Resume", text)]


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(pages)
