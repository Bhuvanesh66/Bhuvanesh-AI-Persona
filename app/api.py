"""FastAPI surface for the chat track.

    uvicorn app.api:app --reload

Endpoints:
  GET  /health        — liveness probe (used by the uptime monitor)
  POST /chat          — JSON: {message, history?} -> {answer, sources}
  POST /chat/stream   — SSE stream of answer deltas, then a final 'sources' event
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.rag.answer import answer, answer_stream
from app.voice.retell_handler import router as voice_router

app = FastAPI(title=f"{settings.persona_name} — AI Persona")

import os as _os
_ALLOWED_ORIGINS = [o.strip() for o in _os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {
        "status": "ok",
        "persona": settings.persona_name,
        "model": settings.effective_chat_model,
    }


@app.get("/debug")
def debug() -> dict:
    """Diagnose configuration issues. Safe to expose — no secrets returned."""
    import traceback

    results: dict = {
        "groq_key_set": bool(settings.groq_api_key),
        "use_groq": settings.use_groq,
        "chat_model": settings.chat_model,
        "effective_chat_model": settings.effective_chat_model,
        "voice_model": settings.voice_model,
        "effective_voice_model": settings.effective_voice_model,
        "github_token_set": bool(settings.github_token),
        "database_url_set": bool(settings.database_url),
        "calcom_key_set": bool(settings.calcom_api_key),
    }

    # Test Groq connectivity
    try:
        from openai import OpenAI
        c = OpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key)
        r = c.chat.completions.create(
            model=settings.effective_chat_model,
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
        )
        results["groq_llm"] = "ok"
        results["groq_response"] = r.choices[0].message.content
    except Exception as exc:
        results["groq_llm"] = f"FAIL: {exc}"

    # Test DB connectivity
    try:
        from app import db
        conn = db.connect()
        count = conn.execute("SELECT count(*) FROM chunks;").fetchone()[0]
        conn.close()
        results["db"] = "ok"
        results["chunk_count"] = count
    except Exception as exc:
        results["db"] = f"FAIL: {exc}"

    # Test embeddings
    try:
        from app.embeddings import embed_query
        vec = embed_query("test")
        results["embeddings"] = f"ok (dim={len(vec)})"
    except Exception as exc:
        results["embeddings"] = f"FAIL: {exc}"

    return results


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return answer(req.message, req.history)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    import logging
    _log = logging.getLogger(__name__)

    def gen():
        try:
            for item in answer_stream(req.message, req.history):
                if isinstance(item, dict):  # final sources payload
                    yield f"event: sources\ndata: {json.dumps(item)}\n\n"
                else:
                    yield f"data: {json.dumps({'delta': item})}\n\n"
        except Exception as exc:
            _log.error("chat_stream error: %s", exc, exc_info=True)
            # Surface a useful message — not a wall of silence
            err = str(exc)
            if "api_key" in err.lower() or "authentication" in err.lower() or "401" in err:
                msg = "Authentication error — please check the API key configuration on the server."
            elif "rate" in err.lower() or "429" in err:
                msg = "Rate limit hit — please wait a moment and try again."
            elif "connect" in err.lower() or "timeout" in err.lower():
                msg = "Server is warming up — please try again in a few seconds."
            else:
                msg = f"Something went wrong on the server. Please try again. (Detail: {exc.__class__.__name__})"
            yield f"data: {json.dumps({'delta': msg})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
