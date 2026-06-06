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
    return {"status": "ok", "persona": settings.persona_name, "model": settings.chat_model}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return answer(req.message, req.history)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    def gen():
        try:
            for item in answer_stream(req.message, req.history):
                if isinstance(item, dict):  # final sources payload
                    yield f"event: sources\ndata: {json.dumps(item)}\n\n"
                else:
                    yield f"data: {json.dumps({'delta': item})}\n\n"
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("chat_stream error: %s", exc)
            msg = "I'm having trouble connecting right now. Please try again in a moment."
            yield f"data: {json.dumps({'delta': msg})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
