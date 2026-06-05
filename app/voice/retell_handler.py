"""Retell AI custom-LLM-server webhook.

Retell calls POST /voice/retell whenever the caller finishes speaking.
We run hybrid RAG + Haiku tool loop and stream the reply in Retell's SSE format.

Retell request body:
  {
    "interaction_type": "response_required" | "reminder_required" | "update_only",
    "response_id": <int>,
    "transcript": [{"role": "agent"|"user", "content": "<str>"}]
  }

Retell SSE response format (one event per chunk):
  data: {"response_id": N, "content": "...", "content_complete": false, "end_call": false}
  ...
  data: {"response_id": N, "content": "...", "content_complete": true,  "end_call": false}
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.voice.voice_answer import voice_respond

router = APIRouter(prefix="/voice", tags=["voice"])

_WORDS_PER_CHUNK = 8  # tune for TTS TTFB vs chunking overhead


def _retell_to_anthropic(transcript: list[dict]) -> list[dict]:
    """Map Retell role names to Anthropic role names."""
    role_map = {"agent": "assistant", "user": "user"}
    return [
        {"role": role_map.get(t["role"], "user"), "content": t["content"]}
        for t in transcript
        if t.get("content", "").strip()
    ]


@router.post("/retell")
async def retell_webhook(request: Request) -> StreamingResponse:
    body = await request.json()
    interaction_type = body.get("interaction_type", "response_required")

    if interaction_type == "update_only":
        return StreamingResponse(iter([]), media_type="text/event-stream")

    response_id: int = body.get("response_id", 0)
    transcript = body.get("transcript", [])
    messages = _retell_to_anthropic(transcript)

    def gen():
        text = voice_respond(messages)
        if not text:
            text = "I'm here — go ahead."

        words = text.split()
        for i in range(0, max(len(words), 1), _WORDS_PER_CHUNK):
            chunk_words = words[i : i + _WORDS_PER_CHUNK]
            is_last = (i + _WORDS_PER_CHUNK) >= len(words)
            payload = {
                "response_id": response_id,
                "content": " ".join(chunk_words) + ("" if is_last else " "),
                "content_complete": is_last,
                "end_call": False,
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
