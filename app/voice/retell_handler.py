"""Retell AI Custom LLM server — WebSocket endpoint.

Retell connects to wss://your-domain/llm-websocket/{call_id}
and sends JSON frames; we reply with streamed JSON frames.

Retell → server frame (response_required):
  {
    "interaction_type": "response_required" | "reminder_required" | "call_details",
    "response_id": <int>,
    "transcript": [{"role": "agent"|"user", "content": "<str>"}]
  }

Server → Retell frames (stream chunks, last has content_complete=true):
  {"response_id": N, "content": "...", "content_complete": false, "end_call": false}
  {"response_id": N, "content": "...", "content_complete": true,  "end_call": false}
"""
from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.voice.voice_answer import voice_respond

router = APIRouter(tags=["voice"])

_WORDS_PER_CHUNK = 8


def _retell_to_openai(transcript: list[dict]) -> list[dict]:
    role_map = {"agent": "assistant", "user": "user"}
    return [
        {"role": role_map.get(t["role"], "user"), "content": t["content"]}
        for t in transcript
        if t.get("content", "").strip()
    ]


@router.websocket("/llm-websocket/{call_id}")
async def retell_websocket(websocket: WebSocket, call_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            interaction_type = request.get("interaction_type", "")

            if interaction_type == "call_details":
                continue

            if interaction_type not in ("response_required", "reminder_required"):
                continue

            response_id: int = request.get("response_id", 0)
            transcript = request.get("transcript", [])
            messages = _retell_to_openai(transcript)

            text = voice_respond(messages) or "I'm here — go ahead."

            words = text.split()
            for i in range(0, max(len(words), 1), _WORDS_PER_CHUNK):
                chunk = words[i : i + _WORDS_PER_CHUNK]
                is_last = (i + _WORDS_PER_CHUNK) >= len(words)
                await websocket.send_json({
                    "response_id": response_id,
                    "content": " ".join(chunk) + ("" if is_last else " "),
                    "content_complete": is_last,
                    "end_call": False,
                })

    except WebSocketDisconnect:
        pass
