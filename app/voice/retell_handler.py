"""Retell AI Custom LLM server — async WebSocket with real-time streaming.

Words are sent to Retell as tokens arrive from the LLM, so the caller
hears the first word in ~2s instead of waiting for the full response.

Protocol:
  Retell → wss://your-domain/llm-websocket/{call_id}
  Frames in:  {"interaction_type": "response_required"|"call_details", "response_id": N, "transcript": [...]}
  Frames out: {"response_id": N, "content": "...", "content_complete": false|true, "end_call": false}
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from app.config import settings
from app.tools.cal_tools import TOOL_MAP, TOOL_SCHEMAS
from app.voice.voice_answer import _GREETING, _system

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

_CHUNK_WORDS = 4

_END_PHRASES = {
    "bye", "goodbye", "good bye", "end the call", "end call",
    "hang up", "gotta go", "talk later", "take care", "see you",
    "that's all", "thats all", "i'm done", "im done",
}

_FAREWELL = "Goodbye! It was great talking with you. Have a wonderful day!"


def _wants_to_end(turns: list[dict]) -> bool:
    last = next((t["content"].lower() for t in reversed(turns) if t["role"] == "user"), "")
    return any(phrase in last for phrase in _END_PHRASES)


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.github_models_base_url,
        api_key=settings.github_token,
    )


def _retell_to_openai(transcript: list[dict]) -> list[dict]:
    role_map = {"agent": "assistant", "user": "user"}
    return [
        {"role": role_map.get(t["role"], "user"), "content": t["content"]}
        for t in transcript
        if t.get("content", "").strip()
    ]


def _build_messages(turns: list[dict]) -> list[dict]:
    """Build messages for the LLM. Bio is in the system prompt so no RAG needed for voice."""
    messages: list[dict] = [{"role": "system", "content": _system()}]
    for i, turn in enumerate(turns):
        if turn["role"] == "user" and i == len(turns) - 1:
            messages.append({"role": "user", "content": f"Caller: {turn['content']}"})
        else:
            messages.append(turn)
    return messages


async def _send_chunk(ws: WebSocket, response_id: int, text: str, complete: bool, end_call: bool = False) -> None:
    await ws.send_json({
        "response_id": response_id,
        "content": text,
        "content_complete": complete,
        "end_call": end_call,
    })


async def _stream_reply(messages: list[dict], response_id: int, ws: WebSocket) -> None:
    """Stream LLM tokens to Retell as they arrive. Handles tool calls too."""
    client = _client()
    stream = await client.chat.completions.create(
        model=settings.voice_model,
        max_tokens=256,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        stream=True,
    )

    buffer = ""
    tool_acc: dict[int, dict] = {}
    has_tools = False

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        # ── Stream text tokens ────────────────────────────────────────────────
        if delta.content:
            buffer += delta.content
            words = buffer.split(" ")
            # Send complete word groups; keep the last (possibly incomplete) word
            while len(words) > _CHUNK_WORDS:
                packet = " ".join(words[:_CHUNK_WORDS]) + " "
                buffer = " ".join(words[_CHUNK_WORDS:])
                words = buffer.split(" ")
                await _send_chunk(ws, response_id, packet, False)

        # ── Accumulate tool call fragments ────────────────────────────────────
        if delta.tool_calls:
            has_tools = True
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_acc:
                    tool_acc[idx] = {"id": "", "name": "", "args": ""}
                if tc.id:
                    tool_acc[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_acc[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_acc[idx]["args"] += tc.function.arguments

    # ── Execute tools and get a final reply ──────────────────────────────────
    if has_tools:
        tool_msgs: list[dict] = []
        result_msgs: list[dict] = []

        for idx in sorted(tool_acc):
            tc = tool_acc[idx]
            fn = TOOL_MAP.get(tc["name"])
            try:
                args = json.loads(tc["args"] or "{}")
                result = fn(**args) if fn else {"error": f"unknown tool: {tc['name']}"}
                logger.info("Tool %s(%s) → %s", tc["name"], args, result)
            except Exception as exc:
                result = {"error": str(exc)}
                logger.warning("Tool %s failed: %s", tc["name"], exc)

            tool_msgs.append({
                "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": tc["args"]},
            })
            result_msgs.append({
                "role": "tool", "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

        follow_up = list(messages) + [
            {"role": "assistant", "content": None, "tool_calls": tool_msgs},
            *result_msgs,
        ]
        final = await client.chat.completions.create(
            model=settings.voice_model,
            max_tokens=256,
            messages=follow_up,
        )
        buffer = (final.choices[0].message.content or "").strip()

    # ── Send remaining buffer as the final packet ─────────────────────────────
    final_text = buffer.strip() or ("I'm here — go ahead." if not has_tools else "")
    await _send_chunk(ws, response_id, final_text, True)


@router.websocket("/llm-websocket/{call_id}")
async def retell_websocket(websocket: WebSocket, call_id: str) -> None:
    await websocket.accept()
    logger.info("Retell WebSocket connected: %s", call_id)
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            itype = req.get("interaction_type", "")

            if itype == "call_details":
                continue
            if itype not in ("response_required", "reminder_required"):
                continue

            response_id: int = req.get("response_id", 0)
            turns = _retell_to_openai(req.get("transcript", []))

            # Empty transcript = call just started → greet immediately
            if not any(t["role"] == "user" for t in turns):
                await _send_chunk(websocket, response_id, _GREETING, True)
                continue

            # End call if user said bye / goodbye
            if _wants_to_end(turns):
                await _send_chunk(websocket, response_id, _FAREWELL, True, end_call=True)
                continue

            messages = _build_messages(turns)
            await _stream_reply(messages, response_id, websocket)

    except WebSocketDisconnect:
        logger.info("Retell WebSocket disconnected: %s", call_id)
    except Exception as exc:
        logger.error("Retell WebSocket error (%s): %s", call_id, exc)
        try:
            await _send_chunk(websocket, 0,
                "I'm having a connection issue. Please try again in a moment.", True)
        except Exception:
            pass
