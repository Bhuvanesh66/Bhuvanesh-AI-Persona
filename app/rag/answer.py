"""Grounded answer generation via GitHub Models (gpt-4o-mini, OpenAI-compatible).

Enforces: evidence-backed answers, citations, refusal when unknown,
fork-authorship honesty rule, and prompt-injection resistance.
Supports tool calling (check_availability, book_meeting) in both sync and stream paths.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from openai import OpenAI

from app.config import settings
from app.rag.retriever import Chunk, retrieve
from app.tools.cal_tools import TOOL_MAP, TOOL_SCHEMAS
from app.voice.voice_answer import BIO

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=settings.github_models_base_url,
        api_key=settings.github_token,
    )


def _system_prompt() -> str:
    from datetime import date
    p = settings.persona_name
    today = date.today().isoformat()
    return f"""You are the AI representative of {p} — a real software engineer and CS student. \
You speak in the first person on {p}'s behalf to a recruiter or visitor.

TODAY'S DATE: {today}

ALWAYS-AVAILABLE PROFILE (answer bio questions from this directly, no citation needed):
{BIO}

GROUNDING:
- For questions beyond the profile above, use the numbered CONTEXT blocks in the user turn.
- The retrieved corpus includes GitHub repositories, README files, source code, and commit history. Prefer that evidence for repo or project questions.
- Cite supporting blocks inline as [n]. NEVER invent facts, repos, dates, or metrics.
- If still unknown: "I don't have that detail — happy to book a call so you can ask {p} directly."

BOOKING:
- To check availability call check_availability(date='YYYY-MM-DD').
- To book a meeting call book_meeting(name, email, datetime_iso) where datetime_iso is UTC ISO 8601.
- Always collect name and email before booking.

FORK / AUTHORSHIP HONESTY:
- A block marked FORK is open-source {p} explored, not built. Never claim authorship.

FOCUS:
- Stay on topic: answer questions about the candidate's background, skills, and availability.
- Do not speculate about salary, offers, or anything not in the profile or context.

STYLE: confident, concise, recruiter-appropriate. Use markdown for structure."""


def _format_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        tag = "FORK" if c.is_fork else "ORIGINAL"
        loc = c.repo or c.source
        where = f"{loc} · {c.file_path}" if c.file_path else loc
        blocks.append(f"[{i}] ({tag} · {where})\n{c.content}")
    return "\n\n".join(blocks) if blocks else "(no additional context retrieved)"


def _build_messages(question: str, chunks: list[Chunk], history: list[dict] | None) -> list[dict]:
    history = history or []
    context = _format_context(chunks)
    user_turn = (
        f"CONTEXT:\n{context}\n\n"
        f"---\nQuestion: {question}"
    )
    return [
        {"role": "system", "content": _system_prompt()},
        *history,
        {"role": "user", "content": user_turn},
    ]


def _sources(chunks: list[Chunk]) -> list[dict]:
    return [
        {"n": i, "repo": c.repo, "file": c.file_path, "type": c.chunk_type,
         "is_fork": c.is_fork, "title": c.title}
        for i, c in enumerate(chunks, 1)
    ]


def _safe_retrieve(question: str) -> list[Chunk]:
    """Return chunks; return [] on any failure so the LLM still runs with BIO fallback."""
    try:
        return retrieve(question)
    except Exception as exc:
        logger.warning("Retrieval failed (BIO-only fallback): %s", exc)
        return []


def answer(question: str, history: list[dict] | None = None) -> dict:
    chunks = _safe_retrieve(question)
    messages = _build_messages(question, chunks, history)

    while True:
        resp = _client().chat.completions.create(
            model=settings.chat_model,
            max_tokens=1024,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        if resp.choices[0].finish_reason == "tool_calls":
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in (msg.tool_calls or [])
                ],
            })
            for tc in (msg.tool_calls or []):
                fn = TOOL_MAP.get(tc.function.name)
                try:
                    result = fn(**json.loads(tc.function.arguments)) if fn else {"error": "unknown tool"}
                    logger.info("Tool %s → %s", tc.function.name, result)
                except Exception as exc:
                    result = {"error": str(exc)}
                    logger.warning("Tool %s failed: %s", tc.function.name, exc)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            continue
        return {"answer": msg.content or "", "sources": _sources(chunks)}


def answer_stream(question: str, history: list[dict] | None = None):
    """Yield text deltas; final item is a dict with sources.
    Handles tool calls (check_availability, book_meeting) mid-stream.
    Retrieval failures are silently swallowed (BIO-only fallback).
    LLM failures propagate so api.py can surface a useful error message.
    """
    chunks = _safe_retrieve(question)
    messages = _build_messages(question, chunks, history)

    stream = _client().chat.completions.create(
        model=settings.chat_model,
        max_tokens=1024,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        stream=True,
    )

    tool_acc: dict[int, dict] = {}
    has_tools = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            yield delta.content

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

    if has_tools:
        tool_msgs = []
        result_msgs = []

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
        final = _client().chat.completions.create(
            model=settings.chat_model,
            max_tokens=1024,
            messages=follow_up,
            stream=True,
        )
        for chunk in final:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    yield {"sources": _sources(chunks)}
