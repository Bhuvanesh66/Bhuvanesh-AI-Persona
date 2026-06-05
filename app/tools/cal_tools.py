"""Cal.com v1 tools: check availability and book a meeting.

These are called by the voice agent (Haiku tool loop) and can also be
wired into the chat interface later.
"""
from __future__ import annotations

import requests

from app.config import settings

_BASE = "https://api.cal.com/v1"
_TIMEOUT = 10


def check_availability(date: str) -> dict:
    """Return available ISO-8601 slots for the given date (YYYY-MM-DD)."""
    resp = requests.get(
        f"{_BASE}/availability",
        params={
            "apiKey": settings.calcom_api_key,
            "eventTypeId": settings.calcom_event_type_id,
            "dateFrom": date,
            "dateTo": date,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    slots = data.get("slots", {}).get(date, [])
    return {"date": date, "slots": [s["time"] for s in slots]}


def book_meeting(name: str, email: str, datetime_iso: str, notes: str = "") -> dict:
    """Create a Cal.com booking for a 30-min call.

    datetime_iso: ISO 8601 UTC, e.g. "2026-06-10T10:00:00Z"
    """
    resp = requests.post(
        f"{_BASE}/bookings",
        params={"apiKey": settings.calcom_api_key},
        json={
            "eventTypeId": settings.calcom_event_type_id,
            "start": datetime_iso,
            "responses": {
                "name": name,
                "email": email,
                "notes": notes or "",
            },
            "timeZone": "UTC",
            "language": "en",
            "metadata": {},
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "booking_id": data.get("id"),
        "uid": data.get("uid"),
        "status": data.get("status"),
        "start": data.get("startTime"),
        "end": data.get("endTime"),
        "title": data.get("title"),
    }


# OpenAI-compatible tool definitions (used by voice_answer.py tool loop)
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check open meeting slots on Cal.com for a given date. "
                "Use this before booking so the caller can pick a time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check, YYYY-MM-DD format (UTC).",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a 30-minute intro call with Bhuvanesh on Cal.com.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Caller's full name."},
                    "email": {"type": "string", "description": "Caller's email address."},
                    "datetime_iso": {
                        "type": "string",
                        "description": "ISO 8601 UTC datetime, e.g. 2026-06-10T10:00:00Z",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional context for the meeting (company, role).",
                    },
                },
                "required": ["name", "email", "datetime_iso"],
            },
        },
    },
]

TOOL_MAP = {"check_availability": check_availability, "book_meeting": book_meeting}
