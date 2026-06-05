"""Cal.com v2 tools: check availability and book a meeting."""
from __future__ import annotations

import logging

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.cal.com/v2"
_TIMEOUT = 10
_API_VERSION = "2024-09-04"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.calcom_api_key}",
        "cal-api-version": _API_VERSION,
        "Content-Type": "application/json",
    }


def check_availability(date: str) -> dict:
    """Return available slots for the given date (YYYY-MM-DD)."""
    # Build URL as a plain string — using params= would encode '/' as '%2F' which Cal.com rejects
    cal_link = f"{settings.calcom_username}/{settings.calcom_event_slug}"
    url = (
        f"{_BASE}/slots/available"
        f"?calLink={cal_link}"
        f"&startTime={date}T00:00:00.000Z"
        f"&endTime={date}T23:59:59.999Z"
    )
    resp = requests.get(url, headers=_headers(), timeout=_TIMEOUT)
    if not resp.ok:
        logger.warning("Cal.com slots → %s %s: %s", resp.status_code, url, resp.text[:300])
    resp.raise_for_status()
    data = resp.json()
    slots_by_date = data.get("data", {}).get("slots", {})
    times = []
    for key, slot_list in slots_by_date.items():
        if key.startswith(date):
            times.extend(s["time"] for s in slot_list)
    return {"date": date, "slots": times, "count": len(times)}


def book_meeting(name: str, email: str, datetime_iso: str, notes: str = "") -> dict:
    """Book a 30-min intro call. datetime_iso: ISO 8601 UTC e.g. 2026-06-10T10:00:00Z"""
    body: dict = {
        "eventTypeId": settings.calcom_event_type_id,
        "start": datetime_iso,
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": "UTC",
            "language": "en",
        },
        "metadata": {},
    }
    if notes:
        body["metadata"]["notes"] = notes

    resp = requests.post(
        f"{_BASE}/bookings",
        json=body,
        headers=_headers(),
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    booking = resp.json().get("data", resp.json())
    return {
        "booking_id": booking.get("id"),
        "uid": booking.get("uid"),
        "status": booking.get("status"),
        "start": booking.get("start") or booking.get("startTime"),
        "end": booking.get("end") or booking.get("endTime"),
        "title": booking.get("title"),
    }


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Check open meeting slots on Cal.com for a given date. "
                "Always call this before booking so the caller can pick a time. "
                "Use today's date or a date the caller specifies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (UTC). Use today's date if not specified.",
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
                        "description": "ISO 8601 UTC datetime e.g. 2026-06-10T10:00:00Z. Must be a future date.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional context (company, role, topic).",
                    },
                },
                "required": ["name", "email", "datetime_iso"],
            },
        },
    },
]

TOOL_MAP = {"check_availability": check_availability, "book_meeting": book_meeting}
