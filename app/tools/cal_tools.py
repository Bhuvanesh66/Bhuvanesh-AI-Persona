"""Cal.com v2 tools: check availability and book a meeting."""
from __future__ import annotations

import logging

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.cal.com/v2"
_TIMEOUT = 10

# Cal.com v2 requires different api-version headers per endpoint
_SLOTS_API_VERSION = "2024-09-04"
_BOOKINGS_API_VERSION = "2026-02-25"


def _slots_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.calcom_api_key}",
        "cal-api-version": _SLOTS_API_VERSION,
    }


def _bookings_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.calcom_api_key}",
        "cal-api-version": _BOOKINGS_API_VERSION,
        "Content-Type": "application/json",
    }


def check_availability(date: str) -> dict:
    """Return available slots for the given date (YYYY-MM-DD).

    Uses GET /v2/slots with eventTypeId + start/end range (correct Cal.com v2 API).
    """
    start = f"{date}T00:00:00Z"
    end = f"{date}T23:59:59Z"
    resp = requests.get(
        f"{_BASE}/slots",
        params={
            "eventTypeId": settings.calcom_event_type_id,
            "start": start,
            "end": end,
        },
        headers=_slots_headers(),
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        logger.warning("Cal.com slots → %s: %s", resp.status_code, resp.text[:300])
    resp.raise_for_status()
    data = resp.json()
    # Response: {"status": "success", "data": {"YYYY-MM-DD": [{"start": "..."}], ...}}
    slots_by_date = data.get("data", {})
    times = []
    for key, slot_list in slots_by_date.items():
        if key.startswith(date):
            times.extend(s["start"] for s in slot_list if s.get("start"))
    return {"date": date, "slots": times, "count": len(times)}


def _clean_email(email: str) -> str:
    """Normalize email spoken by a caller, e.g. 'john dot doe at gmail dot com'."""
    e = email.strip().lower()
    e = e.replace(" dot ", ".").replace(" at ", "@").replace(" ", "")
    return e


def book_meeting(name: str, email: str, datetime_iso: str, notes: str = "") -> dict:
    """Book a 30-min intro call. datetime_iso: ISO 8601 UTC e.g. 2026-06-10T10:00:00Z"""
    clean = _clean_email(email)
    body: dict = {
        "eventTypeId": settings.calcom_event_type_id,
        "start": datetime_iso,
        "attendee": {
            "name": name,
            "email": clean,
            "timeZone": "UTC",
            "language": "en",
        },
        # Cal.com requires standard form fields in bookingFieldsResponses
        # even when already present in attendee — omitting email causes 400
        "bookingFieldsResponses": {
            "name": name,
            "email": clean,
        },
    }
    if notes:
        body["bookingFieldsResponses"]["notes"] = notes

    resp = requests.post(
        f"{_BASE}/bookings",
        json=body,
        headers=_bookings_headers(),
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        logger.warning("Cal.com booking → %s: %s", resp.status_code, resp.text[:300])
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
            "description": "Check available time slots on Bhuvanesh's calendar for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format, e.g. 2026-06-10.",
                    },
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
