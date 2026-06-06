"""Cal.com v2 tools: check availability and book a meeting."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://api.cal.com/v2"
_TIMEOUT = 10

# Cal.com v2 requires different api-version headers per endpoint
_SLOTS_API_VERSION = "2024-09-04"
_BOOKINGS_API_VERSION = "2026-02-25"

# Availability window: 9 AM to 5 PM daily
_HOUR_START = 9
_HOUR_END = 17  # 5 PM in 24-hour format


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


def _validate_date(date_str: str) -> tuple[bool, str]:
    """Validate date format (YYYY-MM-DD) and ensure it's in the future.
    
    Returns: (is_valid, error_message)
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False, f"Invalid date format. Please use YYYY-MM-DD (e.g., 2026-06-10)."
    
    today = datetime.now().date()
    if dt.date() < today:
        return False, f"Date '{date_str}' is in the past. Please select a future date."
    
    if (dt.date() - today).days > 90:
        return False, f"Date '{date_str}' is too far in the future. Please book within 90 days."
    
    return True, ""


def _validate_name(name: str) -> tuple[bool, str]:
    """Validate that name is not empty."""
    if not name or not name.strip():
        return False, "Name cannot be empty. Please provide your full name."
    return True, ""


def _validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    email = email.strip()
    # Simple regex for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, f"Invalid email format: '{email}'. Please provide a valid email address (e.g., name@example.com)."
    return True, ""


def _validate_datetime_iso(datetime_str: str) -> tuple[bool, str]:
    """Validate ISO 8601 UTC datetime and ensure it's within 9 AM - 5 PM.
    
    Returns: (is_valid, error_message)
    """
    try:
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
    except ValueError:
        return False, f"Invalid datetime format: '{datetime_str}'. Use ISO 8601 UTC (e.g., 2026-06-10T14:00:00Z)."
    
    now = datetime.now(dt.tzinfo)
    if dt < now:
        return False, f"Datetime '{datetime_str}' is in the past. Please select a future time."
    
    hour = dt.hour
    if hour < _HOUR_START or hour >= _HOUR_END:
        return False, f"Time {hour}:00 is outside business hours (9 AM–5 PM). Please select a time between 9 AM and 5 PM."
    
    return True, ""


def check_availability(date: str) -> dict:
    """Return available slots for the given date (YYYY-MM-DD).

    Uses GET /v2/slots with eventTypeId + start/end range (correct Cal.com v2 API).
    Validates date format and ensures slots are within 9 AM - 5 PM.
    """
    # Validate date input
    is_valid, error = _validate_date(date)
    if not is_valid:
        return {"error": error}
    
    start = f"{date}T09:00:00Z"
    end = f"{date}T17:00:00Z"
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
        return {"error": f"Failed to check availability. Please try again. (Status: {resp.status_code})"}
    
    data = resp.json()
    # Response: {"status": "success", "data": {"YYYY-MM-DD": [{"start": "..."}], ...}}
    slots_by_date = data.get("data", {})
    times = []
    for key, slot_list in slots_by_date.items():
        if key.startswith(date):
            times.extend(s["start"] for s in slot_list if s.get("start"))
    
    if not times:
        return {
            "date": date,
            "slots": [],
            "count": 0,
            "message": f"No available slots on {date}. Please try a different date."
        }
    
    return {"date": date, "slots": times, "count": len(times)}


def _clean_email(email: str) -> str:
    """Normalize email spoken by a caller.

    Handles formats like:
      - 'john dot doe at gmail dot com'
      - 'r o y c e at gmail dot com'       (space-separated letters)
      - 'roycerkg@gmail.com'               (already clean)
    """
    e = email.strip().lower()

    # Normalise spoken separators to standard chars
    e = e.replace(" dot com", ".com")
    e = e.replace(" dot ", ".")
    e = e.replace(" at ", "@")
    e = e.replace("at the rate ", "@")
    e = e.replace(" dot net", ".net")
    e = e.replace(" dot org", ".org")
    e = e.replace(" dot io", ".io")
    e = e.replace(" dot co", ".co")

    # After keyword substitution, collapse remaining spaces
    # (handles 'r o y c e' → 'royce' while preserving '@' and '.')
    e = "".join(e.split())
    return e


def book_meeting(name: str, email: str, datetime_iso: str, notes: str = "") -> dict:
    """Book a 30-min intro call. datetime_iso: ISO 8601 UTC e.g. 2026-06-10T10:00:00Z
    
    Validates name, email, and datetime before booking. Returns helpful error messages
    if any input is invalid.
    """
    # Validate name
    is_valid, error = _validate_name(name)
    if not is_valid:
        return {"error": error}
    
    # Validate email
    is_valid, error = _validate_email(email)
    if not is_valid:
        return {"error": error}
    
    # Validate datetime and business hours
    is_valid, error = _validate_datetime_iso(datetime_iso)
    if not is_valid:
        return {"error": error}
    
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
        return {"error": f"Failed to book meeting. Please try again. (Status: {resp.status_code})"}
    
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
