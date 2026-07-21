"""
inventory.py  -  the mock room-booking store (Layer B).

The single source of truth for what is already booked, so `check_availability`
and `create_booking` (in agent.py) depend on prior bookings: once a room is booked
for a date range, it is unavailable to the next caller for any OVERLAPPING range.
That is what makes "the next customer can't book it" real.

Stdlib only, in-memory, thread-safe. State lives for the process lifetime (a
restart clears it); evals/tests call reset_inventory() for deterministic runs.
"""

from __future__ import annotations

import threading
from datetime import date, datetime

# First confirmation number. Kept so the FIRST booking after a reset is VH-4827,
# the literal the evals/tests/mock assert. Later bookings increment (VH-4828...).
_BASE_CONFIRMATION = 4827

# strptime defaults year-less dates to 1900; pin them to a stable reference year
# so "August 12"-style ranges compare consistently (and don't look ancient).
_REFERENCE_YEAR = 2026
_DATE_FORMATS = ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%B %d", "%b %d")

_lock = threading.Lock()
_bookings: list[dict] = []


def parse_date(value) -> date | None:
    """Parse a caller/mock date into a date, or None if unparseable.

    Accepts "August 12" / "Aug 12" (month + day, no year), ISO "2026-08-12",
    and the same with a trailing year. Year-less inputs get _REFERENCE_YEAR.
    """
    if isinstance(value, date):
        return value
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=_REFERENCE_YEAR)
        return parsed
    return None


def parse_range(check_in, check_out) -> "tuple[date | None, date | None]":
    """Parse a (check_in, check_out) pair, rolling check_out to the NEXT year on a
    genuine new-year wrap (e.g. Dec 30 -> Jan 2), mirroring agent._validate_stay so the
    interval the agent validated is the same interval inventory stores/compares. Without
    this, a wrap re-parses to end < start and overlap math silently allows double-booking.
    Returns (None, None) if either endpoint is unparseable."""
    start = parse_date(check_in)
    end = parse_date(check_out)
    if start is None or end is None:
        return None, None
    if end <= start:
        end = end.replace(year=end.year + 1)
    return start, end


def _overlaps(a_in: date, a_out: date, b_in: date, b_out: date) -> bool:
    """True if [a_in, a_out) and [b_in, b_out) intersect (checkout day is free)."""
    return a_in < b_out and b_in < a_out


def _conflict(room_key: str, start: date, end: date) -> bool:
    """Caller must hold _lock. True if a stored booking blocks [start, end)."""
    for booking in _bookings:
        if booking["room_key"] != room_key:
            continue
        booked_in, booked_out = parse_range(booking["check_in"], booking["check_out"])
        if booked_in and booked_out and _overlaps(start, end, booked_in, booked_out):
            return True
    return False


def is_available(room_key: str, check_in, check_out) -> bool:
    """Read-only check for check_availability: True when no stored booking for
    `room_key` overlaps [check_in, check_out).

    An unparseable/degenerate range returns True  -  availability is not where bad
    input is rejected (run_tool validates the dates first).
    """
    start, end = parse_range(check_in, check_out)
    if start is None or end is None:
        return True
    with _lock:
        return not _conflict(room_key, start, end)


def book_if_available(room_key: str, check_in, check_out,
                     guest_name=None, contact=None) -> str | None:
    """Atomically book the room for the range IF it is free.

    Returns the confirmation code (VH-4827, VH-4828, ...), or None if the room is
    already taken. The overlap check and the append happen under a SINGLE lock, so
    two concurrent callers can never double-book the same room+dates.
    """
    start, end = parse_range(check_in, check_out)
    with _lock:
        if start is not None and end is not None and _conflict(room_key, start, end):
            return None
        code = f"VH-{_BASE_CONFIRMATION + len(_bookings)}"
        _bookings.append({
            "confirmation": code,
            "room_key": room_key,
            "check_in": str(check_in),
            "check_out": str(check_out),
            "guest_name": guest_name,
            "contact": contact,
        })
        return code


def active_bookings() -> list[dict]:
    """A shallow copy of stored bookings (for tests/debugging)."""
    with _lock:
        return list(_bookings)


def reset_inventory() -> None:
    """Clear all bookings. Evals/tests/demo call this for deterministic runs."""
    with _lock:
        _bookings.clear()
