"""Dates at the precision they were stated. YYYY, YYYY-MM or YYYY-MM-DD. Nothing is invented."""
import calendar
import datetime
import re

from .normalizer import time_value


def span(value):
    """The first and last day a stated date could mean."""
    parts = [int(x) for x in value.split("-")]
    if len(parts) == 1: return datetime.date(parts[0], 1, 1), datetime.date(parts[0], 12, 31)
    if len(parts) == 2: return datetime.date(parts[0], parts[1], 1), datetime.date(parts[0], parts[1], calendar.monthrange(parts[0], parts[1])[1])
    day = datetime.date(*parts)
    return day, day


def overlaps(a, b):
    """True when two stated dates cannot be put in order, such as 2025 and 2025-07."""
    (a1, a2), (b1, b2) = span(a), span(b)
    return a1 <= b2 and b1 <= a2


def step_before(value):
    """The step before a date, at the same precision: 2025-07 gives 2025-06."""
    parts = [int(x) for x in value.split("-")]
    if len(parts) == 1: return f"{parts[0] - 1}"
    if len(parts) == 2: return f"{parts[0] - 1}-12" if parts[1] == 1 else f"{parts[0]}-{parts[1] - 1:02d}"
    return (datetime.date(*parts) - datetime.timedelta(days=1)).isoformat()


def said_on(date_text):
    """The day a sentence was said, from a stored sentence date. None if it has no readable date."""
    return time_value(re.sub(r"^.*? on ", "", date_text or "")) if date_text else None


def covers(start, end, moment):
    """Whether a period, whose start or end may be unknown, touches a moment."""
    if start is None: return False
    first, last = span(moment)
    return span(start)[0] <= last and (end is None or span(end)[1] >= first)
