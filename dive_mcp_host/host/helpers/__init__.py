from datetime import UTC, datetime


def today_datetime() -> str:
    """The current date and time."""
    return datetime.now(tz=UTC).isoformat()
