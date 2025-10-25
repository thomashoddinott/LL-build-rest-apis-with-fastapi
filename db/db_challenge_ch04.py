"""Dummy logs database."""

from datetime import datetime, timedelta
from collections.abc import Iterator


_levels = ["INFO", "WARNING", "ERROR"]
_start_time = datetime(2024, 1, 1, 0, 0, 0)
_delta = timedelta(seconds=12.345)
_log_count = 1000


def query_logs(offset: int, limit: int) -> Iterator[dict]:
    """Return a list of logs."""
    if offset + limit > _log_count:
        limit = _log_count - offset

    for i in range(offset, offset + limit):
        yield {
            "level": _levels[i % 3],
            "time": _start_time + i * _delta,
            "message": f"Log message #{i:04d}",
        }
