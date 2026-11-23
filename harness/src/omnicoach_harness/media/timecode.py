"""`format_timecode`: seconds as `MM:SS.d`."""

from __future__ import annotations


def format_timecode(value: float) -> str:
    units = round(max(0.0, value) * 10)
    whole, fraction = divmod(units, 10)
    minutes, seconds = divmod(whole, 60)
    return f"{minutes:02d}:{seconds:02d}.{fraction}"
