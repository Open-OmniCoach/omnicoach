"""Pure-Python temporal interval merging, overlap, IoU, and 1-D Wasserstein math."""

from __future__ import annotations

import math
from collections.abc import Sequence


def merge_intervals(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or touching positive-length intervals into sorted disjoint tuples."""
    clean: list[tuple[float, float]] = []
    for raw_start, raw_end in spans:
        s, e = float(raw_start), float(raw_end)
        if not (math.isfinite(s) and math.isfinite(e)):
            raise ValueError(f"non-finite interval bounds ({raw_start!r}, {raw_end!r})")
        if e > s:
            clean.append((s, e))
    clean.sort()
    if not clean:
        return []
    merged: list[list[float]] = [[clean[0][0], clean[0][1]]]
    for s, e in clean[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _merged_overlap_duration(
    a: Sequence[tuple[float, float]],
    b: Sequence[tuple[float, float]],
) -> float:
    """Two-pointer O(N + M) overlap duration on already-sorted disjoint intervals."""
    i = j = 0
    overlaps: list[float] = []
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            overlaps.append(end - start)
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return math.fsum(overlaps)


def total_duration(spans: Sequence[tuple[float, float]]) -> float:
    """Return total merged duration across a collection of intervals."""
    merged = merge_intervals(spans)
    return math.fsum(e - s for s, e in merged)


def overlap_duration(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    """Return total overlapping duration between two interval collections."""
    return _merged_overlap_duration(merge_intervals(left), merge_intervals(right))


def temporal_iou(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    """Return set-level temporal IoU in [0.0, 1.0] between two interval collections."""
    a = merge_intervals(left)
    b = merge_intervals(right)
    inter = _merged_overlap_duration(a, b)
    a_dur = math.fsum(e - s for s, e in a)
    b_dur = math.fsum(e - s for s, e in b)
    union = a_dur + b_dur - inter
    if union <= 0.0:
        return 0.0
    return min(1.0, max(0.0, inter / union))


def w1_distance(
    pred: Sequence[tuple[float, float]],
    target: Sequence[tuple[float, float]],
) -> float:
    """Return exact 1-D Wasserstein (earth-mover) distance between uniform span densities."""
    p = merge_intervals(pred)
    t = merge_intervals(target)
    p_dur = math.fsum(e - s for s, e in p)
    t_dur = math.fsum(e - s for s, e in t)
    if p_dur <= 0.0 or t_dur <= 0.0:
        raise ValueError("empty temporal support")

    events: dict[float, float] = {}
    for s, e in p:
        events[s] = events.get(s, 0.0) + 1.0 / p_dur
        events[e] = events.get(e, 0.0) - 1.0 / p_dur
    for s, e in t:
        events[s] = events.get(s, 0.0) - 1.0 / t_dur
        events[e] = events.get(e, 0.0) + 1.0 / t_dur

    cdf = 0.0
    slope = 0.0
    dist = 0.0
    positions = sorted(events)
    prev = positions[0]
    for pos in positions:
        w = pos - prev
        nxt = cdf + slope * w
        if w > 0.0:
            if cdf * nxt >= 0.0:
                dist += w * (abs(cdf) + abs(nxt)) / 2.0
            else:
                rate = abs(nxt - cdf) / w
                dist += (cdf * cdf + nxt * nxt) / (2.0 * rate)
        cdf = nxt
        slope += events[pos]
        prev = pos
    return max(0.0, dist)


__all__ = [
    "merge_intervals",
    "overlap_duration",
    "temporal_iou",
    "total_duration",
    "w1_distance",
]
