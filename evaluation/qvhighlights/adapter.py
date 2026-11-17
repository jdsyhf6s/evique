"""Convert EVIQUE evidence into the QVHighlights prediction schema."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


_CONTAINERS = ("provenance", "record", "scope")


def _identity(item: Mapping[str, Any]) -> str | None:
    for source in (item, *(item.get(name, {}) for name in _CONTAINERS)):
        if not isinstance(source, Mapping):
            continue
        for key in ("video_id", "vid", "source_vid"):
            value = source.get(key)
            if value is not None:
                return str(value)
    return None


def _interval(item: Mapping[str, Any]) -> tuple[float, float] | None:
    for source in (item, *(item.get(name, {}) for name in _CONTAINERS)):
        if not isinstance(source, Mapping):
            continue
        start = source.get("start_time", source.get("start"))
        end = source.get("end_time", source.get("end"))
        if start is None or end is None:
            continue
        try:
            return float(start), float(end)
        except (TypeError, ValueError):
            return None
    return None


def _score(item: Mapping[str, Any]) -> float | None:
    for key in ("score", "retrieval_score", "rank_score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None
    return None


def query_to_prediction(
    query: Mapping[str, Any], retrieval: Mapping[str, Any], *, top_k: int = 10
) -> dict[str, Any]:
    """Build one prediction without consulting annotations or inventing intervals."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    qid = query.get("qid", query.get("query_id"))
    text = query.get("query", query.get("question"))
    vid = query.get("vid", query.get("video_id"))
    if qid is None or not isinstance(text, str) or not isinstance(vid, str):
        raise ValueError("query requires qid, query text, and vid")
    duration = float(query["duration"]) if query.get("duration") is not None else None

    ranked: list[tuple[float, int, float, float]] = []
    evidence = retrieval.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("retrieval evidence must be a list")
    for position, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            continue
        item_vid = _identity(item)
        if item_vid is not None and item_vid != vid:
            raise ValueError(f"video identity mismatch: expected {vid!r}, got {item_vid!r}")
        interval = _interval(item)
        score = _score(item)
        if interval is None or score is None:
            continue
        start, end = interval
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("invalid prediction interval")
        if duration is not None and end > duration:
            raise ValueError("prediction interval exceeds video duration")
        ranked.append((score, position, start, end))

    ranked.sort(key=lambda value: (-value[0], value[1]))
    windows: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for score, _, start, end in ranked:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        windows.append([start, end, score])
        if len(windows) == top_k:
            break
    return {"qid": qid, "query": text, "vid": vid, "pred_relevant_windows": windows}


def validate_submission(rows: Iterable[Mapping[str, Any]]) -> None:
    """Validate identity and temporal structure before official evaluation."""
    seen: set[str] = set()
    for row in rows:
        for key in ("qid", "query", "vid", "pred_relevant_windows"):
            if key not in row:
                raise ValueError(f"submission row is missing {key}")
        identity = str(row["qid"])
        if identity in seen:
            raise ValueError(f"duplicate qid: {identity}")
        seen.add(identity)
        windows = row["pred_relevant_windows"]
        if not isinstance(windows, list):
            raise ValueError("pred_relevant_windows must be a list")
        for window in windows:
            if not isinstance(window, list) or len(window) != 3:
                raise ValueError("each predicted window must contain start, end, and score")
            start, end, score = (float(value) for value in window)
            if not all(math.isfinite(value) for value in (start, end, score)) or not 0 <= start < end:
                raise ValueError("submission contains an invalid predicted window")
