"""Fail-closed inference contract for QVHighlights query records."""

from __future__ import annotations

from typing import Any, Mapping


GT_FIELDS = frozenset(
    {
        "relevant_windows",
        "relevant_clip_ids",
        "saliency_scores",
        "answer",
        "label",
        "gt",
        "ground_truth",
        "gt_windows",
    }
)


def strip_ground_truth(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with every evaluation-only field removed."""
    return {key: value for key, value in row.items() if key not in GT_FIELDS}


def normalize_inference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an official query row into the EVIQUE inference identity schema."""
    clean = strip_ground_truth(row)
    query_id = clean.get("qid", clean.get("query_id"))
    question = clean.get("query", clean.get("question"))
    video_id = clean.get("vid", clean.get("video_id"))
    if query_id is None or str(query_id).strip() == "":
        raise ValueError("QVHighlights inference requires query ID")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QVHighlights inference requires query text")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("QVHighlights inference requires video ID")

    normalized: dict[str, Any] = {
        "dataset": "qvhighlights",
        "query_id": str(query_id),
        "question": question.strip(),
        "video_id": video_id.strip(),
    }
    if clean.get("duration") is not None:
        duration = float(clean["duration"])
        if duration <= 0:
            raise ValueError("QVHighlights duration must be positive")
        normalized["duration"] = duration
    assert_no_ground_truth(normalized)
    return normalized


def assert_no_ground_truth(row: Mapping[str, Any]) -> None:
    """Reject any inference payload containing evaluation-only information."""
    leaked = sorted(GT_FIELDS.intersection(row))
    if leaked:
        raise ValueError(f"ground-truth fields reached inference: {', '.join(leaked)}")
