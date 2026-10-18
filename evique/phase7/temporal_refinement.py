from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Sequence

from .schema import EvidenceFact, QuestionRequirementSet, TimeRange


class TemporalDirection(str, Enum):
    BEFORE = "before"
    ANCHOR = "anchor"
    AFTER = "after"
    OVERLAP = "overlap"
    NONE = "none"


@dataclass(frozen=True)
class RefinedWindow:
    window_id: str
    video_id: str
    time_range: TimeRange
    anchor_time: float | None
    temporal_direction: TemporalDirection
    subject_track_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]
    ranking_reasons: tuple[str, ...]
    original_candidate_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.window_id or not self.video_id:
            raise ValueError("refined window identity must be nonempty")
        if self.anchor_time is not None and (
            not math.isfinite(self.anchor_time) or self.anchor_time < 0
        ):
            raise ValueError("anchor_time must be finite and nonnegative")
        if len(set(self.subject_track_ids)) != len(self.subject_track_ids):
            raise ValueError("subject track IDs must be unique")
        if not self.source_fact_ids or len(set(self.source_fact_ids)) != len(self.source_fact_ids):
            raise ValueError("source fact IDs must be nonempty and unique")


_TEMPORAL_WORDS = frozenset({
    "before", "after", "during", "while", "when", "around", "then", "change", "changes",
})
_ANCHOR_WORDS = ("enter", "exit", "arriv", "depart", "event", "action")


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _window_id(fact: EvidenceFact, direction: TemporalDirection, anchor_time: float | None) -> str:
    payload = json.dumps(
        [fact.video_id, fact.fact_id, fact.time_span.start, fact.time_span.end, direction.value, anchor_time],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "phase-d-window-" + hashlib.sha256(payload).hexdigest()[:20]


def _tracks(fact: EvidenceFact) -> tuple[str, ...]:
    values = list(fact.role_bindings.values())
    if isinstance(fact.value, dict):
        for key in ("track_id", "subject_track_id", "reference_track_id"):
            value = fact.value.get(key)
            if isinstance(value, str) and value:
                values.append(value)
    return tuple(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _anchor_candidate(fact: EvidenceFact, query_tokens: set[str], rank: int) -> tuple[int, int, int]:
    text = " ".join((fact.fact_type, str(fact.value))).casefold()
    is_event = any(token in text for token in _ANCHOR_WORDS)
    query_match = any(token.startswith(stem) and stem in text for token in query_tokens for stem in _ANCHOR_WORDS)
    return (0 if query_match else 1, 0 if is_event else 1, rank)


def _requested_directions(requirements: QuestionRequirementSet) -> set[TemporalDirection]:
    result: set[TemporalDirection] = set()
    for requirement in requirements.requirements:
        reference = requirement.temporal_relation
        if reference is None:
            continue
        if reference.direction in {"before", "before_after"}:
            result.add(TemporalDirection.BEFORE)
        if reference.direction in {"after", "before_after"}:
            result.add(TemporalDirection.AFTER)
        if reference.direction == "around":
            result.update({TemporalDirection.ANCHOR, TemporalDirection.OVERLAP})
    return result


def _direction(fact: EvidenceFact, anchor: EvidenceFact) -> TemporalDirection:
    anchor_time = (anchor.time_span.start + anchor.time_span.end) / 2
    if fact.fact_id == anchor.fact_id:
        return TemporalDirection.ANCHOR
    if fact.time_span.start <= anchor_time <= fact.time_span.end:
        return TemporalDirection.OVERLAP
    if fact.time_span.end <= anchor_time:
        return TemporalDirection.BEFORE
    return TemporalDirection.AFTER


def _fallback(facts: Sequence[EvidenceFact], max_windows: int) -> tuple[RefinedWindow, ...]:
    return tuple(
        RefinedWindow(
            window_id=_window_id(fact, TemporalDirection.NONE, None),
            video_id=fact.video_id,
            time_range=fact.time_span,
            anchor_time=None,
            temporal_direction=TemporalDirection.NONE,
            subject_track_ids=_tracks(fact),
            source_fact_ids=(fact.fact_id,),
            ranking_reasons=("BASELINE_ORDER",),
            original_candidate_ranks=(rank,),
        )
        for rank, fact in enumerate(facts[:max_windows])
    )


def refine_temporal_windows(
    query: str,
    requirements: QuestionRequirementSet,
    facts: Sequence[EvidenceFact],
    *,
    max_windows: int = 2,
) -> tuple[RefinedWindow, ...]:
    if max_windows != 2:
        raise ValueError("max_windows must equal 2")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty")
    if any(not isinstance(fact, EvidenceFact) for fact in facts):
        raise TypeError("facts must contain EvidenceFact values")
    if not facts:
        return ()
    video_ids = {fact.video_id for fact in facts}
    if len(video_ids) != 1:
        raise ValueError("refined windows cannot span videos")

    query_tokens = _tokens(query)
    requested = _requested_directions(requirements)
    if not requested or not (query_tokens & _TEMPORAL_WORDS):
        return _fallback(facts, max_windows)

    anchor_rank, anchor = min(
        enumerate(facts),
        key=lambda item: _anchor_candidate(item[1], query_tokens, item[0]),
    )
    if _anchor_candidate(anchor, query_tokens, anchor_rank)[1] != 0:
        return _fallback(facts, max_windows)
    anchor_time = (anchor.time_span.start + anchor.time_span.end) / 2
    anchor_tracks = set(_tracks(anchor))
    candidates = []
    for rank, fact in enumerate(facts):
        direction = _direction(fact, anchor)
        fact_tokens = _tokens(fact.fact_type) | _tokens(fact.value)
        direction_match = direction in requested or direction in {
            TemporalDirection.ANCHOR,
            TemporalDirection.OVERLAP,
        }
        action_match = bool(fact_tokens & query_tokens) or fact.fact_id == anchor.fact_id
        track_match = bool(anchor_tracks & set(_tracks(fact)))
        score = fact.provenance.get("retrieval_score", 0.0)
        score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
        reasons = tuple(
            reason for condition, reason in (
                (direction_match, "TEMPORAL_DIRECTION_MATCH"),
                (action_match, "ACTION_EVENT_CUE_MATCH"),
                (track_match, "SUBJECT_TRACK_OVERLAP"),
                (score != 0.0, "BASE_SEMANTIC_RELEVANCE"),
            ) if condition
        ) or ("BASELINE_ORDER",)
        span = fact.time_span.end - fact.time_span.start
        window = RefinedWindow(
            window_id=_window_id(fact, direction, anchor_time),
            video_id=fact.video_id,
            time_range=fact.time_span,
            anchor_time=anchor_time,
            temporal_direction=direction,
            subject_track_ids=_tracks(fact),
            source_fact_ids=(fact.fact_id,),
            ranking_reasons=reasons,
            original_candidate_ranks=(rank,),
        )
        key = (
            0 if direction_match else 1,
            0 if action_match else 1,
            0 if track_match else 1,
            -score,
            rank,
            span,
            fact.time_span.start,
            window.window_id,
        )
        candidates.append((key, window))
    candidates.sort(key=lambda item: item[0])
    return tuple(window for _, window in candidates[:max_windows])
