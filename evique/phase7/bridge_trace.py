from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Sequence


RELATION_FUNNEL_STAGES = (
    "RELATION_ARTIFACT_ROWS",
    "RELATION_WINDOW_MATCH",
    "RELATION_ROLE_MATCH",
    "RELATION_CANDIDATES",
    "RELATION_NORMALIZED",
    "RELATION_SUPPORTED",
    "RELATION_RETRIEVED",
    "RELATION_VISIBLE",
)

RELATION_REJECTION_CODES = frozenset({
    "WRONG_VIDEO",
    "INVALID_OR_MISSING_TIME",
    "OUTSIDE_REFINED_WINDOW",
    "UNSUPPORTED_RELATION_TYPE",
    "MISSING_ENDPOINT",
    "ROLE_MISMATCH",
    "IDENTITY_CONFLICT",
    "INVALID_GEOMETRY",
    "LINEAGE_REJECTED",
    "CANDIDATE_BUDGET_REJECTED",
    "NORMALIZATION_REJECTED",
    "NO_REQUIREMENT_SUPPORT",
    "BUNDLE_NOT_SELECTED",
    "VISIBLE_BUDGET_REJECTED",
})

_FORBIDDEN = (
    "gold",
    "answer",
    "credential",
    "authorization",
    "api_key",
    "secret",
    "raw_media",
    "raw-media",
)


def _safe_text(value: object) -> str:
    text = str(value)
    folded = text.casefold()
    if any(token in folded for token in _FORBIDDEN):
        raise ValueError("forbidden content in Phase D trace")
    return text


def stable_trace_id(*parts: object) -> str:
    safe = [_safe_text(part) for part in parts]
    payload = json.dumps(safe, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class RelationFunnelTrace:
    _rows: tuple[str, ...]
    _stage_rows: dict[str, set[str]] = field(default_factory=dict)
    _rejections: dict[str, tuple[str, str]] = field(default_factory=dict)
    _last_stage_index: int = 0

    @classmethod
    def start(cls, row_ids: Sequence[str]) -> "RelationFunnelTrace":
        rows = tuple(dict.fromkeys(_safe_text(row_id) for row_id in row_ids))
        if len(rows) != len(tuple(row_ids)):
            raise ValueError("relation artifact row IDs must be unique")
        return cls(rows, {RELATION_FUNNEL_STAGES[0]: set(rows)})

    def advance(self, stage: str, row_ids: Iterable[str]) -> None:
        if stage not in RELATION_FUNNEL_STAGES:
            raise ValueError(f"unknown relation funnel stage: {stage}")
        index = RELATION_FUNNEL_STAGES.index(stage)
        if index != self._last_stage_index + 1:
            raise ValueError("relation funnel stage advanced out of order")
        rows = {_safe_text(row_id) for row_id in row_ids}
        previous = self._stage_rows[RELATION_FUNNEL_STAGES[index - 1]]
        if not rows <= previous:
            raise ValueError("relation funnel stage contains rows absent from prior stage")
        if rows & self._rejections.keys():
            raise ValueError("terminal relation row cannot advance")
        self._stage_rows[stage] = rows
        self._last_stage_index = index

    def reject(self, row_id: str, *, stage: str, reason: str) -> None:
        row_id = _safe_text(row_id)
        if reason not in RELATION_REJECTION_CODES:
            raise ValueError(f"unknown rejection reason: {reason}")
        if stage not in RELATION_FUNNEL_STAGES[1:]:
            raise ValueError(f"unknown rejection stage: {stage}")
        index = RELATION_FUNNEL_STAGES.index(stage)
        if index > self._last_stage_index + 1:
            raise ValueError("relation rejection stage is out of order")
        if row_id in self._rejections:
            raise ValueError("relation row already has a terminal outcome")
        prior = self._stage_rows[RELATION_FUNNEL_STAGES[index - 1]]
        current = self._stage_rows.get(stage, set())
        if row_id not in prior or row_id in current:
            raise ValueError("relation rejection does not reconcile with stage transition")
        self._rejections[row_id] = (stage, reason)

    def finalize(self) -> dict[str, Any]:
        final_rows = self._stage_rows.get(RELATION_FUNNEL_STAGES[-1], set())
        terminal = final_rows | self._rejections.keys()
        missing = set(self._rows) - terminal
        if missing:
            raise ValueError(f"unaccounted relation rows: {len(missing)}")
        if final_rows & self._rejections.keys():
            raise ValueError("relation rows have multiple terminal outcomes")
        rejection_counts = Counter(reason for _, reason in self._rejections.values())
        result = {
            "protocol_version": "phase-d-temporal-relation-v1",
            "stage_counts": {
                stage: len(self._stage_rows.get(stage, set()))
                for stage in RELATION_FUNNEL_STAGES
            },
            "stage_id_hashes": {
                stage: stable_trace_id(stage, *sorted(self._stage_rows.get(stage, set())))
                for stage in RELATION_FUNNEL_STAGES
            },
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "terminal_count": len(terminal),
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result
