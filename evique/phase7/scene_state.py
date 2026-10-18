from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math

from .schema import (
    EvidenceFact,
    ObservationSource,
    QuestionRequirement,
    RequirementType,
    SceneState,
    TimeRange,
)


@dataclass(frozen=True)
class SceneStateRules:
    version: str
    queue_rule_version: str = "queue_like_stationary_alignment_v1"
    queue_min_stationary: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("SceneStateRules version is required")
        if not isinstance(self.queue_rule_version, str) or not self.queue_rule_version:
            raise ValueError("queue rule version is required")
        if type(self.queue_min_stationary) is not int or self.queue_min_stationary < 2:
            raise ValueError("queue_min_stationary must be an integer of at least two")


def _overlaps(left: TimeRange, right: TimeRange) -> bool:
    return left.start <= right.end and right.start <= left.end


def _state_id(video_id: str, window: TimeRange, fact_ids: tuple[str, ...]) -> str:
    payload = json.dumps([video_id, window.start, window.end, fact_ids], separators=(",", ":"))
    return "scene-state:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_scene_states(facts: list[EvidenceFact], windows: list[TimeRange], *,
                       rules: SceneStateRules) -> list[SceneState]:
    states = []
    for window in windows:
        included = [fact for fact in facts if _overlaps(fact.time_span, window)]
        if not included:
            continue
        videos = {fact.video_id for fact in included}
        if len(videos) != 1:
            raise ValueError("a SceneState window cannot mix video IDs")
        video_id = next(iter(videos))
        if any(not fact.provenance or fact.provenance.get("video_id") != video_id for fact in included):
            raise ValueError("SceneState facts require consistent provenance")

        per_frame: dict[str, Counter] = defaultdict(Counter)
        object_sources: list[str] = []
        for fact in included:
            if fact.fact_type != "visual_object" or fact.source_kind is not ObservationSource.DETECTOR_OBSERVATION:
                continue
            label = fact.value.get("label") if isinstance(fact.value, dict) else None
            if not isinstance(label, str) or not label:
                continue
            frame = fact.frame_ids[0] if fact.frame_ids else f"time:{fact.time_span.start}"
            per_frame[frame][label] += 1
            object_sources.append(fact.fact_id)
        labels = sorted({label for counts in per_frame.values() for label in counts})
        object_counts = {
            label: max((counts[label] for counts in per_frame.values()), default=0)
            for label in labels
        }
        density = float(max((sum(counts.values()) for counts in per_frame.values()), default=0)) if per_frame else None

        motion_facts = [fact for fact in included if fact.fact_type in {"track_motion", "motion"}]
        motions = sorted({
            str(fact.value.get("motion")) for fact in motion_facts
            if isinstance(fact.value, dict) and fact.value.get("motion")
        })
        motion_summary = ", ".join(motions) if motions else None
        aligned: dict[str, list[str]] = defaultdict(list)
        for fact in motion_facts:
            value = fact.value if isinstance(fact.value, dict) else {}
            if value.get("motion") == "stationary" and value.get("alignment_group"):
                aligned[str(value["alignment_group"])].append(fact.fact_id)
        queue_group = next((group for group, ids in sorted(aligned.items())
                            if len(set(ids)) >= rules.queue_min_stationary), None)
        queue = None
        if queue_group is not None:
            queue_sources = sorted(set(aligned[queue_group]))
            queue = {
                "rule": rules.queue_rule_version,
                "alignment_group": queue_group,
                "stationary_count": len(queue_sources),
                "source_fact_ids": queue_sources,
                "semantic_limit": "queue_like_not_congestion",
            }

        caption_facts = [
            fact for fact in included if fact.fact_type == "scope_caption"
            and isinstance(fact.value, dict) and isinstance(fact.value.get("text"), str)
        ]
        scene_caption = next((fact.value["text"] for fact in caption_facts
                              if "congestion" in fact.value["text"].casefold()), None)
        fact_ids = tuple(fact.fact_id for fact in included)
        field_sources = {
            "object_categories": object_sources,
            "object_counts": object_sources,
            "traffic_density_proxy": object_sources,
            "motion_summary": [fact.fact_id for fact in motion_facts],
            "queue_like_observation": [] if queue is None else queue["source_fact_ids"],
            "scene_caption": [] if scene_caption is None else [
                fact.fact_id for fact in caption_facts if fact.value["text"] == scene_caption
            ],
        }
        states.append(SceneState(
            state_id=_state_id(video_id, window, fact_ids), video_id=video_id, time_window=window,
            object_categories=tuple(labels), object_counts=object_counts,
            motion_summary=motion_summary, queue_like_observation=queue,
            traffic_density_proxy=density, scene_caption=scene_caption,
            source_fact_ids=fact_ids,
            provenance={
                "video_id": video_id,
                "rules_version": rules.version,
                "object_counts_semantics": "maximum sampled detector count per frame",
                "traffic_density_proxy_rule": f"{rules.version}:density_max_visible_objects",
                "absence_supported": False,
                "field_sources": field_sources,
            },
        ))
    return states


def _anchor(state: SceneState) -> float | None:
    value = state.provenance.get("anchor_time")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def compare_scene_states(before: SceneState, after: SceneState,
                         requirement: QuestionRequirement) -> list[EvidenceFact]:
    if requirement.requirement_type is not RequirementType.STATE_CHANGE:
        return []
    before_anchor, after_anchor = _anchor(before), _anchor(after)
    if (
        before.video_id != after.video_id
        or not before.source_fact_ids or not after.source_fact_ids
        or before.time_window.end > after.time_window.start
        or before_anchor is None or after_anchor is None or before_anchor != after_anchor
        or not (before.time_window.end <= before_anchor <= after.time_window.start)
    ):
        return []
    source_ids = before.source_fact_ids + tuple(
        fact_id for fact_id in after.source_fact_ids if fact_id not in before.source_fact_ids
    )
    payload = json.dumps([requirement.requirement_id, before.state_id, after.state_id], separators=(",", ":"))
    return [EvidenceFact(
        fact_id="state-change:" + hashlib.sha256(payload.encode()).hexdigest()[:16],
        fact_type="state_change", video_id=before.video_id,
        time_span=TimeRange(before.time_window.start, after.time_window.end),
        role_bindings={},
        value={
            "before_state_id": before.state_id,
            "after_state_id": after.state_id,
            "anchor_time": before_anchor,
            "object_count_delta": {
                key: after.object_counts.get(key, 0) - before.object_counts.get(key, 0)
                for key in sorted(set(before.object_counts) | set(after.object_counts))
            },
            "causality_claimed": False,
        },
        source_evidence_ids=source_ids,
        source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
        derivation_rule="scene_state_change_v1",
        provenance={
            "video_id": before.video_id, "requirement_id": requirement.requirement_id,
            "before_state_id": before.state_id, "after_state_id": after.state_id,
        },
    )]
