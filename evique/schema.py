from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Provenance:
    video_name: str
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None
    segment_id: str | None = None
    segment_index: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    frame_times: list[float] = field(default_factory=list)
    source: str = "segment_store"


@dataclass
class ScopeViewRecord:
    id: str
    node_id: str
    video_name: str
    segment_id: str
    segment_index: str
    start_time: float | None
    end_time: float | None
    scene_tags: list[str]
    caption: str
    transcript: str
    text: str
    provenance: Provenance
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None
    caption_status: str = "unknown"
    quality_status: str = "unknown"


@dataclass
class TargetViewRecord:
    id: str
    node_id: str
    video_name: str
    segment_id: str
    label: str
    aliases: list[str]
    attributes: list[str]
    confidence: float
    bbox: Any
    text: str
    provenance: Provenance
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class TrackViewRecord:
    id: str
    node_id: str
    video_name: str
    label: str
    object_ids: list[str]
    segment_ids: list[str]
    start_time: float | None
    end_time: float | None
    motion_summary: str
    action_tags: list[str]
    evidence_text: str = ""
    action_keywords: list[str] = field(default_factory=list)
    direction_keywords: list[str] = field(default_factory=list)
    state_keywords: list[str] = field(default_factory=list)
    related_event_ids: list[str] = field(default_factory=list)
    neighbor_track_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class EventViewRecord:
    id: str
    node_id: str
    video_name: str
    start_time: float | None
    end_time: float | None
    event_type: str
    summary: str
    state_signature: dict[str, Any]
    action_tags: list[str]
    related_segment_ids: list[str]
    related_object_ids: list[str]
    related_track_ids: list[str]
    previous_event_ids: list[str] = field(default_factory=list)
    next_event_ids: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class AdaptiveEventViewRecord:
    id: str
    node_id: str
    event_id: str
    event_type: str
    event_segmentation_mode: str
    event_source: str
    video_name: str
    start_time: float
    end_time: float
    duration: float
    boundary_reason: str
    change_score: float
    dominant_signals: list[str]
    scene_tags: list[str]
    action_tags: list[str]
    object_counts: dict[str, int]
    related_segment_ids: list[str]
    related_keyframes: list[str]
    related_tracks: list[str]
    related_relations: list[str]
    summary: str
    provenance: dict[str, Any] = field(default_factory=dict)
    video_id: str | None = None
    source_vid: str | None = None
    video_path: str | None = None


@dataclass
class EvidenceNode:
    id: str
    type: str
    view_id: str
    text: str
    provenance: dict[str, Any]


@dataclass
class EvidenceRelation:
    source: str
    target: str
    relation: str
    provenance: dict[str, Any] = field(default_factory=dict)
    # Legacy links are explicit structural references, not model probabilities.
    confidence: float = 1.0
    constraints: dict[str, bool] = field(default_factory=dict)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


@dataclass
class EvidenceRecord:
    """Lossless evidence contract; retrieval scores are not observation confidence."""

    evidence_id: str = ""
    evidence_type: str = ""
    source_type: str = "rule"
    video_id: str | None = None
    segment_id: str | None = None
    timestamp: float | None = None
    time_range: list[float | None] = field(default_factory=list)
    subject_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    raw_text: str = ""
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    quality_status: str = "unknown"
    supporting_ids: list[str] = field(default_factory=list)


def normalize_evidence(value: dict[str, Any] | EvidenceRecord) -> dict[str, Any]:
    """Add the contract without removing legacy fields or the original source record."""
    item = asdict(value) if isinstance(value, EvidenceRecord) else dict(value)
    record = item.get("record") or {}
    provenance = item.get("provenance") or record.get("provenance") or {}
    def get(key: str, default: Any = None) -> Any:
        for source in (item, record, provenance):
            if source.get(key) is not None:
                return source[key]
        return default
    view = str(item.get("view") or item.get("evidence_type") or "unknown")
    item.setdefault("view", view)
    item.setdefault("id", item.get("evidence_id") or item.get("node_id") or "")
    attrs = get("attributes", {})
    attrs = dict(attrs) if isinstance(attrs, dict) else {"attribute_tags": attrs}
    for key in ("label", "color", "bbox", "frame_id", "track_id", "object_id", "event_type",
                "relation_type", "target_object_id", "related_object_id", "target_track_id",
                "related_track_id", "motion_summary", "compact_points"):
        val = get(key)
        if val is not None:
            attrs[key] = val
            item.setdefault(key, val)
    if view == "visual_object":
        attrs.setdefault("color", "unknown")
    caption, transcript = get("caption", ""), get("transcript", "")
    for key, val in (("caption", caption), ("transcript", transcript)):
        if val:
            item[key] = val
    raw = get("raw_text")
    if raw is None:
        raw = "\n".join(str(v) for v in (caption, transcript) if v) or str(
            record.get("evidence_text") or record.get("text") or record.get("summary")
            or record.get("motion_summary") or item.get("text") or "")
    source = get("source_type") or {"visual_object": "detector", "visual_track": "tracker",
        "scope": "caption" if caption else "transcript", "segment": "caption" if caption else "transcript",
        "caption_context": "caption" if caption else "transcript"}.get(view, "rule")
    if source not in {"caption", "transcript", "detector", "tracker", "event_model", "rule"}:
        source = "rule"
    support = list(get("supporting_ids", []) or [])
    for key in ("object_ids", "related_object_ids", "related_track_ids", "related_segment_ids", "segment_ids"):
        for entry in get(key, []) or []:
            if entry not in support:
                support.append(entry)
    for key in ("target_object_id", "related_object_id", "neighbor_object_id", "target_track_id", "related_track_id", "neighbor_track_id"):
        entry = get(key)
        if entry and entry not in support:
            support.append(entry)
    contract = EvidenceRecord(
        evidence_id=str(item.get("evidence_id") or item.get("id") or item.get("node_id") or ""),
        evidence_type=view, source_type=source, video_id=get("video_id", get("video_name")),
        segment_id=get("segment_id"), timestamp=get("timestamp"),
        time_range=get("time_range", [get("start_time"), get("end_time")]),
        subject_id=get("subject_id") or get("track_id") or get("object_id"),
        attributes=attrs, text=str(item.get("text") or item.get("short_text") or ""),
        raw_text=str(raw), confidence=get("confidence"), provenance=provenance,
        quality_status=get("quality_status", "unknown"), supporting_ids=support,
    )
    item.update(asdict(contract))
    return item
