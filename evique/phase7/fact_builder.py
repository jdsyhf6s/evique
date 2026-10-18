from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .normalization import normalize_source_lineage
from .schema import (
    CaptionStatus,
    EvidenceFact,
    ObservationSource,
    ScopeRecord,
    SourceFamily,
    TimeRange,
    TranscriptStatus,
)


def _identifier(record: dict[str, Any]) -> str:
    value = record.get("evidence_id") or record.get("id") or record.get("node_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("visual record requires evidence_id")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_visual_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("visual record must be an object")
    evidence_id = _identifier(record)
    video_id = record.get("video_id")
    provenance = record.get("provenance")
    if not isinstance(video_id, str) or not video_id or not isinstance(provenance, dict) or not provenance:
        raise ValueError("visual record requires video identity and provenance")
    if provenance.get("video_id") != video_id:
        raise ValueError("contradictory visual video identity")
    timestamp = _finite(record.get("timestamp"), "timestamp")
    if timestamp < 0:
        raise ValueError("timestamp must be nonnegative")
    normalized = dict(record)
    normalized.update({"evidence_id": evidence_id, "video_id": video_id, "timestamp": timestamp})
    bbox = record.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        box = tuple(_finite(value, "bbox coordinate") for value in bbox)
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError("bbox coordinates must be ordered")
        normalized["bbox"] = box
    return normalized


def _fact_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


_SOURCE_FAMILY_KINDS = {
    SourceFamily.DETECTOR: (ObservationSource.DETECTOR_OBSERVATION,),
    SourceFamily.SCOPE_SEMANTIC: (
        ObservationSource.OFFLINE_CAPTION,
        ObservationSource.TRANSCRIPT,
    ),
    SourceFamily.VISUAL_SEMANTIC: (ObservationSource.QUERY_CONDITIONED_OBSERVATION,),
    SourceFamily.TRACK_DERIVED: (ObservationSource.RULE_DERIVED_OBSERVATION,),
    SourceFamily.EVENT_DERIVED: (ObservationSource.RULE_DERIVED_OBSERVATION,),
}
_DEFAULT_FAMILY_BY_KIND = {
    ObservationSource.DETECTOR_OBSERVATION: SourceFamily.DETECTOR,
    ObservationSource.OFFLINE_CAPTION: SourceFamily.SCOPE_SEMANTIC,
    ObservationSource.TRANSCRIPT: SourceFamily.SCOPE_SEMANTIC,
    ObservationSource.QUERY_CONDITIONED_OBSERVATION: SourceFamily.VISUAL_SEMANTIC,
    ObservationSource.RULE_DERIVED_OBSERVATION: SourceFamily.EVENT_DERIVED,
}


def _declared_enum(record: dict[str, Any], key: str, enum):
    values = []
    for mapping in (record, record["provenance"]):
        if key in mapping:
            values.append(enum(mapping[key]))
    if values and any(value is not values[0] for value in values[1:]):
        raise ValueError(f"conflicting visual {key} declarations")
    return values[0] if values else None


def _default_source_family(fact_type: str) -> SourceFamily:
    kind = fact_type.casefold()
    if kind.startswith("track_") or kind in {"track", "visual_track", "motion"}:
        return SourceFamily.TRACK_DERIVED
    if kind.startswith("event_") or kind in {
        "event", "visual_event", "entry", "exit", "entry_exit", "state_change",
    }:
        return SourceFamily.EVENT_DERIVED
    return SourceFamily.DETECTOR


def _visual_source_metadata(
    record: dict[str, Any], fact_type: str,
) -> tuple[ObservationSource, SourceFamily]:
    kind = _declared_enum(record, "source_kind", ObservationSource)
    family = _declared_enum(record, "source_family", SourceFamily)
    default_family = _default_source_family(fact_type)
    derived_families = {SourceFamily.TRACK_DERIVED, SourceFamily.EVENT_DERIVED}

    if family is None:
        if default_family in derived_families:
            family = default_family
        elif kind is not None:
            family = _DEFAULT_FAMILY_BY_KIND[kind]
        else:
            family = default_family
    if default_family in derived_families and family is not default_family:
        raise ValueError("contradictory visual source family and fact type")
    if kind is None:
        kind = (_SOURCE_FAMILY_KINDS.get(family) or
                (ObservationSource.DETECTOR_OBSERVATION,))[0]
    if family is not SourceFamily.OTHER_INDEPENDENT and kind not in _SOURCE_FAMILY_KINDS[family]:
        raise ValueError("contradictory visual source kind and family")
    return kind, family


def _visual_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("visual ancestry must be a list or tuple")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in value):
        raise ValueError("visual ancestry must contain nonempty strings")
    return tuple(dict.fromkeys(value))


def _visual_lineage_metadata(
    record: dict[str, Any], evidence_id: str,
) -> tuple[tuple[str, ...], str | None]:
    mappings = (record, record["provenance"])
    declared = {
        key: [_visual_ids(mapping[key]) for mapping in mappings if key in mapping]
        for key in ("derived_from", "parent_evidence_ids")
    }

    def consistent(values: list[tuple[str, ...]]) -> tuple[str, ...]:
        nonempty = [value for value in values if value]
        if nonempty and any(set(value) != set(nonempty[0]) for value in nonempty[1:]):
            raise ValueError("conflicting visual ancestry declarations")
        return nonempty[0] if nonempty else ()

    derived_from = consistent(declared["derived_from"])
    parent_ids = consistent(declared["parent_evidence_ids"])
    if derived_from and parent_ids and set(derived_from) != set(parent_ids):
        raise ValueError("conflicting visual ancestry declarations")
    parents = derived_from or parent_ids
    if evidence_id in parents:
        raise ValueError("visual evidence cannot reference itself as a parent")

    rules = []
    for mapping in mappings:
        if "derivation_rule" not in mapping or mapping["derivation_rule"] is None:
            continue
        rule = mapping["derivation_rule"]
        if not isinstance(rule, str):
            raise TypeError("visual derivation rule must be a string or null")
        if rule.strip():
            rules.append(rule.strip())
    if rules and any(rule != rules[0] for rule in rules[1:]):
        raise ValueError("conflicting visual derivation_rule declarations")
    return parents, rules[0] if rules else None


def facts_from_scope(records: list[ScopeRecord]) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    for record in records:
        if not record.provenance or record.provenance.get("video_id") != record.video_id:
            raise ValueError("Scope fact requires consistent provenance")
        if record.caption_status is CaptionStatus.VALID:
            facts.append(EvidenceFact(
                fact_id=f"{record.scope_id}:caption", fact_type="scope_caption",
                video_id=record.video_id, time_span=record.time_range,
                frame_ids=record.source_frame_ids, value={"text": record.caption_text},
                source_evidence_ids=(record.scope_id,), source_kind=record.caption_source,
                provenance=dict(record.provenance),
            ))
        if record.transcript_status is TranscriptStatus.VALID:
            facts.append(EvidenceFact(
                fact_id=f"{record.scope_id}:transcript", fact_type="scope_transcript",
                video_id=record.video_id, time_span=record.time_range,
                frame_ids=record.source_frame_ids, value={"text": record.transcript_text},
                source_evidence_ids=(record.scope_id,), source_kind=ObservationSource.TRANSCRIPT,
                provenance=dict(record.provenance),
            ))
    return list(normalize_source_lineage(facts))


def facts_from_visual_records(records: list[dict]) -> list[EvidenceFact]:
    facts = []
    for source in records:
        record = normalize_visual_record(source)
        value = {
            key: record[key] for key in (
                "label", "color", "bbox", "frame_id", "camera_id", "object_id", "track_id", "event_type",
            ) if record.get(key) is not None
        }
        if source.get("detector_confidence") is not None:
            value["observation_confidence"] = _finite(source["detector_confidence"], "detector confidence")
        provenance = dict(record["provenance"])
        if source.get("score") is not None:
            provenance["retrieval_score"] = _finite(source["score"], "retrieval score")
        fact_type = str(source.get("fact_type") or source.get("view") or "visual_object")
        source_kind, source_family = _visual_source_metadata(record, fact_type)
        derived_from, derivation_rule = _visual_lineage_metadata(record, record["evidence_id"])
        facts.append(EvidenceFact(
            fact_id=_fact_id("observation", record["video_id"], record["evidence_id"]),
            fact_type=fact_type, video_id=record["video_id"],
            time_span=TimeRange(record["timestamp"], record["timestamp"]),
            frame_ids=(str(record["frame_id"]),) if record.get("frame_id") else (),
            role_bindings={"observed": str(record.get("object_id") or record["evidence_id"])},
            value=value, source_evidence_ids=(record["evidence_id"],),
            source_kind=source_kind, source_family=source_family,
            derived_from=derived_from,
            derivation_rule=derivation_rule, provenance=provenance,
        ))
    return list(normalize_source_lineage(facts))
