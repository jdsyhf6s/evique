"""Real, evidence-only Phase 7.5 A/B/C experiment harness.

This module only adapts frozen artifacts to already implemented algorithms.
It never generates answers and never instantiates a network client.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from evique.index_quality import FALLBACK_CAPTION_MARKER
from evique.phase7.schema import (
    EvidenceFact, ObservationSource, Optionality, ParseStatus, QuestionRequirementSet,
    RequirementType, SourceFamily, TimeRange, to_dict,
)


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _frozen_ids(value: Any) -> tuple[str, ...]:
    """Read explicit record references without treating text as lineage."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("frozen source IDs must be a list or tuple")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in value):
        raise ValueError("frozen source IDs must be nonempty strings")
    return tuple(dict.fromkeys(value))


def _frozen_source_metadata(mappings, default_kind, default_family):
    """Reconcile explicit source declarations before deciding independence."""
    def declared(key, enum=None):
        values = []
        for mapping in mappings:
            if key not in mapping:
                continue
            value = mapping[key]
            if enum is not None:
                value = enum(value)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("semantic source ID must be a nonempty string")
            values.append(value)
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"conflicting frozen {key} declarations")
        return values[0] if values else None

    kind = declared("source_kind", ObservationSource)
    family = declared("source_family", SourceFamily)
    semantic_source_id = declared("semantic_source_id")
    compatible = {
        SourceFamily.DETECTOR: (ObservationSource.DETECTOR_OBSERVATION,),
        SourceFamily.SCOPE_SEMANTIC: (ObservationSource.OFFLINE_CAPTION, ObservationSource.TRANSCRIPT),
        SourceFamily.VISUAL_SEMANTIC: (ObservationSource.QUERY_CONDITIONED_OBSERVATION,),
        SourceFamily.TRACK_DERIVED: (ObservationSource.RULE_DERIVED_OBSERVATION,),
        SourceFamily.EVENT_DERIVED: (ObservationSource.RULE_DERIVED_OBSERVATION,),
    }
    derived_families = {SourceFamily.TRACK_DERIVED, SourceFamily.EVENT_DERIVED}
    if family is None:
        if kind is ObservationSource.RULE_DERIVED_OBSERVATION:
            family = default_family if default_family in derived_families else SourceFamily.EVENT_DERIVED
        elif kind is not None:
            family = next(family for family, kinds in compatible.items() if kind in kinds)
        elif semantic_source_id is not None and default_family not in derived_families:
            family = SourceFamily.VISUAL_SEMANTIC
        else:
            family = default_family
    if family not in compatible:
        raise ValueError("unsupported frozen source family")
    if kind is None:
        kind = default_kind if default_kind in compatible[family] else compatible[family][0]
    if kind not in compatible[family]:
        raise ValueError("contradictory frozen source kind and family")
    if default_family in derived_families and family is not default_family:
        raise ValueError("derived frozen view has conflicting source family")
    if semantic_source_id is not None and family is not SourceFamily.VISUAL_SEMANTIC:
        raise ValueError("semantic source requires an independent visual semantic record")
    if family is SourceFamily.VISUAL_SEMANTIC and semantic_source_id is None:
        raise ValueError("visual semantic record requires an explicit semantic source ID")
    return kind, family, semantic_source_id


def _frozen_lineage_metadata(mappings, evidence_id, *, derived, track_derived):
    """Read original declarations so empty outer metadata cannot erase ancestry."""
    declarations = {
        key: [_frozen_ids(mapping[key]) for mapping in mappings if key in mapping]
        for key in ("derived_from", "parent_evidence_ids", "source_evidence_ids")
    }

    def consistent(values, key):
        nonempty = [value for value in values if value]
        if nonempty and any(set(value) != set(nonempty[0]) for value in nonempty[1:]):
            raise ValueError(f"conflicting frozen {key} declarations")
        return nonempty[0] if nonempty else ()

    ancestry = {key: consistent(declarations[key], key)
                for key in ("derived_from", "parent_evidence_ids")}
    if (ancestry["derived_from"] and ancestry["parent_evidence_ids"]
            and set(ancestry["derived_from"]) != set(ancestry["parent_evidence_ids"])):
        raise ValueError("conflicting frozen parent declarations")
    if any(evidence_id in parents for parents in ancestry.values()):
        raise ValueError("frozen evidence cannot reference itself as a parent")
    rules = []
    for mapping in mappings:
        rule = mapping.get("derivation_rule")
        if rule is None:
            continue
        if not isinstance(rule, str):
            raise ValueError("frozen derivation rule must be a string or null")
        if rule.strip():
            rules.append(rule.strip())
    if rules and any(rule != rules[0] for rule in rules[1:]):
        raise ValueError("conflicting frozen derivation_rule declarations")
    rule = rules[0] if rules else None
    if not derived and (any(ancestry.values()) or rule):
        raise ValueError("observed frozen source has derived metadata")

    parents = ()
    if derived:
        parents = ancestry["derived_from"] or ancestry["parent_evidence_ids"]
        if not any(declarations[key] for key in ancestry):
            if declarations["source_evidence_ids"]:
                inputs = [tuple(identifier for identifier in value if identifier != evidence_id)
                          for value in declarations["source_evidence_ids"]]
                parents = consistent(inputs, "source_evidence_ids as parents")
            elif track_derived:
                parents = consistent([_frozen_ids(mapping["object_ids"]) for mapping in mappings
                                      if "object_ids" in mapping], "object_ids")
        parents = tuple(identifier for identifier in parents if identifier != evidence_id)
    citations = tuple(dict.fromkeys(identifier for values in declarations["source_evidence_ids"]
                                   for identifier in values))
    return parents, rule, citations


def _frozen_semantic_category(mappings):
    """Bind an independent semantic record to its declared category."""
    key = "semantic_label" if any("semantic_label" in mapping for mapping in mappings) else "label"
    labels = [mapping[key] for mapping in mappings if key in mapping]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError(f"frozen {key} must be a nonempty string")
    if labels and any(label.strip().casefold() != labels[0].strip().casefold() for label in labels[1:]):
        raise ValueError(f"conflicting frozen {key} declarations")
    return key, labels[0] if labels else None


def fact_from_visible_item(item: dict[str, Any]) -> EvidenceFact:
    """Map one already retrieved record into the frozen Phase 7 fact schema."""
    if not isinstance(item, dict):
        raise TypeError("visible evidence item must be an object")
    record = item.get("record") if isinstance(item.get("record"), dict) else {}
    row = {**record, **{key: value for key, value in item.items() if key != "record"}}
    provenance = dict(record.get("provenance") or {})
    provenance.update(item.get("provenance") or {})
    video_id = str(_first(row, "video_id", "source_vid", "video_name") or "")
    provenance_video = str(_first(provenance, "video_id", "source_vid", "video_name") or video_id)
    if not video_id or provenance_video != video_id:
        raise ValueError("visible fact has conflicting or missing video identity")
    provenance["video_id"] = video_id
    evidence_id = str(_first(row, "evidence_id", "id", "node_id", "object_id", "track_id", "event_id", "relation_id") or "")
    if not evidence_id:
        raise ValueError("visible fact requires evidence identity")
    start = _finite(_first(row, "start_time", "timestamp"), 0.0)
    end = _finite(_first(row, "end_time", "timestamp"), start)
    if end < start:
        start, end = end, start
    view = str(_first(row, "view", "source_view") or "scope")
    text = str(_first(row, "caption", "transcript", "summary", "evidence_text", "text", "motion_summary") or "").strip()
    if FALLBACK_CAPTION_MARKER in text.casefold():
        raise ValueError("placeholder caption cannot become a Phase 7.5 fact")
    frame_ids = row.get("source_frame_ids") or ([] if row.get("frame_id") is None else [row["frame_id"]])
    frame_ids = tuple(str(value) for value in frame_ids if value is not None)
    role_bindings: dict[str, str] = {}
    value: dict[str, Any] = {"text": text}
    derivation_rule = None
    source_kind = ObservationSource.DETECTOR_OBSERVATION
    source_family = SourceFamily.DETECTOR

    if view == "scope":
        fact_type = "scope_transcript" if row.get("transcript") and not row.get("caption") else "scope_caption"
        source_kind = ObservationSource.TRANSCRIPT if fact_type == "scope_transcript" else ObservationSource.OFFLINE_CAPTION
        source_family = SourceFamily.SCOPE_SEMANTIC
    elif view in {"visual_object", "target", "object"}:
        fact_type = "visual_object"
        value.update({key: row[key] for key in ("label", "color", "bbox") if row.get(key) is not None})
        subject = _first(row, "object_id", "target_object_id", "id")
        if subject:
            role_bindings["observed"] = str(subject)
            role_bindings["subject"] = str(subject)
    elif view == "visual_relation":
        fact_type = str(row.get("relation_type") or "interaction")
        subject = _first(row, "target_object_id", "object_id", "target_track_id")
        reference = _first(row, "related_object_id", "neighbor_object_id", "related_track_id", "neighbor_track_id")
        if subject:
            role_bindings["subject"] = str(subject)
        if reference:
            role_bindings["reference"] = str(reference)
        value.update({"relation_type": fact_type})
        source_kind = ObservationSource.RULE_DERIVED_OBSERVATION
        source_family = SourceFamily.EVENT_DERIVED
        derivation_rule = "frozen_visual_relation_record"
    elif view in {"visual_track", "track"}:
        fact_type = "track_motion"
        motion = str(_first(row, "motion", "motion_class", "motion_summary") or "observed_trajectory")
        value.update({"motion": motion, "track_id": _first(row, "track_id", "id")})
        subject = _first(row, "track_id", "id")
        if subject:
            role_bindings["subject"] = str(subject)
        source_kind = ObservationSource.RULE_DERIVED_OBSERVATION
        source_family = SourceFamily.TRACK_DERIVED
        derivation_rule = "frozen_visual_track_record"
    elif view in {"event", "adaptive_event", "visual_event", "fixed_window_event"}:
        raw_type = str(_first(row, "event_type", "type") or "event").casefold()
        fact_type = raw_type if raw_type in {"entry", "exit", "entry_exit"} else "event"
        value.update({"event_type": raw_type})
        subject = _first(row, "track_id", "target_track_id", "object_id")
        if subject:
            role_bindings["subject"] = str(subject)
        source_kind = ObservationSource.RULE_DERIVED_OBSERVATION
        source_family = SourceFamily.EVENT_DERIVED
        derivation_rule = "frozen_event_record"
    else:
        fact_type = "event" if "event" in view else "visual_object" if "object" in view else "scope_caption"

    # Raw tracker identity is target metadata, never a semantic source citation.
    raw_track_id = (_first(row, "track_id", "target_track_id", "raw_track_id")
                    or _first(provenance, "raw_track_id", "track_id"))
    if raw_track_id is None and view in {"visual_track", "track"}:
        raw_track_id = _first(row, "id")
    if raw_track_id is not None:
        value["track_id"] = str(raw_track_id)
        provenance["raw_track_id"] = str(raw_track_id)
    value.update({key: row[key] for key in ("label", "semantic_label", "semantic_source_id", "local_target_id")
                  if row.get(key) is not None})
    original_mappings = (item, record, dict(item.get("provenance") or {}), dict(record.get("provenance") or {}))
    source_kind, source_family, semantic_source_id = _frozen_source_metadata(
        original_mappings, source_kind, source_family,
    )
    if semantic_source_id is not None:
        value["semantic_source_id"] = semantic_source_id
        provenance["semantic_source_id"] = semantic_source_id
        category_key, category = _frozen_semantic_category(original_mappings)
        if category is not None:
            if category_key == "semantic_label":
                if "label" in row:
                    provenance["frozen_label"] = row["label"]
                value["semantic_label"] = category
            # Identity consumers read label; a detector label cannot borrow
            # the semantic record's independence when semantic_label exists.
            value["label"] = category

    derived = source_family in {SourceFamily.TRACK_DERIVED, SourceFamily.EVENT_DERIVED}
    parents, explicit_rule, citations = _frozen_lineage_metadata(
        original_mappings, evidence_id, derived=derived,
        track_derived=source_family is SourceFamily.TRACK_DERIVED,
    )
    if derived:
        derivation_rule = explicit_rule or derivation_rule
        provenance["frozen_evidence_id"] = evidence_id
        if any("source_evidence_ids" in mapping for mapping in original_mappings):
            provenance["frozen_source_evidence_ids"] = list(citations)
    # The output alias and immediate parents are the only resolver citations.
    # Transitive citations would alias ancestors to their own descendants.
    sources = tuple(dict.fromkeys((evidence_id, *parents))) if derived else (evidence_id,)
    roots = () if derived else (evidence_id,)
    if not derived and source_family is SourceFamily.VISUAL_SEMANTIC and semantic_source_id:
        # The wrapper and semantic source describe one observation, not two votes.
        sources = roots = (semantic_source_id,)
        provenance["frozen_evidence_id"] = evidence_id

    return EvidenceFact(
        fact_id="phase7.5:" + hashlib.sha256(f"{video_id}|{view}|{evidence_id}".encode()).hexdigest()[:20],
        fact_type=fact_type,
        video_id=video_id,
        time_span=TimeRange(start, end),
        frame_ids=frame_ids,
        role_bindings=role_bindings,
        value=value,
        source_evidence_ids=sources,
        source_kind=source_kind,
        source_family=source_family,
        derived_from=parents,
        independence_root_ids=roots,
        derivation_rule=derivation_rule,
        provenance=provenance,
    )


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _semantic_key(fact: EvidenceFact) -> str:
    value = [
        fact.fact_type, fact.video_id, fact.role_bindings, fact.value,
        round(fact.time_span.start, 1), round(fact.time_span.end, 1),
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _requirement_support(requirements: QuestionRequirementSet,
                         facts: list[EvidenceFact]) -> tuple[list[str], list[list[str]]]:
    from evique.phase7.bundle_builder import _supports

    supported: list[str] = []
    pairs: list[list[str]] = []
    for requirement in requirements.requirements:
        matching = [fact for fact in facts if _supports(requirement, fact)]
        if matching:
            supported.append(requirement.requirement_id)
            pairs.extend([[requirement.requirement_id, fact.fact_id] for fact in matching])
    return supported, pairs


def _fact_payload(fact: EvidenceFact) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id, "fact_type": fact.fact_type,
        "video_id": fact.video_id,
        "time_span": {"start": fact.time_span.start, "end": fact.time_span.end},
        "frame_ids": list(fact.frame_ids), "role_bindings": dict(fact.role_bindings),
        "value": fact.value, "source_evidence_ids": list(fact.source_evidence_ids),
        "source_kind": fact.source_kind.value,
        "source_family": fact.source_family.value if fact.source_family else None,
        "derived_from": list(fact.derived_from),
        "independence_root_ids": list(fact.independence_root_ids),
        "episode_id": fact.episode_id,
        "derivation_rule": fact.derivation_rule,
        "provenance": dict(fact.provenance),
    }


class CandidateFactProvider:
    """Adapter over the frozen Phase 2 multi-channel retrieval contract."""

    def __init__(self, retriever: Any, query: str):
        self.retriever = retriever
        self.query = query
        self.last_trace: dict[str, Any] = {}
        self.last_facts: list[EvidenceFact] = []

    def _retain_spatial_references(self, fused, requirements, video_id, limit):
        """Reserve up to two slots for missing same-frame endpoint observations.

        This is bounded candidate retrieval, not a relation inference. Overlapping
        detections cannot count as distinct objects. No caption supplies an endpoint.
        """
        def box(row):
            value = row.get("bbox")
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                return None
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
                return None
            return value if value[2] > value[0] and value[3] > value[1] else None

        def duplicate(a, b):
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
            return intersection / union >= .5

        def matches(row, role):
            return (str(row.get("label") or "").casefold() == (role.category or "").casefold()
                    and all(m.casefold() in {str(row.get("color") or "").casefold(),
                                             *(str(a).casefold() for a in row.get("attributes", ()))}
                            for m in role.modifiers))

        observations = getattr(self.retriever, "visual_objects", ())
        if limit < 3 or not observations:
            return fused, []
        selected = [(item, item["evidence"].get("record") or item["evidence"])
                    for item in fused if item["evidence"].get("view") == "visual_object"]
        for requirement in requirements.requirements:
            if (requirement.requirement_type is not RequirementType.SPATIAL_RELATION
                    or requirement.subject is None or requirement.reference_object is None):
                continue
            if any(matches(row, requirement.reference_object) for _, row in selected):
                continue
            for anchor, row in selected:
                anchor_box = box(row)
                if not matches(row, requirement.subject) or anchor_box is None or not row.get("frame_id"):
                    continue
                if row.get("video_id") != video_id or not isinstance(row.get("timestamp"), (int, float)):
                    continue
                neighbors = [other for other in observations
                             if other.get("video_id") == video_id
                             and other.get("frame_id") == row["frame_id"]
                             and other.get("timestamp") == row["timestamp"]
                             and other.get("object_id") != row.get("object_id")
                             and matches(other, requirement.reference_object)
                             and box(other) is not None and not duplicate(anchor_box, box(other))]
                center = lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2)
                neighbors.sort(key=lambda other: (math.dist(center(anchor_box), center(box(other))), str(other.get("object_id"))))
                kept = []
                for other in neighbors:
                    if other.get("object_id") and not any(duplicate(box(other), box(previous)) for previous in kept):
                        kept.append(other)
                    if len(kept) == 2:
                        break
                if not kept:
                    continue
                retained = list(fused)
                dropped = []
                while len(retained) + len(kept) > limit:
                    index = next(i for i in range(len(retained)-1, -1, -1) if retained[i] is not anchor)
                    dropped.append(retained.pop(index)["id"])
                added = [{"id": other["object_id"], "video_id": video_id,
                          "evidence": self.retriever._visual_evidence_item("visual_object", other, 0.0)}
                         for other in kept]
                return retained + added, [{"anchor_id": anchor["id"], "frame_id": row["frame_id"],
                                            "timestamp": row["timestamp"], "added_ids": [a["id"] for a in added],
                                            "displaced_ids": dropped, "reason": "missing_same_frame_reference_candidates"}]
        return fused, []

    def get_facts(self, requirements: QuestionRequirementSet, video_id: str,
                  shared_index: Any) -> list[EvidenceFact]:
        from evique.candidate_fusion import reciprocal_rank_fusion

        limit = int(self.retriever.max_evidence)
        candidates = self.retriever.retrieve_candidates(
            self.query, {"video_id": video_id}, top_k=limit,
        )
        channels = {
            source: [candidate for candidate in candidates if candidate.source == source]
            for source in ("keyword", "embedding", "visual", "event")
        }
        fused = reciprocal_rank_fusion(channels, limit=limit, video_ids={video_id})
        fused, reference_retrieval = self._retain_spatial_references(fused, requirements, video_id, limit)
        facts: list[EvidenceFact] = []
        rejected: list[dict[str, str]] = []
        seen: set[str] = set()
        for candidate in fused:
            if candidate.get("video_id") != video_id:
                rejected.append({"id": str(candidate.get("id")), "reason": "wrong_video"})
                continue
            try:
                fact = fact_from_visible_item(candidate["evidence"])
            except (TypeError, ValueError) as exc:
                rejected.append({"id": str(candidate.get("id")), "reason": type(exc).__name__})
                continue
            if fact.fact_id in seen:
                continue
            seen.add(fact.fact_id)
            facts.append(fact)
        self.last_facts = facts
        self.last_trace = {
            "channel_counts": {key: len(value) for key, value in channels.items()},
            "fused_candidate_count": len(fused),
            "typed_fact_count": len(facts),
            "rejected": rejected,
            "fused_ids": [str(item.get("id")) for item in fused],
            "same_frame_reference_retrieval": reference_retrieval,
        }
        return facts


def build_phase7_fact_provider(
    retriever: Any,
    query: str,
    *,
    frozen_index_dir: Path,
    phase7_config: Any,
) -> Any:
    """Select the original c73 provider unless the explicit Phase D gate is on."""
    base = CandidateFactProvider(retriever, query)
    bridge = phase7_config.temporal_relation_bridge
    if not bridge.enabled:
        return base
    from evique.phase7.local_completion import FrozenIndexCompletionSource
    from evique.phase7.temporal_relation_bridge import TemporalRelationFactProvider

    index_dir = Path(frozen_index_dir).resolve()
    return TemporalRelationFactProvider(
        base,
        query=query,
        source_factory=lambda requirements: FrozenIndexCompletionSource(index_dir, requirements),
        config=bridge,
        candidate_limit=int(retriever.max_evidence),
    )


def _status_text(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, dict):
        return _status_text(value.get("status"))
    return str(value or "NOT_ASSESSED")


def _support_block(requirements: QuestionRequirementSet, facts: list[EvidenceFact], *,
                   claimed_status: str) -> tuple[dict[str, Any], list[list[str]]]:
    supported, pairs = _requirement_support(requirements, facts)
    mandatory = [
        item.requirement_id for item in requirements.requirements
        if item.optionality is Optionality.MANDATORY
    ]
    missing = [identifier for identifier in mandatory if identifier not in supported]
    relation_requirements = {
        RequirementType.SPATIAL_RELATION, RequirementType.INTERACTION,
    }
    temporal_requirements = {
        RequirementType.BEFORE_STATE, RequirementType.AFTER_STATE,
        RequirementType.STATE_CHANGE, RequirementType.ENTRY_EXIT,
    }
    relation_needed = any(item.requirement_type in relation_requirements for item in requirements.requirements)
    temporal_needed = any(item.requirement_type in temporal_requirements for item in requirements.requirements)
    relation_facts = [fact for fact in facts if fact.fact_type in {
        "left_of", "right_of", "above", "below", "overlap", "image_near", "interaction",
    }]
    temporal_facts = [fact for fact in facts if fact.fact_type in {
        "before_state", "after_state", "scene_state_before", "scene_state_after", "state_change",
        "entry", "exit", "entry_exit",
    }]
    role_known = [
        bool((fact.role_bindings.get("subject") or fact.role_bindings.get("actor"))
             and (fact.role_bindings.get("reference") or fact.role_bindings.get("patient")))
        for fact in relation_facts
    ]
    temporal_consistent: bool | None
    if not temporal_needed:
        temporal_consistent = True
    elif temporal_facts:
        temporal_consistent = all(fact.derivation_rule or fact.fact_type in {"entry", "exit", "entry_exit"}
                                  for fact in temporal_facts)
    else:
        temporal_consistent = False
    role_consistent = (all(role_known) if role_known else not relation_needed)
    provenance_complete = all(
        fact.provenance.get("video_id") == fact.video_id and fact.source_evidence_ids
        for fact in facts
    )
    strict_complete = (
        requirements.parse_status is ParseStatus.COMPLETE and not missing
        and role_consistent and temporal_consistent and provenance_complete
    )
    return ({
        "supported_requirement_ids": supported,
        "missing_requirement_ids": missing,
        "compatibility_status": claimed_status,
        "complete": claimed_status == "COMPLETE",
        "parse_complete": requirements.parse_status is ParseStatus.COMPLETE,
        "role_consistent": role_consistent,
        "temporal_consistent": temporal_consistent,
        "provenance_complete": provenance_complete,
        "strict_complete": strict_complete,
        "role_checks": [{"consistent": value} for value in role_known],
        "temporal_checks": ([{"consistent": temporal_consistent}] if temporal_needed else []),
    }, pairs)


def _metric_row(query_id: str, category: str, arm: str, requirements: QuestionRequirementSet,
                facts: list[EvidenceFact], support: dict[str, Any], pairs: list[list[str]],
                *, visible_tokens: int, evidence_item_count: int,
                completion_added_fact_count: int, latency_seconds: float) -> dict[str, Any]:
    mandatory = [item for item in requirements.requirements if item.optionality is Optionality.MANDATORY]
    semantic = [_semantic_key(fact) for fact in facts]
    duplicate = len(semantic) - len(set(semantic))
    role_checks = support.get("role_checks") or []
    temporal_checks = support.get("temporal_checks") or []
    claimed_complete = support.get("complete") is True or support.get("compatibility_status") == "COMPLETE"
    row = {
        "query_id": query_id, "category": category, "arm": arm,
        "question_requirement_coverage": (
            len(support["supported_requirement_ids"]) / len(mandatory) if mandatory else None
        ),
        "semantic_density": len(pairs) * 1000 / visible_tokens if visible_tokens else None,
        "duplicate_evidence_ratio": duplicate / len(semantic) if semantic else None,
        "role_bound_consistency": (
            sum(item.get("consistent") is True for item in role_checks) / len(role_checks)
            if role_checks else None
        ),
        "temporal_consistency": (
            sum(item.get("consistent") is True for item in temporal_checks) / len(temporal_checks)
            if temporal_checks else None
        ),
        "complete_false_positive": int(claimed_complete and not support.get("strict_complete")),
        "complete_claimed": int(claimed_complete),
        "visible_evidence_tokens": visible_tokens,
        "evidence_item_count": evidence_item_count,
        "completion_added_fact_count": completion_added_fact_count,
        "latency_seconds": latency_seconds,
    }
    if "safety_audit" in support:
        row.update({key: support[key] for key in (
            "full_support_count", "conditional_support_count", "no_support_count", "safety_audit",
        )})
    return row


def run_real_abc(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run exactly Warsaw-12 A/B/C evidence construction and stop."""
    from evique.embedding_index import FastEmbedEncoder
    from evique.experiments.protocol import shared_messages
    from evique.phase4_trace import finalize_phase4_package
    from evique.phase7.captionrich_baseline import CaptionRichHybrid
    from evique.phase7.config import Phase7Config
    from evique.phase7.pipeline import Phase7Pipeline
    from evique.phase7.requirement_parser import parse_requirements
    from evique.phase7.scope_index import SharedPhase7Index
    from evique.retriever import EvidenceRetriever

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if config.get("mode") != "evidence_only" or config.get("context_limit_chars") != 12000:
        raise ValueError("Phase 7.5 permits only evidence_only at 12000 characters")
    queries_doc = _json(Path(config["query_file"]))
    queries = queries_doc.get("queries")
    if not isinstance(queries, list) or len(queries) != 12:
        raise ValueError("Phase 7.5 requires exactly Warsaw-12")
    if _sha256(Path(config["query_file"])) != config["query_file_sha256"]:
        raise ValueError("Warsaw-12 query hash mismatch")
    selection = _json(Path(config["selection_manifest"]))
    categories = {row["query_id"]: row["category"] for row in selection["selected_queries"]}
    if [row["query_id"] for row in queries] != [row["query_id"] for row in selection["selected_queries"]]:
        raise ValueError("query order differs from frozen selection")
    preflight = _json(Path(config["execution_lock"]))
    if _sha256(Path(config["execution_lock"])) != config["execution_lock_sha256"]:
        raise ValueError("Phase 7.5 execution lock hash mismatch")
    if preflight.get("actual_head") != config.get("phase7_head"):
        raise ValueError("Phase 7 branch HEAD mismatch")

    index_root = Path(config["index_root"])
    encoder = FastEmbedEncoder(
        model_id=config["embedding_model_id"], cache_dir=config["embedding_cache_dir"],
        local_files_only=True, threads=2,
    )
    shared = SharedPhase7Index.load(index_root / "embeddings", encoder, index_root / "index")
    if shared.manifest["scope_sha256"] != config["scope_sha256"]:
        raise ValueError("shared Scope hash mismatch")
    if shared.manifest["embedding_manifest_sha256"] != config["embedding_manifest_sha256"]:
        raise ValueError("embedding manifest hash mismatch")
    retriever = EvidenceRetriever(
        index_root / "index", max_evidence=18, token_budget=12000,
        embedding_index=shared.dense_index, hybrid_enabled=True, construction_enabled=True,
    )
    baseline = CaptionRichHybrid(shared)
    phase7_config = Phase7Config.from_dict(_json(Path(config["phase7_config"])))
    token_counter = lambda messages: math.ceil(sum(len(item["content"]) for item in messages) / 4)

    output.mkdir(parents=True)
    trace_dir = output / "per_query_trace"
    trace_dir.mkdir()
    metric_rows: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}
    for position, query in enumerate(queries, 1):
        query_id, question = query["query_id"], query["query"]
        category = categories[query_id]
        video_id = "4"
        requirements = parse_requirements(question)

        began = time.perf_counter()
        a_package = baseline.retrieve(question, video_id, context_limit_chars=12000)
        a_visible = []
        for item in a_package.visible_items:
            row = {
                "id": item.evidence_id, "view": "scope", "video_id": item.video_id,
                "start_time": item.time_range.start, "end_time": item.time_range.end,
                "source_frame_ids": list(item.source_frame_ids), "provenance": item.provenance,
            }
            row["transcript" if item.source_kind == "transcript" else "caption"] = item.text
            a_visible.append(row)
        a_facts = [fact_from_visible_item(item) for item in a_visible]
        a_support, a_pairs = _support_block(requirements, a_facts, claimed_status="NOT_ASSESSED")
        a_latency = time.perf_counter() - began
        a_tokens = math.ceil(len(a_package.visible_context) / 4)
        a_trace = {
            "schema_version": "phase7_5_real_trace_v1", "query_id": query_id,
            "question": question, "video_id": video_id, "category": category,
            "arm": "A CaptionRich-Hybrid", "requirements": to_dict(requirements),
            "visible_evidence": a_visible, "visible_facts": [_fact_payload(fact) for fact in a_facts],
            "support": a_support, "requirement_fact_pairs": a_pairs,
            "budget": {"context_limit_chars": 12000, "visible_tokens_estimated": a_tokens,
                       "evidence_items_visible": len(a_visible), "truncated": a_package.truncated},
            "latency_seconds": a_latency, "model_calls": {"caption": 0, "llm": 0, "judge": 0, "embedding_query": 1},
        }

        began = time.perf_counter()
        b_raw = retriever.retrieve(question, {"video_id": video_id, "query_id": query_id})
        b_package = finalize_phase4_package(b_raw)
        b_items = list(b_package.get("evidence") or []) + list(b_package.get("temporal_sequence_evidence") or [])
        unique: dict[tuple[Any, ...], dict] = {}
        for item in b_items:
            unique.setdefault((item.get("video_id"), item.get("view"), item.get("evidence_id") or item.get("id")), item)
        _messages, b_visible, b_prompt = shared_messages(question, list(unique.values()), max_context_chars=12000)
        b_facts = []
        b_rejected = []
        for item in b_visible:
            try:
                b_facts.append(fact_from_visible_item(item))
            except (TypeError, ValueError) as exc:
                b_rejected.append({"id": item.get("evidence_id"), "reason": type(exc).__name__})
        b_claimed = _status_text(b_package.get("evidence_status") or b_raw.get("evidence_status"))
        b_support, b_pairs = _support_block(requirements, b_facts, claimed_status=b_claimed)
        b_latency = time.perf_counter() - began
        b_tokens = math.ceil(b_prompt["context_chars"] / 4)
        b_trace = {
            "schema_version": "phase7_5_real_trace_v1", "query_id": query_id,
            "question": question, "video_id": video_id, "category": category,
            "arm": "B Repaired Full EVIQUE", "requirements": to_dict(requirements),
            "visible_evidence": b_visible, "visible_facts": [_fact_payload(fact) for fact in b_facts],
            "support": b_support, "requirement_fact_pairs": b_pairs,
            "adapter_rejections": b_rejected,
            "budget": {"context_limit_chars": 12000, "visible_tokens_estimated": b_tokens,
                       "evidence_items_visible": len(b_visible), "truncated": bool(b_prompt["dropped_ids"])},
            "latency_seconds": b_latency, "model_calls": {"caption": 0, "llm": 0, "judge": 0, "embedding_query": 1},
        }

        began = time.perf_counter()
        provider = build_phase7_fact_provider(
            retriever,
            question,
            frozen_index_dir=index_root / "index",
            phase7_config=phase7_config,
        )
        pipeline = Phase7Pipeline(
            shared, provider, config=phase7_config, token_counter=token_counter,
            completion_sources={}, runtime_metadata={"experiment": "phase7.5-real-evidence-only"},
        )
        c_package = pipeline.run(question, video_id)
        visible_ids = set(c_package.visible_context.visible_bundle_ids)
        visible_bundles = [bundle for bundle in c_package.bundles if bundle.bundle_id in visible_ids]
        c_visible_facts = []
        for bundle in visible_bundles:
            c_visible_facts.extend(bundle.provenance.get("visible_facts") or [])
        by_id = {fact.fact_id: fact for fact in provider.last_facts}
        c_facts = [by_id[item["fact_id"]] for item in c_visible_facts if item.get("fact_id") in by_id]
        assessment = c_package.visible_context.assessment
        c_claimed = assessment.compatibility_status
        c_support, c_pairs = _support_block(requirements, c_facts, claimed_status=c_claimed)
        c_support.update({
            **c_package.visible_context.trace["support_counts"],
            "safety_audit": c_package.visible_context.trace["safety_audit"],
            "supported_requirement_ids": list(assessment.supported_requirement_ids),
            "missing_requirement_ids": list(assessment.missing_requirement_ids),
            "compatibility_status": assessment.compatibility_status,
            "complete": assessment.complete,
            "parse_complete": assessment.parse_complete,
            "role_consistent": assessment.role_consistent,
            "temporal_consistent": assessment.temporal_consistent,
            "provenance_complete": assessment.provenance_complete,
            "strict_complete": assessment.complete,
        })
        admitted = sum(
            len(entry.get("admitted_new_fact_ids") or [])
            for entry in c_package.completion.trace.get("fetches", [])
        )
        c_latency = time.perf_counter() - began
        c_tokens = c_package.visible_context.budget_status.actual_message_tokens
        c_trace = {
            "schema_version": "phase7_5_real_trace_v1", "query_id": query_id,
            "question": question, "video_id": video_id, "category": category,
            "arm": "C Phase7", "requirements": to_dict(requirements),
            "initial_candidates": provider.last_trace,
            "visible_evidence": c_visible_facts,
            "visible_facts": [_fact_payload(fact) for fact in c_facts],
            "visible_bundles": [to_dict(bundle) for bundle in visible_bundles],
            "support": c_support, "requirement_fact_pairs": c_pairs,
            "consistency": {
                "role_checks": [check for bundle in visible_bundles for check in (bundle.consistency.get("role_checks") or {}).values()],
                "temporal_checks": [{"consistent": bundle.consistency.get("temporal_consistency")} for bundle in visible_bundles if "temporal_consistency" in bundle.consistency],
            },
            "completion": c_package.completion.trace,
            "budget": {"context_limit_chars": 12000, "visible_tokens_estimated": c_tokens,
                       "evidence_items_visible": len(c_visible_facts),
                       "bundle_count": len(visible_bundles),
                       "truncated": bool(c_package.visible_context.omitted_bundle_ids)},
            "latency_seconds": c_latency, "model_calls": {"caption": 0, "llm": 0, "judge": 0, "embedding_query": 1},
        }

        traces = ((a_trace, a_facts, a_support, a_pairs, a_tokens, len(a_visible), 0, a_latency),
                  (b_trace, b_facts, b_support, b_pairs, b_tokens, len(b_visible), 0, b_latency),
                  (c_trace, c_facts, c_support, c_pairs, c_tokens, len(c_visible_facts), admitted, c_latency))
        for trace, facts, support, pairs, tokens, count, completion_count, latency in traces:
            name = f"{query_id}__{trace['arm'][0].lower()}.json"
            path = trace_dir / name
            _write_json(path, trace)
            trace_hashes[path.relative_to(output).as_posix()] = _sha256(path)
            metric_rows.append(_metric_row(
                query_id, category, trace["arm"], requirements, facts, support, pairs,
                visible_tokens=tokens, evidence_item_count=count,
                completion_added_fact_count=completion_count, latency_seconds=latency,
            ))
        print(json.dumps({"query": position, "query_id": query_id, "total": 12}), flush=True)

    metrics_path = output / "per_query_metrics.json"
    _write_json(metrics_path, metric_rows)
    trace_hashes[metrics_path.relative_to(output).as_posix()] = _sha256(metrics_path)
    summary = {
        "schema_version": "phase7_5_real_abc_summary_v1",
        "status": "evidence_only_complete", "query_count": 12, "trace_count": 36,
        "hard_stop": "warsaw12", "answers_generated": 0, "judge_calls": 0,
        "external_requests": 0, "raw_frame_requests": 0,
        "scope_sha256": shared.manifest["scope_sha256"],
        "embedding_manifest_sha256": shared.manifest["embedding_manifest_sha256"],
        "trace_hashes": trace_hashes,
        "support_counts": {
            key: sum(row[key] for row in metric_rows if "safety_audit" in row)
            for key in ("full_support_count", "conditional_support_count", "no_support_count")
        },
        "safety_audit": {
            key: sum(row["safety_audit"][key] for row in metric_rows if "safety_audit" in row)
            for key in ("IDENTITY_OVER_SUPPORT", "TYPING_MISS",
                        "CROSS_EPISODE_IDENTITY_MERGE", "UNSUPPORTED_STATE_CHANGE")
        },
    }
    _write_json(output / "run_summary.json", summary)
    return summary
