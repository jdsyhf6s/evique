"""Bounded typed completion over an immutable local Phase 7 index.

The source never performs generic retrieval.  It reconstructs only relations and
ordered scene states whose raw detector/event records are already present in the
frozen index and whose identity is anchored by the current evidence bundle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .normalization import DEFAULT_NORMALIZATION_POLICY, build_track_episodes, canonical_category
from .role_consistency import (
    IdentityStatus,
    assess_identity_categories,
    identity_check_payload,
)
from .scene_state import SceneStateRules, build_scene_states, compare_scene_states
from .temporal_refinement import RefinedWindow, TemporalDirection
from .schema import (
    EvidenceBundle,
    EvidenceFact,
    ObservationSource,
    QuestionRequirement,
    QuestionRequirementSet,
    RequirementType,
    RoleReference,
    SceneState,
    TimeRange,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{line_number} is not an object")
            rows.append(row)
    return rows


def _id(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"record lacks identity: {names}")


def _number(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _matches(row: dict[str, Any], role: RoleReference | None) -> bool:
    if role is None:
        return True
    if role.category and str(row.get("label") or "").casefold() != role.category.casefold():
        return False
    available = {str(row.get("color") or "").casefold()}
    available.update(str(value).casefold() for value in (row.get("attributes") or ()))
    return all(modifier.casefold() in available for modifier in role.modifiers)


def _center(row: dict[str, Any]) -> tuple[float, float]:
    box = row.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("object relation requires bbox")
    values = [float(value) for value in box]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bbox must be finite")
    return (values[0] + values[2]) / 2, (values[1] + values[3]) / 2


def _digest(prefix: str, *values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _certificate_record(fact: EvidenceFact) -> dict[str, Any]:
    """Serialize the raw fields needed to replay episode normalization."""
    return json.loads(json.dumps({
        "fact_id": fact.fact_id,
        "fact_type": fact.fact_type,
        "video_id": fact.video_id,
        "time_span": {"start": fact.time_span.start, "end": fact.time_span.end},
        "frame_ids": list(fact.frame_ids),
        "role_bindings": dict(fact.role_bindings),
        "value": fact.value,
        "source_evidence_ids": list(fact.source_evidence_ids),
        "source_kind": fact.source_kind.value,
        "source_family": fact.source_family.value if fact.source_family else None,
        "derived_from": list(fact.derived_from),
        "derivation_rule": fact.derivation_rule,
        "provenance": fact.provenance,
    }))


@dataclass(frozen=True)
class RelationRowRejection:
    row_id: str
    reason: str


@dataclass(frozen=True)
class RelationConversionResult:
    facts: tuple[EvidenceFact, ...]
    fact_to_row_id: dict[str, str]
    window_matched_row_ids: tuple[str, ...]
    role_matched_row_ids: tuple[str, ...]
    rejected: tuple[RelationRowRejection, ...]


@dataclass(frozen=True)
class TemporalConversionResult:
    facts: tuple[EvidenceFact, ...]
    anchor_time: float | None
    anchor_source_ids: tuple[str, ...]
    rejected_reasons: tuple[str, ...]


class FrozenIndexCompletionSource:
    """Typed, query-independent completion backed by frozen local JSONL views."""

    def __init__(self, index_dir: Path, requirements: QuestionRequirementSet, *,
                 state_window_seconds: float = 10.0,
                 scene_state_rules_version: str = "frozen_detector_scene_state_v1"):
        self.index_dir = Path(index_dir).resolve()
        if isinstance(state_window_seconds, bool) or not isinstance(state_window_seconds, (int, float)):
            raise TypeError("state_window_seconds must be numeric")
        self.state_window_seconds = float(state_window_seconds)
        if not math.isfinite(self.state_window_seconds) or self.state_window_seconds <= 0:
            raise ValueError("state_window_seconds must be finite and positive")
        self.requirements = requirements
        self.rules = SceneStateRules(scene_state_rules_version)
        self.objects = _read_jsonl(self.index_dir / "visual_object_view.jsonl")
        self.relations = _read_jsonl(self.index_dir / "visual_relation_view.jsonl")
        self.events = _read_jsonl(self.index_dir / "visual_event_view.jsonl")
        self.objects_by_id = {
            _id(row, "object_id", "id", "node_id"): row for row in self.objects
        }
        objects_by_frame: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in self.objects:
            objects_by_frame[(
                str(row.get("video_id") or ""),
                str(row.get("frame_id") or ""),
                _number(row, "timestamp"),
            )].append(row)
        self.objects_by_frame = dict(objects_by_frame)
        observations = [self._object_fact(row) for row in self.objects]
        for event in self.events:
            identifier = _id(event, "event_id", "id")
            timestamp = _number(event, "timestamp")
            track_id = str(event.get("track_id") or event.get("target_track_id") or "")
            parents = tuple(_id(row, "object_id", "id") for row in self.objects
                            if str(row.get("video_id")) == str(event["video_id"])
                            and str(row.get("track_id") or "") == track_id
                            and _number(row, "timestamp") == timestamp)
            observations.append(EvidenceFact(
                identifier, "visual_event", str(event["video_id"]), TimeRange(timestamp, timestamp),
                value={"track_id": track_id, "label": event.get("label")},
                source_evidence_ids=(identifier, *parents), derived_from=parents,
                source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
                provenance={"video_id": str(event["video_id"]), "raw_track_id": track_id},
            ))
        self.episode_observations = {fact.fact_id: fact for fact in observations}
        self.track_episodes = build_track_episodes(observations, policy=DEFAULT_NORMALIZATION_POLICY)
        observation_ids_by_track: dict[tuple[str, str], set[str]] = defaultdict(set)
        for episode in self.track_episodes:
            observation_ids_by_track[(episode.video_id, episode.raw_track_id)].update(
                episode.ordered_observation_ids
            )
        self.observation_ids_by_track = {
            key: tuple(sorted(
                observation_ids,
                key=lambda identifier: (
                    self.episode_observations[identifier].time_span.start, identifier,
                ),
            ))
            for key, observation_ids in observation_ids_by_track.items()
        }
        self.episode_by_record = {identifier: episode.episode_id for episode in self.track_episodes
                                  for identifier in episode.ordered_observation_ids}
        self.episode_by_id = {episode.episode_id: episode for episode in self.track_episodes}

    def _target_metadata(self, row: dict[str, Any], *, role: str = "subject") -> dict[str, Any]:
        identifier = _id(row, "object_id", "id")
        track_id = str(row.get("track_id") or "") or None
        episode_id = getattr(self, "episode_by_record", {}).get(identifier)
        episode = getattr(self, "episode_by_id", {}).get(episode_id)
        metadata = {
            f"{role}_episode_id": episode_id,
            f"{role}_raw_track_id": track_id,
            f"{role}_identity_target_id": track_id or identifier,
            f"{role}_identity_evidence_ids": [identifier],
            **({"raw_track_id": track_id, "identity_target_id": track_id or identifier,
                "identity_evidence_ids": [identifier]} if role == "subject" else {}),
        }
        if episode is not None:
            metadata[f"{role}_episode_proof"] = {
                "policy_version": episode.source_provenance["policy_version"],
                "policy": json.loads(json.dumps(asdict(DEFAULT_NORMALIZATION_POLICY))),
                "ordered_observation_ids": list(episode.ordered_observation_ids),
                "boundary_observation_ids": list(
                    episode.source_provenance["boundary_observation_ids"]
                ),
                "observations": [
                    _certificate_record(self.episode_observations[observation_id])
                    for observation_id in self.observation_ids_by_track[
                        (episode.video_id, episode.raw_track_id)
                    ]
                ],
            }
        return metadata

    def fetch(self, requirement: QuestionRequirement, bundle: EvidenceBundle, *,
              limit: int) -> list[EvidenceFact]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if requirement.requirement_type is RequirementType.SPATIAL_RELATION:
            return self._relations(requirement, bundle)[:limit]
        if requirement.requirement_type in {
            RequirementType.ENTRY_EXIT,
            RequirementType.BEFORE_STATE, RequirementType.AFTER_STATE,
            RequirementType.STATE_CHANGE,
        }:
            result = self._convert_temporal(
                requirement,
                video_id=bundle.video_id,
                source_evidence_ids=self._bundle_sources(bundle),
                windows=(),
                limit=limit,
                include_anchor_state=False,
            )
            return list(result.facts)
        return []

    def _bundle_sources(self, bundle: EvidenceBundle) -> set[str]:
        return {
            str(source_id)
            for source_ids in (bundle.provenance.get("fact_sources") or {}).values()
            for source_id in source_ids
        }

    def _bundle_subject_objects(self, requirement: QuestionRequirement,
                                bundle: EvidenceBundle) -> set[str]:
        result: set[str] = set()
        for fact in bundle.provenance.get("visible_facts") or ():
            if not isinstance(fact, dict):
                continue
            value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
            candidate = {"label": value.get("label"), "color": value.get("color")}
            if not _matches(candidate, requirement.subject):
                continue
            sources = fact.get("source_evidence_ids") or ()
            result.update(str(source_id) for source_id in sources if source_id in self.objects_by_id)
        return result

    def _relations(self, requirement: QuestionRequirement,
                   bundle: EvidenceBundle, *,
                   relation_rows: list[dict[str, Any]] | None = None,
                   include_geometry: bool = True,
                   anchored_subjects_override: set[str] | None = None) -> list[EvidenceFact]:
        anchored_subjects = (
            anchored_subjects_override if anchored_subjects_override is not None
            else self._bundle_subject_objects(requirement, bundle)
        )
        candidates: list[tuple[tuple[Any, ...], tuple[Any, ...], EvidenceFact]] = []
        predicate = (requirement.predicate or "").casefold()
        accepted_types = {predicate}
        if predicate == "near":
            accepted_types = {"nearest_to", "overlap_or_near", "image_near"}
        for relation in self.relations if relation_rows is None else relation_rows:
            if str(relation.get("video_id") or "") != bundle.video_id:
                continue
            relation_type = str(relation.get("relation_type") or "").casefold()
            if relation_type not in accepted_types:
                continue
            left_id = str(relation.get("target_object_id") or "")
            right_id = str(relation.get("related_object_id") or relation.get("neighbor_object_id") or "")
            left, right = self.objects_by_id.get(left_id), self.objects_by_id.get(right_id)
            if left is None or right is None:
                continue
            oriented = None
            if _matches(left, requirement.subject) and _matches(right, requirement.reference_object):
                oriented = (left, right)
            elif _matches(right, requirement.subject) and _matches(left, requirement.reference_object):
                oriented = (right, left)
            if oriented is None:
                continue
            subject, reference = oriented
            subject_id = _id(subject, "object_id", "id")
            reference_id = _id(reference, "object_id", "id")
            if anchored_subjects and subject_id not in anchored_subjects:
                continue
            timestamp = _number(subject, "timestamp")
            if timestamp != _number(reference, "timestamp") or subject.get("frame_id") != reference.get("frame_id"):
                continue
            key = (subject_id, reference_id, timestamp)
            subject_center, reference_center = _center(subject), _center(reference)
            positions = []
            if subject_center[0] < reference_center[0]:
                positions.append("left_of")
            elif subject_center[0] > reference_center[0]:
                positions.append("right_of")
            if subject_center[1] < reference_center[1]:
                positions.append("above")
            elif subject_center[1] > reference_center[1]:
                positions.append("below")
            relation_id = _id(relation, "relation_id", "id")
            subject_identity = assess_identity_categories(
                detector_category=subject.get("label"),
                track_category=subject.get("track_category") or subject.get("track_label"),
                role_category=requirement.subject.category if requirement.subject else None,
                semantic_category=subject.get("semantic_category") or subject.get("visual_category"),
            )
            reference_identity = assess_identity_categories(
                detector_category=reference.get("label"),
                track_category=reference.get("track_category") or reference.get("track_label"),
                role_category=(
                    requirement.reference_object.category if requirement.reference_object else None
                ),
                semantic_category=reference.get("semantic_category") or reference.get("visual_category"),
            )
            statuses = {subject_identity.status, reference_identity.status}
            if IdentityStatus.CONFLICT in statuses:
                overall_status = IdentityStatus.CONFLICT
            elif statuses == {IdentityStatus.CONSISTENT}:
                overall_status = IdentityStatus.CONSISTENT
            else:
                overall_status = IdentityStatus.UNVERIFIED
            identity_consistency = {
                "status": overall_status.value,
                "roles": {
                    "subject": identity_check_payload(subject_identity),
                    "reference": identity_check_payload(reference_identity),
                },
                "warnings": sorted({
                    warning
                    for check in (subject_identity, reference_identity)
                    for warning in check.warnings
                }),
            }
            fact = EvidenceFact(
                fact_id=_digest("local-relation", relation_id, subject_id, reference_id),
                fact_type="image_near" if predicate == "near" else relation_type,
                video_id=bundle.video_id, time_span=TimeRange(timestamp, timestamp),
                frame_ids=(str(subject["frame_id"]),),
                role_bindings={"subject": subject_id, "reference": reference_id},
                value={
                    "relation_type": relation_type,
                    "coordinate_space": "image_plane",
                    "subject": {"label": subject.get("label"), "color": subject.get("color"),
                                "object_id": subject_id},
                    "reference": {"label": reference.get("label"), "color": reference.get("color"),
                                  "object_id": reference_id},
                    "subject_relative_to_reference": positions,
                    "distance_pixels": relation.get("distance_pixels"),
                    "text": (
                        f"{subject.get('color', '')} {subject.get('label', 'object')} {subject_id} is "
                        f"{relation_type.replace('_', ' ')} {reference.get('color', '')} "
                        f"{reference.get('label', 'object')} {reference_id}; image position: "
                        f"{', '.join(positions) or 'aligned'}."
                    ).strip(),
                },
                source_evidence_ids=(relation_id, subject_id, reference_id),
                derived_from=(relation_id, subject_id, reference_id),
                independence_root_ids=(relation_id, subject_id, reference_id),
                episode_id=self.episode_by_record.get(subject_id),
                source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
                derivation_rule="frozen_same_frame_relation_v1",
                provenance={
                    "video_id": bundle.video_id, "frame_id": subject["frame_id"],
                    "source_relation_type": relation_type,
                    "source_index": str(self.index_dir),
                    "identity_consistency": identity_consistency,
                    **self._target_metadata(subject),
                    **self._target_metadata(reference, role="reference"),
                },
            )
            priority = (0 if relation_type == "nearest_to" else 1, timestamp, subject_id, reference_id)
            candidates.append((priority, key, fact))
        if include_geometry:
            candidates.extend(self._geometry_relations(requirement, bundle, anchored_subjects))
        selected = []
        seen: set[tuple[Any, ...]] = set()
        for _, key, fact in sorted(candidates, key=lambda item: item[0]):
            if key in seen:
                continue
            seen.add(key)
            selected.append(fact)
        return selected

    @staticmethod
    def _relation_row_id(row: dict[str, Any], index: int) -> str:
        try:
            return _id(row, "relation_id", "id")
        except ValueError:
            return _digest("relation-row", index, row)

    @staticmethod
    def _relation_time_range(
        row: dict[str, Any], left: dict[str, Any] | None, right: dict[str, Any] | None,
    ) -> TimeRange:
        nested = row.get("time_span") if isinstance(row.get("time_span"), dict) else {}
        if "start_time" in row or "end_time" in row:
            start, end = row.get("start_time"), row.get("end_time")
        elif "start" in row or "end" in row:
            start, end = row.get("start"), row.get("end")
        elif nested:
            start, end = nested.get("start"), nested.get("end")
        elif "timestamp" in row:
            start = end = row.get("timestamp")
        elif left is not None and right is not None:
            start = end = _number(left, "timestamp")
        else:
            raise ValueError("relation row has no time")
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("relation time must be numeric")
        return TimeRange(float(start), float(end))

    def convert_relations(
        self,
        requirement: QuestionRequirement,
        *,
        video_id: str,
        subject_object_ids: Any = (),
        windows: Any = (),
    ) -> RelationConversionResult:
        if requirement.requirement_type is not RequirementType.SPATIAL_RELATION:
            return RelationConversionResult((), {}, (), (), ())
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id must be nonempty")
        if any(not isinstance(window, TimeRange) for window in windows):
            raise TypeError("windows must contain TimeRange values")
        anchored = {str(identifier) for identifier in subject_object_ids}
        predicate = (requirement.predicate or "").casefold()
        accepted_types = {predicate}
        if predicate == "near":
            accepted_types = {"nearest_to", "overlap_or_near", "image_near"}
        normalized_rows: list[dict[str, Any]] = []
        spans: dict[str, TimeRange] = {}
        window_matched: list[str] = []
        role_matched: list[str] = []
        rejected: list[RelationRowRejection] = []

        for index, raw in enumerate(self.relations):
            row_id = self._relation_row_id(raw, index)
            declared_videos = {
                str(raw[name]) for name in ("video_id", "video_name")
                if isinstance(raw.get(name), str) and raw.get(name)
            }
            if not declared_videos or declared_videos != {video_id}:
                rejected.append(RelationRowRejection(row_id, "WRONG_VIDEO"))
                continue
            left_id = str(raw.get("target_object_id") or "")
            right_id = str(raw.get("related_object_id") or raw.get("neighbor_object_id") or "")
            left, right = self.objects_by_id.get(left_id), self.objects_by_id.get(right_id)
            try:
                span = self._relation_time_range(raw, left, right)
            except (TypeError, ValueError, OverflowError):
                rejected.append(RelationRowRejection(row_id, "INVALID_OR_MISSING_TIME"))
                continue
            if windows and not any(
                span.start <= window.end and window.start <= span.end for window in windows
            ):
                rejected.append(RelationRowRejection(row_id, "OUTSIDE_REFINED_WINDOW"))
                continue
            window_matched.append(row_id)
            relation_type = str(raw.get("relation_type") or "").casefold()
            if relation_type not in accepted_types:
                rejected.append(RelationRowRejection(row_id, "UNSUPPORTED_RELATION_TYPE"))
                continue
            if left is None or right is None:
                rejected.append(RelationRowRejection(row_id, "MISSING_ENDPOINT"))
                continue
            if any(str(endpoint.get("video_id") or "") != video_id for endpoint in (left, right)):
                rejected.append(RelationRowRejection(row_id, "WRONG_VIDEO"))
                continue
            oriented = None
            if _matches(left, requirement.subject) and _matches(right, requirement.reference_object):
                oriented = (left, right)
            elif _matches(right, requirement.subject) and _matches(left, requirement.reference_object):
                oriented = (right, left)
            if oriented is None:
                rejected.append(RelationRowRejection(row_id, "ROLE_MISMATCH"))
                continue
            subject, reference = oriented
            subject_id = _id(subject, "object_id", "id")
            if anchored and subject_id not in anchored:
                rejected.append(RelationRowRejection(row_id, "ROLE_MISMATCH"))
                continue
            identities = (
                assess_identity_categories(
                    detector_category=subject.get("label"),
                    track_category=subject.get("track_category") or subject.get("track_label"),
                    role_category=requirement.subject.category if requirement.subject else None,
                    semantic_category=subject.get("semantic_category") or subject.get("visual_category"),
                ),
                assess_identity_categories(
                    detector_category=reference.get("label"),
                    track_category=reference.get("track_category") or reference.get("track_label"),
                    role_category=requirement.reference_object.category if requirement.reference_object else None,
                    semantic_category=reference.get("semantic_category") or reference.get("visual_category"),
                ),
            )
            explicit_conflict = False
            for endpoint, role in ((subject, requirement.subject),
                                   (reference, requirement.reference_object)):
                if role is None or not role.category:
                    continue
                expected = canonical_category(role.category, DEFAULT_NORMALIZATION_POLICY)
                declared = (
                    endpoint.get("track_category") or endpoint.get("track_label"),
                    endpoint.get("semantic_category") or endpoint.get("visual_category"),
                )
                if any(
                    isinstance(value, str) and value.strip()
                    and canonical_category(value, DEFAULT_NORMALIZATION_POLICY) != expected
                    for value in declared
                ):
                    explicit_conflict = True
            if explicit_conflict or any(item.status is IdentityStatus.CONFLICT for item in identities):
                rejected.append(RelationRowRejection(row_id, "IDENTITY_CONFLICT"))
                continue
            try:
                if (_number(subject, "timestamp") != _number(reference, "timestamp")
                        or subject.get("frame_id") != reference.get("frame_id")):
                    raise ValueError("relation endpoints are not same-frame")
                _center(subject)
                _center(reference)
            except (TypeError, ValueError, OverflowError):
                rejected.append(RelationRowRejection(row_id, "INVALID_GEOMETRY"))
                continue
            normalized = dict(raw)
            normalized["video_id"] = video_id
            normalized["relation_id"] = row_id
            normalized_rows.append(normalized)
            spans[row_id] = span
            role_matched.append(row_id)

        bundle_span = TimeRange(
            min((window.start for window in windows), default=0.0),
            max((window.end for window in windows), default=0.0),
        )
        bundle = EvidenceBundle(
            _digest("phase-d-relation-bundle", video_id, requirement.requirement_id),
            video_id,
            bundle_span,
            provenance={"video_id": video_id},
        )
        converted = self._relations(
            requirement,
            bundle,
            relation_rows=normalized_rows,
            include_geometry=False,
            anchored_subjects_override=anchored,
        )
        facts = []
        mapping = {}
        for fact in converted:
            row_id = fact.source_evidence_ids[0]
            facts.append(replace(fact, time_span=spans[row_id]))
            mapping[fact.fact_id] = row_id
        return RelationConversionResult(
            tuple(facts), mapping, tuple(window_matched), tuple(role_matched), tuple(rejected)
        )

    @staticmethod
    def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
        box = row.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        values = tuple(float(value) for value in box)
        if (any(not math.isfinite(value) for value in values)
                or values[0] > values[2] or values[1] > values[3]):
            return None
        return values

    @staticmethod
    def _geometry_identity(
        requirement: QuestionRequirement,
        subject: dict[str, Any],
        references: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        checks = {
            "subject": assess_identity_categories(
                detector_category=subject.get("label"),
                track_category=subject.get("track_category") or subject.get("track_label"),
                role_category=requirement.subject.category if requirement.subject else None,
                semantic_category=subject.get("semantic_category") or subject.get("visual_category"),
            ),
        }
        for index, reference in enumerate(references, 1):
            role = "reference" if index == 1 else f"reference_{index}"
            checks[role] = assess_identity_categories(
                detector_category=reference.get("label"),
                track_category=reference.get("track_category") or reference.get("track_label"),
                role_category=(
                    requirement.reference_object.category if requirement.reference_object else None
                ),
                semantic_category=(
                    reference.get("semantic_category") or reference.get("visual_category")
                ),
            )
        statuses = {check.status for check in checks.values()}
        if IdentityStatus.CONFLICT in statuses:
            status = IdentityStatus.CONFLICT
        elif statuses == {IdentityStatus.CONSISTENT}:
            status = IdentityStatus.CONSISTENT
        else:
            status = IdentityStatus.UNVERIFIED
        return {
            "status": status.value,
            "roles": {role: identity_check_payload(check) for role, check in checks.items()},
            "warnings": sorted({warning for check in checks.values() for warning in check.warnings}),
        }

    def _geometry_fact(
        self,
        requirement: QuestionRequirement,
        subject: dict[str, Any],
        references: tuple[dict[str, Any], ...],
        *,
        predicate: str,
        geometry_test: str,
        distance_pixels: float | None = None,
    ) -> EvidenceFact:
        subject_id = _id(subject, "object_id", "id")
        reference_ids = tuple(_id(row, "object_id", "id") for row in references)
        timestamp = _number(subject, "timestamp")
        source_ids = (subject_id, *reference_ids)
        subject_box = self._bbox(subject)
        reference_boxes = tuple(self._bbox(row) for row in references)
        if subject_box is None or any(box is None for box in reference_boxes):
            raise ValueError("geometry fact requires valid endpoint bboxes")
        subject_center = _center({"bbox": list(subject_box)})
        reference_centers = tuple(_center({"bbox": list(box)}) for box in reference_boxes)
        roles = {"subject": subject_id, "reference": reference_ids[0]}
        if len(reference_ids) == 2:
            roles["reference_2"] = reference_ids[1]
        value = {
            "relation_type": predicate,
            "coordinate_space": "image_plane",
            "subject": {"label": subject.get("label"), "color": subject.get("color"),
                        "object_id": subject_id},
            "reference": {"label": references[0].get("label"),
                          "color": references[0].get("color"),
                          "object_id": reference_ids[0]},
            "distance_pixels": distance_pixels,
            "text": (
                f"{subject.get('color', '')} {subject.get('label', 'object')} {subject_id} is "
                f"{predicate.replace('_', ' ')} {references[0].get('color', '')} "
                f"{references[0].get('label', 'object')} {reference_ids[0]} by explicit "
                f"same-frame image geometry."
            ).strip(),
        }
        if len(references) == 2:
            value["reference_2"] = {
                "label": references[1].get("label"), "color": references[1].get("color"),
                "object_id": reference_ids[1],
            }
        provenance = {
            "video_id": str(subject["video_id"]),
            "frame_id": str(subject["frame_id"]),
            "timestamp": timestamp,
            "source_index": str(self.index_dir),
            "source_observation_ids": list(source_ids),
            "geometry_test": geometry_test,
            "subject_bbox": list(subject_box),
            "subject_center": list(subject_center),
            "reference_bbox": list(reference_boxes[0]),
            "reference_center": list(reference_centers[0]),
            "identity_consistency": self._geometry_identity(requirement, subject, references),
            **self._target_metadata(subject),
            **self._target_metadata(references[0], role="reference"),
        }
        if len(references) == 2:
            provenance.update({
                "predicate_test": "between",
                "reference_2_bbox": list(reference_boxes[1]),
                "reference_2_center": list(reference_centers[1]),
                **self._target_metadata(references[1], role="reference_2"),
            })
        return EvidenceFact(
            fact_id=_digest("local-geometry", predicate, *source_ids, timestamp),
            fact_type="image_near" if predicate == "near" else predicate,
            video_id=str(subject["video_id"]),
            time_span=TimeRange(timestamp, timestamp),
            frame_ids=(str(subject["frame_id"]),),
            role_bindings=roles,
            value=value,
            source_evidence_ids=source_ids,
            derived_from=source_ids,
            independence_root_ids=source_ids,
            source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
            derivation_rule=f"frozen_same_frame_bbox_geometry_v1:{predicate}",
            provenance=provenance,
        )

    def _geometry_relations(
        self,
        requirement: QuestionRequirement,
        bundle: EvidenceBundle,
        anchored_subjects: set[str],
    ) -> list[tuple[tuple[Any, ...], tuple[Any, ...], EvidenceFact]]:
        predicate = (requirement.predicate or "").casefold()
        if predicate not in {"left_of", "right_of", "near", "between"}:
            return []
        candidates = []
        subjects = []
        for row in self.objects:
            if str(row.get("video_id") or "") != bundle.video_id or not _matches(row, requirement.subject):
                continue
            subject_id = _id(row, "object_id", "id")
            if anchored_subjects and subject_id not in anchored_subjects:
                continue
            if self._bbox(row) is not None:
                subjects.append(row)
        for subject in subjects:
            subject_id = _id(subject, "object_id", "id")
            timestamp = _number(subject, "timestamp")
            frame_id = subject.get("frame_id")
            references = []
            frame_objects = self.objects_by_frame.get((bundle.video_id, str(frame_id), timestamp), ())
            for row in frame_objects:
                reference_id = _id(row, "object_id", "id")
                if (reference_id == subject_id
                        or not _matches(row, requirement.reference_object)
                        or self._bbox(row) is None):
                    continue
                references.append(row)
            subject_center = _center(subject)
            if predicate == "between":
                ordered = sorted(references, key=lambda row: (_center(row)[0], _id(row, "object_id", "id")))
                for left_index, left in enumerate(ordered):
                    for right in ordered[left_index + 1:]:
                        left_id = _id(left, "object_id", "id")
                        right_id = _id(right, "object_id", "id")
                        left_x, right_x = _center(left)[0], _center(right)[0]
                        if left_id == right_id or not left_x < subject_center[0] < right_x:
                            continue
                        fact = self._geometry_fact(
                            requirement, subject, (left, right), predicate=predicate,
                            geometry_test="strict_horizontal_center_between",
                        )
                        key = (subject_id, left_id, right_id, timestamp)
                        candidates.append(((2, timestamp, subject_id, left_id, right_id), key, fact))
                continue
            for reference in references:
                reference_id = _id(reference, "object_id", "id")
                reference_center = _center(reference)
                distance = math.dist(subject_center, reference_center)
                if predicate == "left_of":
                    matched = subject_center[0] < reference_center[0]
                    geometry_test = "center_x_strict_order"
                elif predicate == "right_of":
                    matched = subject_center[0] > reference_center[0]
                    geometry_test = "center_x_strict_order"
                else:
                    subject_box = self._bbox(subject)
                    reference_box = self._bbox(reference)
                    subject_extent = max(subject_box[2] - subject_box[0], subject_box[3] - subject_box[1], 1.0)
                    reference_extent = max(reference_box[2] - reference_box[0],
                                           reference_box[3] - reference_box[1], 1.0)
                    matched = distance <= 1.5 * max(subject_extent, reference_extent)
                    geometry_test = "scaled_center_distance_lte_1.5_max_bbox_extent"
                if not matched:
                    continue
                fact = self._geometry_fact(
                    requirement, subject, (reference,), predicate=predicate,
                    geometry_test=geometry_test, distance_pixels=distance,
                )
                key = (subject_id, reference_id, timestamp)
                candidates.append(((2, timestamp, subject_id, reference_id), key, fact))
        return candidates

    def _temporal_target(self) -> RoleReference | None:
        modified = [
            requirement.subject for requirement in self.requirements.requirements
            if requirement.subject and requirement.subject.modifiers
        ]
        if modified:
            return modified[0]
        concrete = [
            requirement.subject for requirement in self.requirements.requirements
            if requirement.subject and requirement.subject.category not in {None, "traffic", "scene"}
        ]
        return concrete[0] if concrete else None

    def _anchor(self, bundle: EvidenceBundle) -> dict[str, Any] | None:
        known_sources = self._bundle_sources(bundle)
        target = self._temporal_target()
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for event in self.events:
            if str(event.get("video_id") or "") != bundle.video_id:
                continue
            event_type = str(event.get("event_type") or event.get("type") or "").casefold()
            if "enter" not in event_type:
                continue
            event_id = _id(event, "event_id", "id")
            event_time = _number(event, "timestamp")
            track_id = str(event.get("track_id") or event.get("target_track_id") or "")
            matches = [
                row for row in self.objects
                if str(row.get("video_id") or "") == bundle.video_id
                and str(row.get("track_id") or "") == track_id
                and _number(row, "timestamp") == event_time
                and _matches(row, target)
            ]
            if not matches:
                continue
            observed = sorted(matches, key=lambda row: _id(row, "object_id", "id"))[0]
            identity_consistency = identity_check_payload(assess_identity_categories(
                detector_category=observed.get("label"),
                track_category=(
                    event.get("track_category") or event.get("track_label") or event.get("label")
                ),
                role_category=target.category if target else None,
                semantic_category=(
                    observed.get("semantic_category") or observed.get("visual_category")
                    or event.get("semantic_category") or event.get("visual_category")
                ),
            ))
            candidate = {
                "event": event, "object": observed, "event_id": event_id,
                "object_id": _id(observed, "object_id", "id"),
                "timestamp": event_time, "track_id": track_id,
                "identity_consistency": identity_consistency,
            }
            priority = (0 if event_id in known_sources else 1, event_time, event_id)
            candidates.append((priority, candidate))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _entry_fact(self, requirement: QuestionRequirement,
                    anchor: dict[str, Any]) -> EvidenceFact:
        event, observed = anchor["event"], anchor["object"]
        timestamp = anchor["timestamp"]
        return EvidenceFact(
            fact_id=_digest("local-entry", anchor["event_id"], anchor["object_id"]),
            fact_type="entry", video_id=str(event["video_id"]),
            time_span=TimeRange(timestamp, timestamp),
            frame_ids=(str(event.get("frame_id") or observed.get("frame_id")),),
            role_bindings={"subject": anchor["track_id"]},
            value={
                "event_type": event.get("event_type"), "region": "middle",
                "label": observed.get("label"), "color": observed.get("color"),
                "track_id": anchor["track_id"], "object_id": anchor["object_id"],
                "identity_status": anchor["identity_consistency"]["status"],
                "text": (
                    f"{observed.get('color', '')} {observed.get('label', 'object')} track "
                    f"{anchor['track_id']} enters the middle region at {timestamp:.2f}s."
                ).strip(),
            },
            source_evidence_ids=(anchor["event_id"], anchor["object_id"]),
            derived_from=(anchor["event_id"], anchor["object_id"]),
            independence_root_ids=(anchor["object_id"],),
            episode_id=self.episode_by_record.get(anchor["object_id"]),
            source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
            derivation_rule="event_object_identity_join_v1",
            provenance={
                "video_id": str(event["video_id"]), "anchor_event_id": anchor["event_id"],
                "anchor_object_id": anchor["object_id"], "source_index": str(self.index_dir),
                "identity_consistency": anchor["identity_consistency"],
                **self._target_metadata(observed),
            },
        )

    def _object_fact(self, row: dict[str, Any]) -> EvidenceFact:
        identifier = _id(row, "object_id", "id")
        timestamp = _number(row, "timestamp")
        return EvidenceFact(
            fact_id=identifier, fact_type="visual_object", video_id=str(row["video_id"]),
            time_span=TimeRange(timestamp, timestamp),
            frame_ids=(str(row["frame_id"]),),
            role_bindings={"subject": identifier},
            value={"label": row.get("label"), "color": row.get("color"),
                   "bbox": row.get("bbox"), "object_id": identifier, "track_id": row.get("track_id")},
            source_evidence_ids=(identifier,),
            source_kind=ObservationSource.DETECTOR_OBSERVATION,
            independence_root_ids=(identifier,),
            episode_id=getattr(self, "episode_by_record", {}).get(identifier),
            provenance={"video_id": str(row["video_id"]), "source_index": str(self.index_dir),
                        **self._target_metadata(row)},
        )

    def _anchor_state_fact(self, requirement: QuestionRequirement,
                           anchor: dict[str, Any]) -> EvidenceFact:
        timestamp = anchor["timestamp"]
        return EvidenceFact(
            fact_id=_digest("anchor-state", anchor["event_id"], anchor["object_id"]),
            fact_type="anchor_state",
            video_id=str(anchor["event"]["video_id"]),
            time_span=TimeRange(timestamp, timestamp),
            frame_ids=(str(anchor["event"].get("frame_id") or anchor["object"].get("frame_id")),),
            role_bindings={"subject": anchor["track_id"]},
            value={
                "event_type": anchor["event"].get("event_type"),
                "anchor_time": timestamp,
                "anchor_role_id": (
                    requirement.temporal_relation.anchor_role_id
                    if requirement.temporal_relation else requirement.requirement_id
                ),
                "identity_status": anchor["identity_consistency"]["status"],
                "causality_claimed": False,
            },
            source_evidence_ids=(anchor["event_id"], anchor["object_id"]),
            derived_from=(anchor["event_id"], anchor["object_id"]),
            independence_root_ids=(anchor["object_id"],),
            episode_id=self.episode_by_record.get(anchor["object_id"]),
            source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
            derivation_rule="event_object_identity_join_v1:anchor_state",
            provenance={
                "video_id": str(anchor["event"]["video_id"]),
                "anchor_event_id": anchor["event_id"],
                "anchor_object_id": anchor["object_id"],
                "source_index": str(self.index_dir),
                "identity_consistency": anchor["identity_consistency"],
                **self._target_metadata(anchor["object"]),
            },
        )

    def _state_from_refined_window(
        self, anchor: dict[str, Any], window: RefinedWindow, *, before: bool,
    ) -> SceneState | None:
        timestamp = anchor["timestamp"]
        rows = [
            row for row in self.objects
            if str(row.get("video_id") or "") == str(anchor["event"]["video_id"])
            and window.time_range.start <= _number(row, "timestamp") <= window.time_range.end
            and (_number(row, "timestamp") < timestamp if before else _number(row, "timestamp") > timestamp)
        ]
        if not rows:
            return None
        observed = [self._object_fact(row) for row in rows]
        states = build_scene_states(observed, [window.time_range], rules=self.rules)
        if not states:
            return None
        anchor_fields = {
            "anchor_time": timestamp,
            "anchor_event_id": anchor["event_id"],
            "anchor_object_id": anchor["object_id"],
            "refined_window_id": window.window_id,
        }
        return replace(
            states[0],
            source_fact_ids=self._representative_state_sources(rows),
            provenance={
                **states[0].provenance,
                **anchor_fields,
                "representative_source_rule": "per_label_max_plus_density_max_v1",
            },
        )

    def _change_fact(
        self, requirement: QuestionRequirement, before: SceneState, after: SceneState,
        anchor: dict[str, Any],
    ) -> EvidenceFact | None:
        changes = compare_scene_states(before, after, requirement)
        if not changes:
            return None
        change = changes[0]
        return replace(
            change,
            derived_from=change.source_evidence_ids,
            independence_root_ids=change.source_evidence_ids,
            role_bindings={"subject": "scene:traffic"},
            value={
                **change.value,
                "anchor_role_id": requirement.temporal_relation.anchor_role_id,
                "identity_status": anchor["identity_consistency"]["status"],
            },
            provenance={
                **change.provenance,
                **self._state_scope(anchor),
                "anchor_time": anchor["timestamp"],
                "anchor_role_id": requirement.temporal_relation.anchor_role_id,
                "anchor_event_id": anchor["event_id"],
                "anchor_object_id": anchor["object_id"],
                "rules_version": self.rules.version,
                "identity_consistency": anchor["identity_consistency"],
                "anchor_episode_id": self.episode_by_record.get(anchor["object_id"]),
                "anchor_identity_target_id": anchor["track_id"],
                "anchor_identity_evidence_ids": [anchor["object_id"]],
            },
        )

    def convert_temporal(
        self,
        requirement: QuestionRequirement,
        *,
        video_id: str,
        source_evidence_ids: Any,
        windows: Any,
        limit: int,
    ) -> TemporalConversionResult:
        return self._convert_temporal(
            requirement,
            video_id=video_id,
            source_evidence_ids=source_evidence_ids,
            windows=windows,
            limit=limit,
            include_anchor_state=True,
        )

    def _convert_temporal(
        self,
        requirement: QuestionRequirement,
        *,
        video_id: str,
        source_evidence_ids: Any,
        windows: Any,
        limit: int,
        include_anchor_state: bool,
    ) -> TemporalConversionResult:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        window_values = tuple(windows)
        if any(not isinstance(window, RefinedWindow) for window in window_values):
            raise TypeError("windows must contain RefinedWindow values")
        if any(window.video_id != video_id for window in window_values):
            return TemporalConversionResult((), None, (), ("WINDOW_VIDEO_MISMATCH",))
        sources = tuple(dict.fromkeys(str(value) for value in source_evidence_ids))
        bundle = EvidenceBundle(
            _digest("phase-d-temporal-bundle", video_id, requirement.requirement_id, sources),
            video_id,
            TimeRange(0, 0),
            provenance={"video_id": video_id, "fact_sources": {"anchor": list(sources)}},
        )
        anchor = self._anchor(bundle)
        if anchor is None:
            return TemporalConversionResult((), None, (), ("ANCHOR_NOT_FOUND",))
        anchor_sources = (anchor["event_id"], anchor["object_id"])
        anchor_episode = self.episode_by_record.get(anchor["object_id"])
        source_episodes = {
            self.episode_by_record[source] for source in sources if source in self.episode_by_record
        }
        if source_episodes and anchor_episode not in source_episodes:
            return TemporalConversionResult((), anchor["timestamp"], anchor_sources,
                                            ("CROSS_EPISODE",))
        if requirement.requirement_type is RequirementType.ENTRY_EXIT:
            return TemporalConversionResult(
                (self._entry_fact(requirement, anchor),)[:limit],
                anchor["timestamp"], anchor_sources, (),
            )
        if requirement.requirement_type not in {
            RequirementType.BEFORE_STATE,
            RequirementType.AFTER_STATE,
            RequirementType.STATE_CHANGE,
        }:
            return TemporalConversionResult((), anchor["timestamp"], anchor_sources,
                                            ("UNSUPPORTED_TEMPORAL_REQUIREMENT",))

        if window_values:
            before_windows = [window for window in window_values if
                              window.temporal_direction is TemporalDirection.BEFORE
                              or window.time_range.end <= anchor["timestamp"]]
            after_windows = [window for window in window_values if
                             window.temporal_direction is TemporalDirection.AFTER
                             or window.time_range.start >= anchor["timestamp"]]
            before = self._state_from_refined_window(anchor, before_windows[0], before=True) \
                if before_windows else None
            after = self._state_from_refined_window(anchor, after_windows[0], before=False) \
                if after_windows else None
        else:
            before, after = self._states(anchor)

        rejected = []
        anchor_fact = self._anchor_state_fact(requirement, anchor) if include_anchor_state else None
        if requirement.requirement_type is RequirementType.BEFORE_STATE:
            if before is None:
                return TemporalConversionResult((), anchor["timestamp"], anchor_sources,
                                                ("BEFORE_STATE_UNAVAILABLE",))
            facts = [self._state_fact(requirement, before, anchor, "before_state")]
            if anchor_fact is not None:
                facts.append(anchor_fact)
            return TemporalConversionResult(tuple(facts[:limit]), anchor["timestamp"], anchor_sources, ())
        if requirement.requirement_type is RequirementType.AFTER_STATE:
            if after is None:
                return TemporalConversionResult((), anchor["timestamp"], anchor_sources,
                                                ("AFTER_STATE_UNAVAILABLE",))
            facts = [] if anchor_fact is None else [anchor_fact]
            facts.append(self._state_fact(requirement, after, anchor, "after_state"))
            return TemporalConversionResult(tuple(facts[:limit]), anchor["timestamp"], anchor_sources, ())
        if before is None or after is None:
            rejected.append("ONE_SIDED_STATE_CHANGE")
            available = []
            if before is not None:
                available.append(self._state_fact(requirement, before, anchor, "before_state"))
            if anchor_fact is not None:
                available.append(anchor_fact)
            if after is not None:
                available.append(self._state_fact(requirement, after, anchor, "after_state"))
            return TemporalConversionResult(tuple(available[:limit]), anchor["timestamp"],
                                            anchor_sources, tuple(rejected))
        before_fact = self._state_fact(requirement, before, anchor, "before_state")
        after_fact = self._state_fact(requirement, after, anchor, "after_state")
        change = self._change_fact(requirement, before, after, anchor)
        facts = [before_fact]
        if anchor_fact is not None:
            facts.append(anchor_fact)
        facts.append(after_fact)
        if change is not None:
            facts.append(change)
        else:
            rejected.append("NO_STATE_CHANGE")
        return TemporalConversionResult(
            tuple(facts[:limit]), anchor["timestamp"], anchor_sources, tuple(rejected)
        )

    def _states(self, anchor: dict[str, Any]) -> tuple[SceneState | None, SceneState | None]:
        timestamp = anchor["timestamp"]
        before_rows = [
            row for row in self.objects
            if str(row.get("video_id") or "") == str(anchor["event"]["video_id"])
            and timestamp - self.state_window_seconds <= _number(row, "timestamp") < timestamp
        ]
        after_rows = [
            row for row in self.objects
            if str(row.get("video_id") or "") == str(anchor["event"]["video_id"])
            and timestamp < _number(row, "timestamp") <= timestamp + self.state_window_seconds
        ]
        if not before_rows or not after_rows:
            return None, None
        before_window = TimeRange(
            min(_number(row, "timestamp") for row in before_rows),
            max(_number(row, "timestamp") for row in before_rows),
        )
        after_window = TimeRange(
            min(_number(row, "timestamp") for row in after_rows),
            max(_number(row, "timestamp") for row in after_rows),
        )
        before = build_scene_states(
            [self._object_fact(row) for row in before_rows], [before_window], rules=self.rules,
        )
        after = build_scene_states(
            [self._object_fact(row) for row in after_rows], [after_window], rules=self.rules,
        )
        if not before or not after:
            return None, None
        before_sources = self._representative_state_sources(before_rows)
        after_sources = self._representative_state_sources(after_rows)
        anchor_fields = {
            "anchor_time": timestamp, "anchor_event_id": anchor["event_id"],
            "anchor_object_id": anchor["object_id"],
        }
        return (
            replace(before[0], source_fact_ids=before_sources,
                    provenance={**before[0].provenance, **anchor_fields,
                                "representative_source_rule": "per_label_max_plus_density_max_v1"}),
            replace(after[0], source_fact_ids=after_sources,
                    provenance={**after[0].provenance, **anchor_fields,
                                "representative_source_rule": "per_label_max_plus_density_max_v1"}),
        )

    @staticmethod
    def _state_scope(anchor: dict[str, Any]) -> dict[str, str]:
        video_id = str(anchor["event"]["video_id"])
        scope_id = _digest(
            "entry-local-scope", video_id, anchor["event_id"], anchor["timestamp"],
        )
        return {
            "scene_id": f"video:{video_id}",
            "scope_id": scope_id,
            "scope_rule": "same_video_entry_anchor_local_window_v1",
        }

    @staticmethod
    def _representative_state_sources(rows: list[dict[str, Any]]) -> tuple[str, ...]:
        by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[str(row["frame_id"])].append(row)
        chosen_frames = {
            min(by_frame, key=lambda frame: (-len(by_frame[frame]), frame))
        }
        labels = sorted({str(row.get("label") or "") for row in rows if row.get("label")})
        for label in labels:
            counts = {
                frame: sum(str(row.get("label") or "") == label for row in frame_rows)
                for frame, frame_rows in by_frame.items()
            }
            chosen_frames.add(min(counts, key=lambda frame: (-counts[frame], frame)))
        return tuple(dict.fromkeys(
            _id(row, "object_id", "id")
            for frame in sorted(chosen_frames)
            for row in by_frame[frame]
        ))

    def _state_fact(self, requirement: QuestionRequirement, state: SceneState,
                    anchor: dict[str, Any], fact_type: str) -> EvidenceFact:
        label = "Before" if fact_type == "before_state" else "After"
        counts = ", ".join(f"{key}={value}" for key, value in sorted(state.object_counts.items()))
        anchor_role_id = requirement.temporal_relation.anchor_role_id
        return EvidenceFact(
            fact_id=_digest(fact_type, state.state_id, anchor["event_id"]),
            fact_type=fact_type, video_id=state.video_id, time_span=state.time_window,
            role_bindings={"subject": "scene:traffic"},
            value={
                "state_id": state.state_id, "object_counts": state.object_counts,
                "traffic_density_proxy": state.traffic_density_proxy,
                "object_count_semantics": "maximum sampled detector count per frame",
                "anchor_time": anchor["timestamp"],
                "anchor_role_id": anchor_role_id,
                "identity_status": anchor["identity_consistency"]["status"],
                "text": (
                    f"{label} the verified entry at {anchor['timestamp']:.2f}s, detector scene state "
                    f"for {state.time_window.start:.2f}-{state.time_window.end:.2f}s: {counts}; "
                    f"maximum visible-object density={state.traffic_density_proxy}."
                ),
            },
            source_evidence_ids=state.source_fact_ids,
            derived_from=state.source_fact_ids,
            independence_root_ids=state.source_fact_ids,
            episode_id=self.episode_by_record.get(anchor["object_id"]),
            source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
            derivation_rule=self.rules.version,
            provenance={
                **state.provenance, "video_id": state.video_id,
                **self._state_scope(anchor),
                "anchor_role_id": anchor_role_id,
                "anchor_event_id": anchor["event_id"], "anchor_object_id": anchor["object_id"],
                "state_id": state.state_id, "source_index": str(self.index_dir),
                "identity_consistency": anchor["identity_consistency"],
                **self._target_metadata(anchor["object"]),
            },
        )
