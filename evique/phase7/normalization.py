"""The shared, deterministic boundary for source lineage normalization."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from statistics import median
from typing import Any, Protocol, Sequence

from .schema import (
    EvidenceFact, IdentityAssessment, IdentityEvidence, IdentityStatus, ObservationSource,
    QuestionRequirement, RegionReference, RequirementSupportDecision, RequirementType, RoleReference, SourceFamily,
    SupportComponentDecision, SupportLevel, TimeRange, TrackEpisode,
)


@dataclass(frozen=True)
class NormalizationPolicy:
    # Implementation defaults; neither paper-derived nor tuned to replay outcomes.
    policy_version: str = "paper-aligned-core-v1"
    max_observation_gap_seconds: float = 5.0
    max_center_jump_ratio: float = 3.0
    long_empty_gap_cadence_multiplier: float = 4.0
    category_compatibility_version: str = "identity-categories-v1"
    category_aliases: tuple[tuple[str, str], ...] = ()


DEFAULT_NORMALIZATION_POLICY = NormalizationPolicy()


@dataclass(frozen=True)
class NormalizationResult:
    facts: tuple[EvidenceFact, ...]
    track_episodes: tuple[TrackEpisode, ...]
    identity_assessments: tuple[IdentityAssessment, ...]
    trace: dict[str, Any]


class Normalizer(Protocol):
    def __call__(
        self, facts: Sequence[EvidenceFact], *, video_id: str,
        policy: NormalizationPolicy = DEFAULT_NORMALIZATION_POLICY,
    ) -> NormalizationResult: ...


def _derived_fact_type_family(fact_type: str) -> SourceFamily | None:
    kind = fact_type.lower()
    if kind.startswith("track_") or kind in {"track", "visual_track", "motion"}:
        return SourceFamily.TRACK_DERIVED
    if kind.startswith("event_") or kind in {
        "event", "visual_event", "entry", "exit", "entry_exit", "entry_exit_event",
        "state_change",
    }:
        return SourceFamily.EVENT_DERIVED
    return None


def _source_family(fact: EvidenceFact) -> SourceFamily:
    derived_family = _derived_fact_type_family(fact.fact_type)
    if derived_family is not None:
        return derived_family
    if fact.source_family is not None:
        return fact.source_family
    return {
        ObservationSource.DETECTOR_OBSERVATION: SourceFamily.DETECTOR,
        ObservationSource.OFFLINE_CAPTION: SourceFamily.SCOPE_SEMANTIC,
        ObservationSource.TRANSCRIPT: SourceFamily.SCOPE_SEMANTIC,
        ObservationSource.QUERY_CONDITIONED_OBSERVATION: SourceFamily.VISUAL_SEMANTIC,
        ObservationSource.RULE_DERIVED_OBSERVATION: SourceFamily.EVENT_DERIVED,
    }.get(fact.source_kind, SourceFamily.OTHER_INDEPENDENT)


def _source_declarations_conflict(fact: EvidenceFact, family: SourceFamily) -> bool:
    """Reject explicit source declarations that could manufacture independence."""
    derived_family = _derived_fact_type_family(fact.fact_type)
    if derived_family is not None:
        # The fact type is authoritative at this boundary and forces derived
        # semantics even when a legacy provider declared an observed family.
        return False
    if fact.source_family is None:
        # An absent family is inferred rather than a contradictory declaration.
        # Any inferred derived semantics still pass the lineage checks below.
        return False
    if family is SourceFamily.OTHER_INDEPENDENT:
        # No dedicated observed kind exists for this legacy escape hatch. It
        # must use its established rule-kind plus explicit observed-provenance
        # contract; an ordinary detector/semantic kind cannot be relabelled as
        # another independent family.
        return fact.source_kind is not ObservationSource.RULE_DERIVED_OBSERVATION
    if family in {SourceFamily.TRACK_DERIVED, SourceFamily.EVENT_DERIVED}:
        return False
    if fact.source_kind is ObservationSource.RULE_DERIVED_OBSERVATION:
        # Rule-derived semantics already require ancestry and therefore fail
        # closed through MISSING_DERIVED_FROM/MISSING_PARENT when malformed.
        return False
    compatible = {
        SourceFamily.DETECTOR: {ObservationSource.DETECTOR_OBSERVATION},
        SourceFamily.SCOPE_SEMANTIC: {
            ObservationSource.OFFLINE_CAPTION, ObservationSource.TRANSCRIPT,
        },
        SourceFamily.VISUAL_SEMANTIC: {ObservationSource.QUERY_CONDITIONED_OBSERVATION},
        SourceFamily.TRACK_DERIVED: {ObservationSource.RULE_DERIVED_OBSERVATION},
        SourceFamily.EVENT_DERIVED: {ObservationSource.RULE_DERIVED_OBSERVATION},
    }
    return fact.source_kind not in compatible[family]


def _is_derived(fact: EvidenceFact) -> bool:
    return (
        bool(fact.derived_from or fact.derivation_rule)
        or fact.source_family in {SourceFamily.TRACK_DERIVED, SourceFamily.EVENT_DERIVED}
        or (fact.source_kind is ObservationSource.RULE_DERIVED_OBSERVATION
            and fact.source_family is not SourceFamily.OTHER_INDEPENDENT)
    )


_SEMANTIC_FAMILIES = {SourceFamily.SCOPE_SEMANTIC, SourceFamily.VISUAL_SEMANTIC}


def _semantic_source_tokens(fact: EvidenceFact) -> set[str]:
    context = (
        fact.video_id,
        _local_target_id(fact),
        fact.provenance.get("scene_id"),
        fact.provenance.get("scope_id"),
    )
    tokens = {
        json.dumps((*context, "frame", frame_id), separators=(",", ":"))
        for frame_id in fact.frame_ids
    }
    value = fact.value if isinstance(fact.value, dict) else {}
    text = value.get("text")
    if isinstance(text, str) and text.strip():
        normalized_text = " ".join(text.split()).casefold()
        tokens.add(json.dumps(
            (*context, "generated_text", fact.time_span.start, fact.time_span.end,
             normalized_text),
            separators=(",", ":"),
        ))
    return tokens


def _collapsed_semantic_roots(facts: Sequence[EvidenceFact]) -> dict[str, str]:
    """Give cross-family wrappers over one local source one provenance root."""
    candidates = [
        fact for fact in facts
        if fact.source_family in _SEMANTIC_FAMILIES
        and not _is_derived(fact)
        and fact.source_evidence_ids
        and _semantic_source_tokens(fact)
    ]
    parents = {fact.fact_id: fact.fact_id for fact in candidates}

    def root(identifier: str) -> str:
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    def join(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    tokens = {fact.fact_id: _semantic_source_tokens(fact) for fact in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1:]:
            if (left.source_family is not right.source_family
                    and tokens[left.fact_id] & tokens[right.fact_id]):
                join(left.fact_id, right.fact_id)

    components: dict[str, list[EvidenceFact]] = {}
    for fact in candidates:
        components.setdefault(root(fact.fact_id), []).append(fact)
    nonsemantic_roots = {
        record_id for fact in facts if fact.source_family not in _SEMANTIC_FAMILIES
        for record_id in fact.source_evidence_ids
    }
    collapsed: dict[str, str] = {}
    for members in components.values():
        if len({fact.source_family for fact in members}) < 2:
            continue
        source_roots = sorted({record_id for fact in members
                               for record_id in fact.source_evidence_ids})
        shared_nonsemantic = sorted(set(source_roots) & nonsemantic_roots)
        canonical_root = (shared_nonsemantic or source_roots)[0]
        collapsed.update({fact.fact_id: canonical_root for fact in members})
    return collapsed


def _resolve_lineage(
    facts: Sequence[EvidenceFact],
) -> tuple[tuple[EvidenceFact, ...], list[dict[str, str]]]:
    normalized = tuple(replace(fact, source_family=_source_family(fact)) for fact in facts)
    by_id = {fact.fact_id: fact for fact in normalized}
    if len(by_id) != len(normalized):
        raise ValueError("duplicate fact_id in normalization input")
    by_record: dict[str, list[str]] = {}
    for fact in normalized:
        for record_id in fact.source_evidence_ids:
            # Explicit parent records are inputs, not outputs represented by
            # this derivation. Copies sharing an external input must not become
            # each other's ancestors.
            if record_id not in fact.derived_from:
                by_record.setdefault(record_id, []).append(fact.fact_id)

    collapsed_semantic_roots = _collapsed_semantic_roots(normalized)
    roots = {fact_id: set() for fact_id in by_id}
    errors = {
        fact.fact_id: ({"CONTRADICTORY_SOURCE_DECLARATION"}
                       if _source_declarations_conflict(fact, normalized_fact.source_family)
                       else set())
        for fact, normalized_fact in zip(facts, normalized)
    }
    dependencies = {fact_id: set() for fact_id in by_id}
    for fact in normalized:
        fact_id = fact.fact_id
        if _is_derived(fact):
            if not fact.derived_from:
                errors[fact_id].add("MISSING_DERIVED_FROM")
            for parent in fact.derived_from:
                if parent in by_id:
                    parent_ids = [parent]
                else:
                    # A record reference follows its represented fact(s), rather
                    # than minting a new root for an already derived record.
                    parent_ids = sorted(identifier for identifier in by_record.get(parent, [])
                                        if identifier != fact_id)
                if parent_ids:
                    dependencies[fact_id].update(parent_ids)
                elif parent in fact.source_evidence_ids:
                    roots[fact_id].add(parent)
                else:
                    errors[fact_id].add("MISSING_PARENT")
        elif fact.source_family is SourceFamily.OTHER_INDEPENDENT:
            # Explicit provenance must identify the actual observed record.
            provenance = fact.provenance
            if (provenance.get("observed") is True
                    and provenance.get("is_derived") is False
                    and provenance.get("source_record_id") in fact.source_evidence_ids):
                roots[fact_id].add(provenance["source_record_id"])
            else:
                errors[fact_id].add("UNVERIFIED_INDEPENDENT_SOURCE")
        elif fact.source_evidence_ids:
            collapsed_root = collapsed_semantic_roots.get(fact_id)
            roots[fact_id].update(
                (collapsed_root,) if collapsed_root is not None else fact.source_evidence_ids
            )
        else:
            errors[fact_id].add("MISSING_SOURCE_RECORD")

    children = {fact_id: set() for fact_id in by_id}
    for fact_id, parents in dependencies.items():
        for parent in parents:
            children[parent].add(fact_id)
    remaining = {fact_id: len(parents) for fact_id, parents in dependencies.items()}
    ready = deque(sorted(fact_id for fact_id, count in remaining.items() if not count))
    processed: set[str] = set()
    while ready:
        fact_id = ready.popleft()
        processed.add(fact_id)
        if errors[fact_id]:
            roots[fact_id].clear()
        for child in sorted(children[fact_id]):
            roots[child].update(roots[fact_id])
            errors[child].update(errors[fact_id])
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)

    # Topological leftovers are precisely cycles and their descendants. Spread
    # all inherited failures to a fixed point so trace codes do not depend on
    # traversal order, even when a cycle also has an absent ancestor.
    cyclic = set(by_id) - processed
    pending = deque(sorted(cyclic))
    for fact_id in cyclic:
        roots[fact_id].clear()
        errors[fact_id].add("CYCLIC_DERIVATION")
    while pending:
        fact_id = pending.popleft()
        for child in sorted(children[fact_id]):
            added = errors[fact_id] - errors[child]
            if added:
                errors[child].update(added)
                pending.append(child)

    output = tuple(replace(fact, independence_root_ids=tuple(sorted(roots[fact.fact_id])))
                   for fact in normalized)
    failures = [
        {"fact_id": fact_id, "status": status}
        for fact_id in sorted(errors) for status in sorted(errors[fact_id])
    ]
    return output, failures


def normalize_source_lineage(facts: Sequence[EvidenceFact]) -> tuple[EvidenceFact, ...]:
    """Keep every fact in input order and recompute roots from observed ancestry."""
    return _resolve_lineage(facts)[0]


def _category(fact: EvidenceFact) -> str | None:
    if isinstance(fact.value, dict):
        for key in ("observed_category", "category", "label"):
            label = fact.value.get(key)
            if isinstance(label, str) and label.strip():
                return label
    return None


def _has_identity_label(fact: EvidenceFact) -> bool:
    return _category(fact) is not None


def canonical_category(value: str, policy: NormalizationPolicy) -> str:
    """Case-folded equality, with optional globally configured category aliases."""
    category = value.strip().casefold()
    aliases = {alias.strip().casefold(): target.strip().casefold()
               for alias, target in policy.category_aliases}
    return aliases.get(category, category)


def _raw_track_id(fact: EvidenceFact) -> str | None:
    value = fact.value if isinstance(fact.value, dict) else {}
    for candidate in (value.get("track_id"), fact.provenance.get("raw_track_id"),
                      fact.provenance.get("track_id")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _local_target_id(fact: EvidenceFact) -> str:
    value = fact.value if isinstance(fact.value, dict) else {}
    for candidate in (value.get("local_target_id"), fact.provenance.get("local_target_id"),
                      value.get("object_id"), _raw_track_id(fact)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    endpoints = set(fact.role_bindings.values())
    return next(iter(endpoints)) if len(endpoints) == 1 else fact.fact_id


_IDENTITY_REFERENCE_KEYS = {"identity_assessment_id", "identity_status"}
_ELIGIBLE_IDENTITY_FAMILIES = {
    SourceFamily.DETECTOR, SourceFamily.SCOPE_SEMANTIC,
    SourceFamily.VISUAL_SEMANTIC, SourceFamily.OTHER_INDEPENDENT,
}


def _identity_endpoint_matches(fact: EvidenceFact, local_target_id: str) -> bool:
    """Reject an identity label that names a different explicit subject endpoint."""
    value = fact.value if isinstance(fact.value, dict) else {}
    wrapper_ids = set(fact.source_evidence_ids)
    frozen_wrapper = fact.provenance.get("frozen_evidence_id")
    if isinstance(frozen_wrapper, str) and frozen_wrapper:
        wrapper_ids.add(frozen_wrapper)
    for key in ("subject", "actor"):
        endpoint = fact.role_bindings.get(key)
        if endpoint is None or endpoint == local_target_id:
            continue
        # The frozen object adapter historically exposes its record wrapper as
        # both ``observed`` and generic ``subject`` when no object endpoint is
        # present. Accept only that traceable wrapper alias; an arbitrary
        # observed/subject pair must not override an explicit target mismatch.
        if (
            key == "subject"
            and fact.fact_type == "visual_object"
            and fact.role_bindings.get("observed") == endpoint
            and endpoint in wrapper_ids
            and value.get("object_id") is None
            and value.get("local_target_id") is None
            and _raw_track_id(fact) == local_target_id
        ):
            continue
        return False
    return True


def identity_evidence_from_facts(facts: Sequence[EvidenceFact]) -> tuple[IdentityEvidence, ...]:
    """Extract explicit labels from normalized facts without inventing roots."""
    return tuple(
        IdentityEvidence(
            fact.fact_id, label, _source_family(fact), fact.derived_from,
            fact.independence_root_ids, fact.episode_id, _local_target_id(fact), fact.time_span,
            deepcopy({key: value for key, value in fact.provenance.items()
                      if key not in _IDENTITY_REFERENCE_KEYS}),
        )
        for fact in facts
        if (label := _category(fact)) is not None
        and _identity_endpoint_matches(fact, _local_target_id(fact))
    )


def assess_identity(
    evidence: Sequence[IdentityEvidence], *, episode_id: str | None,
    local_target_id: str, policy: NormalizationPolicy,
) -> IdentityAssessment:
    """Assess one target and episode, counting connected roots as one source."""
    labels = tuple(sorted(
        (item for item in evidence if item.episode_id == episode_id
         and item.local_target_id == local_target_id), key=lambda item: item.evidence_id,
    ))
    components: list[tuple[set[str], list[IdentityEvidence]]] = []
    for item in labels:
        if not item.independence_root_ids:
            continue
        roots, members = set(item.independence_root_ids), [item]
        separate = []
        for previous_roots, previous_members in components:
            if roots.isdisjoint(previous_roots):
                separate.append((previous_roots, previous_members))
            else:
                roots.update(previous_roots)
                members.extend(previous_members)
        # Existing components are disjoint; all components intersecting an
        # item's original roots have now been joined, including transitive links.
        components = separate + [(roots, members)]
    components.sort(key=lambda component: tuple(sorted(component[0])))
    groups = tuple({
        "root_ids": sorted(roots),
        "evidence_ids": sorted(item.evidence_id for item in members),
        "source_families": sorted({item.source_family.value for item in members}),
        "canonical_categories": sorted({canonical_category(item.observed_category, policy)
                                         for item in members}),
    } for roots, members in components)
    voting_components = []
    for _, members in components:
        eligible_members = [
            item for item in members
            if item.source_family in _ELIGIBLE_IDENTITY_FAMILIES and not item.derived_from
        ]
        families = {item.source_family for item in eligible_members}
        if len(families) == 1:
            voting_components.append((
                families,
                {canonical_category(item.observed_category, policy)
                 for item in eligible_members},
            ))
    # Independent roots retain contradictory labels even when one root also
    # carries a copied representation from another family.  Mixed-family
    # components cannot confirm identity, but the wrapper must not hide the
    # originating root's conflict with a different independent root.
    conflict = any(
        left != right for index, group in enumerate(groups)
        for other in groups[index + 1:]
        for left in group["canonical_categories"] for right in other["canonical_categories"]
    )
    confirmed = any(
        left != right for index, (families, _) in enumerate(voting_components)
        for other, _ in voting_components[index + 1:]
        for left in families for right in other
    )
    categories = {canonical_category(item.observed_category, policy) for item in labels}
    if conflict:
        status, reasons = IdentityStatus.CONFLICT, ("INDEPENDENT_CATEGORY_CONFLICT",)
    elif confirmed and len(categories) == 1:
        status, reasons = IdentityStatus.CONFIRMED, ("INDEPENDENT_COMPATIBLE_SOURCE_FAMILIES",)
    else:
        status, reasons = IdentityStatus.UNVERIFIED, ("INSUFFICIENT_INDEPENDENT_SOURCE_FAMILIES",)
    return IdentityAssessment(
        status, episode_id, local_target_id, labels, groups,
        next(iter(categories)) if len(categories) == 1 else None, reasons,
    )


def identity_for_target(
    normalization: NormalizationResult, target_id: str, episode_id: str | None,
) -> IdentityAssessment | None:
    assessments = [assessment for assessment in normalization.identity_assessments
                   if assessment.local_target_id == target_id and assessment.episode_id == episode_id]
    return assessments[0] if len(assessments) == 1 else None


def _observation_context(fact: EvidenceFact) -> str:
    """A raw local ID supplies no continuity outside this exact observed context."""
    parts = (fact.video_id, fact.provenance.get("scene_id"), fact.provenance.get("scope_id"),
             sorted(fact.frame_ids), fact.time_span.start, fact.time_span.end)
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _same_observation_context(left: EvidenceFact, right: EvidenceFact) -> bool:
    """Require an explicit frame before treating simultaneous labels as one observation."""
    return bool(left.frame_ids and right.frame_ids) and _observation_context(left) == _observation_context(right)


def _identity_assessment_id(
    assessment: IdentityAssessment, policy: NormalizationPolicy, video_id: str,
    *, observation_context: str | None = None,
) -> str:
    parts = (policy.policy_version, policy.category_compatibility_version,
             video_id, assessment.local_target_id, assessment.episode_id,
             observation_context)
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"identity:{digest}"


def _geometry(fact: EvidenceFact) -> tuple[float, float, float] | None:
    box = fact.value.get("bbox") if isinstance(fact.value, dict) else None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in box):
        return None
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, math.hypot(x2 - x1, y2 - y1)


def _track_episode_id(
    policy_version: str, video_id: str, raw_track_id: str,
    ordered_observation_ids: Sequence[str],
) -> str:
    parts = (policy_version, video_id, raw_track_id,
             ordered_observation_ids[0], ordered_observation_ids[-1])
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"episode:{digest}"


def _certified_observation(
    payload: Any, *, video_id: str, raw_track_id: str,
) -> EvidenceFact | None:
    """Rebuild one certificate record without trusting normalized outputs."""
    if not isinstance(payload, dict):
        return None
    span = payload.get("time_span")
    provenance = payload.get("provenance")
    sequence_fields = {
        name: payload.get(name)
        for name in ("frame_ids", "source_evidence_ids", "derived_from")
    }
    if (not isinstance(span, dict) or not isinstance(provenance, dict)
            or any(not isinstance(value, (list, tuple)) for value in sequence_fields.values())):
        return None
    try:
        source_kind = ObservationSource(payload.get("source_kind"))
        source_family_value = payload.get("source_family")
        source_family = (None if source_family_value is None
                         else SourceFamily(source_family_value))
        observation = EvidenceFact(
            fact_id=payload["fact_id"], fact_type=payload["fact_type"],
            video_id=payload["video_id"],
            time_span=TimeRange(span["start"], span["end"]),
            frame_ids=tuple(sequence_fields["frame_ids"]),
            role_bindings=deepcopy(payload.get("role_bindings", {})),
            value=deepcopy(payload.get("value")),
            source_evidence_ids=tuple(sequence_fields["source_evidence_ids"]),
            source_kind=source_kind, source_family=source_family,
            derived_from=tuple(sequence_fields["derived_from"]),
            derivation_rule=payload.get("derivation_rule"),
            provenance=deepcopy(provenance),
        )
    except (KeyError, TypeError, ValueError):
        return None
    value = observation.value if isinstance(observation.value, dict) else {}
    declared_track_ids = {
        candidate for candidate in (
            value.get("track_id"), observation.provenance.get("raw_track_id"),
            observation.provenance.get("track_id"),
        )
        if isinstance(candidate, str) and candidate
    }
    if (observation.video_id != video_id
            or observation.provenance.get("video_id") != video_id
            or not observation.source_evidence_ids
            or declared_track_ids != {raw_track_id}):
        return None
    return observation


def _explicit_context_conflict(left: EvidenceFact, right: EvidenceFact) -> bool:
    """Treat only two present, unequal local context declarations as a conflict."""
    return any(
        left.provenance.get(key) is not None
        and right.provenance.get(key) is not None
        and left.provenance.get(key) != right.provenance.get(key)
        for key in ("scene_id", "scope_id")
    )


def _certificate_binds_fact(
    fact: EvidenceFact, *, observation_ids: Sequence[str],
    evidence_ids: Sequence[str], certified: Sequence[EvidenceFact],
) -> bool:
    """Require a local observed-member link, not a reusable episode assertion."""
    members = {item.fact_id: item for item in certified if item.fact_id in observation_ids}
    if len(members) != len(observation_ids):
        return False
    source_index = fact.provenance.get("source_index")
    source_bound = (
        isinstance(source_index, str) and bool(source_index)
        and any(member.provenance.get("source_index") == source_index
                for member in members.values())
    )
    if not _has_scope(fact) and not source_bound:
        return False
    if any(_explicit_context_conflict(fact, member) for member in members.values()):
        return False
    member_values = tuple(members.values())
    if any(_explicit_context_conflict(left, right)
           for index, left in enumerate(member_values)
           for right in member_values[index + 1:]):
        return False

    anchor_object_id = fact.provenance.get("anchor_object_id")
    if anchor_object_id is not None:
        if not isinstance(anchor_object_id, str) or anchor_object_id not in members:
            return False
        anchor_source_bound = (
            isinstance(source_index, str) and bool(source_index)
            and members[anchor_object_id].provenance.get("source_index") == source_index
        )
        if not _has_scope(fact) and not anchor_source_bound:
            return False
        anchor_time = _anchor_time(fact)
        if (_finite_number(anchor_time)
                and not members[anchor_object_id].time_span.start
                <= float(anchor_time) <= members[anchor_object_id].time_span.end):
            return False
        return True

    linked_ids = (set(fact.source_evidence_ids) | set(fact.derived_from)) & set(members)
    if linked_ids:
        return True

    # Compatibility for the legacy singleton certificate: an exact identity
    # member at the same observed instant is bounded to that one observation.
    # Multi-member proofs need an immutable ancestry/anchor link and therefore
    # cannot be reused across unrelated before/after facts.
    if len(members) == 1 and set(evidence_ids) == set(members):
        member = next(iter(members.values()))
        return (fact.time_span.start <= member.time_span.end
                and member.time_span.start <= fact.time_span.end)
    return False


def _preassigned_observed_episode(
    fact: EvidenceFact, policy: NormalizationPolicy,
) -> str | None:
    """Accept a derived fact's episode only with endpoint-specific observation proof."""
    if not _is_derived(fact) or not isinstance(fact.episode_id, str) or not fact.episode_id:
        return None
    raw_track_id = _raw_track_id(fact)
    if raw_track_id is None:
        return None
    for role_key in fact.role_bindings:
        if fact.provenance.get(f"{role_key}_episode_id") != fact.episode_id:
            continue
        target_ids = {
            value for suffix in ("raw_track_id", "identity_target_id", "track_id")
            if isinstance((value := fact.provenance.get(f"{role_key}_{suffix}")), str)
            and value
        }
        evidence_ids = fact.provenance.get(f"{role_key}_identity_evidence_ids")
        proof = fact.provenance.get(f"{role_key}_episode_proof")
        if (raw_track_id not in target_ids or not isinstance(evidence_ids, (list, tuple))
                or not evidence_ids
                or not all(isinstance(value, str) and value for value in evidence_ids)
                or not isinstance(proof, dict)
                or proof.get("policy_version") != policy.policy_version):
            continue
        expected_policy = json.loads(json.dumps(asdict(policy)))
        if proof.get("policy") != expected_policy:
            continue
        observation_ids = proof.get("ordered_observation_ids")
        boundary_ids = proof.get("boundary_observation_ids")
        observation_records = proof.get("observations")
        if (not isinstance(observation_ids, (list, tuple)) or not observation_ids
                or not all(isinstance(value, str) and value for value in observation_ids)
                or len(set(observation_ids)) != len(observation_ids)
                or not set(evidence_ids) <= set(observation_ids)
                or not isinstance(boundary_ids, (list, tuple)) or not boundary_ids
                or len(boundary_ids) > 2
                or not all(isinstance(value, str) and value for value in boundary_ids)
                or len(set(boundary_ids)) != len(boundary_ids)
                or not set(boundary_ids) <= set(observation_ids)
                or not isinstance(observation_records, (list, tuple))
                or not observation_records):
            continue
        certified = tuple(
            _certified_observation(
                payload, video_id=fact.video_id, raw_track_id=raw_track_id,
            )
            for payload in observation_records
        )
        certified_ids = tuple(item.fact_id for item in certified if item is not None)
        if (any(item is None for item in certified)
                or len(set(certified_ids)) != len(certified_ids)
                or not set(observation_ids) <= set(certified_ids)):
            continue
        try:
            canonical_certified, lineage_failures = _resolve_lineage(
                tuple(item for item in certified if item is not None),
            )
            if lineage_failures:
                continue
            reconstructed = build_track_episodes(
                canonical_certified, policy=policy,
            )
        except (TypeError, ValueError):
            continue
        matches = [
            episode for episode in reconstructed
            if (episode.video_id == fact.video_id
                and episode.raw_track_id == raw_track_id
                and episode.episode_id == fact.episode_id
                and episode.ordered_observation_ids == tuple(observation_ids)
                and tuple(episode.source_provenance.get("boundary_observation_ids", ()))
                == tuple(boundary_ids))
        ]
        if (len(matches) == 1 and _certificate_binds_fact(
                fact, observation_ids=observation_ids, evidence_ids=evidence_ids,
                certified=canonical_certified,
        )):
            return fact.episode_id
    return None


def _derivation_requires_observed_episode(fact: EvidenceFact) -> bool:
    return (_is_derived(fact) and _geometry(fact) is None
            and fact.fact_type in {
                "before_state", "after_state", "scene_state_before", "scene_state_after",
                "event", "visual_event", "entry", "exit", "entry_exit", "entry_exit_event",
            })


def _derived_event_can_seed_episode(fact: EvidenceFact) -> bool:
    return (_is_derived(fact) and _geometry(fact) is None
            and fact.fact_type in {
                "event", "visual_event", "entry", "exit", "entry_exit", "entry_exit_event",
            })


def _observation_snapshot(fact: EvidenceFact) -> dict[str, Any]:
    value = fact.value if isinstance(fact.value, dict) else {}
    return {
        "observation_id": fact.fact_id, "timestamp": fact.time_span.start,
        "bbox": deepcopy(value.get("bbox")),
        "frame_id": value.get("frame_id") or (fact.frame_ids[0] if fact.frame_ids else None),
        "evidence_id": fact.source_evidence_ids[0] if fact.source_evidence_ids else None,
        "source_provenance": deepcopy(fact.provenance),
    }


def _episode(
    observations: Sequence[EvidenceFact], *, video_id: str, raw_track_id: str,
    policy: NormalizationPolicy, split_reasons: tuple[str, ...],
    split_measurements: dict[str, Any], cadence: float | None,
) -> TrackEpisode:
    observation_ids = tuple(fact.fact_id for fact in observations)
    episode_id = _track_episode_id(
        policy.policy_version, video_id, raw_track_id, observation_ids,
    )
    labels = tuple(
        IdentityEvidence(
            evidence_id=fact.fact_id, observed_category=_category(fact),
            source_family=fact.source_family, derived_from=fact.derived_from,
            independence_root_ids=fact.independence_root_ids, episode_id=episode_id,
            local_target_id=raw_track_id, time_span=fact.time_span,
            provenance=deepcopy(fact.provenance),
        )
        for fact in observations if _has_identity_label(fact)
    )
    snapshots = [_observation_snapshot(fact) for fact in observations]
    return TrackEpisode(
        episode_id=episode_id, raw_track_id=raw_track_id, video_id=video_id,
        start_time=observations[0].time_span.start,
        end_time=max(fact.time_span.end for fact in observations),
        ordered_observation_ids=observation_ids,
        category_evidence=labels, split_reasons=split_reasons,
        source_provenance={
            "video_id": video_id, "raw_track_id": raw_track_id,
            "policy_version": policy.policy_version,
            "policy": json.loads(json.dumps(asdict(policy))),
            "boundary_observation_ids": [observations[0].fact_id, observations[-1].fact_id]
            if len(observations) > 1 else [observations[0].fact_id],
            "median_positive_cadence_seconds": cadence,
            "split_reasons": list(split_reasons), "split_measurements": split_measurements,
            "observations": snapshots,
        },
    )


def _attach_geometryless_derivations(
    episodes: Sequence[TrackEpisode], group: Sequence[EvidenceFact],
) -> tuple[TrackEpisode, ...]:
    """Attach derived state/event facts only to a unique observed-root episode."""
    by_id = {fact.fact_id: fact for fact in group}
    roots_by_episode = {
        episode.episode_id: {
            root for fact_id in episode.ordered_observation_ids
            if _geometry(by_id[fact_id]) is not None or not _is_derived(by_id[fact_id])
            for root in by_id[fact_id].independence_root_ids
        }
        for episode in episodes
    }
    anchored_ids = {
        fact_id for episode in episodes for fact_id in episode.ordered_observation_ids
    }
    assignments: dict[str, list[EvidenceFact]] = {episode.episode_id: [] for episode in episodes}
    for fact in group:
        if (fact.fact_id in anchored_ids or not _derivation_requires_observed_episode(fact)
                or not fact.independence_root_ids):
            continue
        roots = set(fact.independence_root_ids)
        matches = [episode.episode_id for episode in episodes
                   if roots & roots_by_episode[episode.episode_id]]
        if len(matches) == 1:
            assignments[matches[0]].append(fact)

    attached = []
    for episode in episodes:
        additions = assignments[episode.episode_id]
        if not additions:
            attached.append(episode)
            continue
        members = [by_id[fact_id] for fact_id in episode.ordered_observation_ids] + additions
        ordered_ids = tuple(fact.fact_id for fact in sorted(
            members, key=lambda fact: (fact.time_span.start, fact.fact_id),
        ))
        snapshots = {
            snapshot["observation_id"]: deepcopy(snapshot)
            for snapshot in episode.source_provenance.get("observations", [])
        }
        snapshots.update({fact.fact_id: _observation_snapshot(fact) for fact in additions})
        source_provenance = deepcopy(episode.source_provenance)
        source_provenance["observations"] = [snapshots[fact_id] for fact_id in ordered_ids]
        attached.append(replace(
            episode, ordered_observation_ids=ordered_ids,
            source_provenance=source_provenance,
        ))
    return tuple(attached)


def build_track_episodes(
    facts: Sequence[EvidenceFact], *, policy: NormalizationPolicy,
) -> tuple[TrackEpisode, ...]:
    """Split raw track references deterministically without changing observations."""
    groups: dict[tuple[str, str], list[EvidenceFact]] = {}
    for fact in normalize_source_lineage(facts):
        raw_track_id = _raw_track_id(fact)
        if raw_track_id is not None:
            groups.setdefault((fact.video_id, raw_track_id), []).append(fact)
    episodes = []
    for (video_id, raw_track_id), group in sorted(groups.items()):
        # Geometryless derivations represent earlier observations; their own
        # timestamps cannot establish continuity or cadence. Derived state/event
        # facts receive membership from observed ancestry after splitting.
        all_ordered = sorted(group, key=lambda fact: (fact.time_span.start, fact.fact_id))
        continuity = [fact for fact in all_ordered
                      if _geometry(fact) is not None or not _is_derived(fact)]
        if not continuity:
            # A frozen tracker row is itself the only available raw-track
            # observation. It may seed an episode, but once direct observations
            # are present it is attached by lineage like every other derivation.
            continuity = [fact for fact in all_ordered
                          if fact.fact_type in {"track", "visual_track", "track_motion"}]
        if not continuity:
            # With no direct/geometric observations, standalone event anchors
            # are the only temporal continuity evidence and must still obey the
            # default episode gap policy.
            continuity = [fact for fact in all_ordered if _derived_event_can_seed_episode(fact)]
        ordered = [fact for fact in all_ordered if not _derivation_requires_observed_episode(fact)]
        if not ordered:
            ordered = list(continuity) or [fact for fact in all_ordered if _derived_event_can_seed_episode(fact)]
        if not ordered:
            continue
        group_episode_start = len(episodes)
        continuity_ids = {fact.fact_id for fact in continuity}
        intervals = [current.time_span.start - previous.time_span.start
                     for previous, current in zip(continuity, continuity[1:])
                     if current.time_span.start > previous.time_span.start]
        cadence = median(intervals) if intervals else None
        current_episode = [ordered[0]]
        previous_geometric = ordered[0] if _geometry(ordered[0]) is not None else None
        previous_temporal = ordered[0] if ordered[0].fact_id in continuity_ids else None
        start_reasons: tuple[str, ...] = ()
        start_measurements: dict[str, Any] = {}
        for current in ordered[1:]:
            is_continuity = current.fact_id in continuity_ids
            gap = (current.time_span.start - previous_temporal.time_span.start
                   if is_continuity and previous_temporal is not None else None)
            reasons = []
            measurements = {
                "previous_observation_id": previous_temporal.fact_id if previous_temporal is not None else None,
                "next_observation_id": current.fact_id, "gap_seconds": gap,
                "median_positive_cadence_seconds": cadence,
                "cadence_gap_ratio": gap / cadence if gap is not None and cadence is not None else None,
                "center_distance": None, "larger_positive_bbox_diagonal": None,
                "center_jump_ratio": None,
            }
            if gap is not None and gap > policy.max_observation_gap_seconds:
                reasons.append("TIME_GAP")
            before = _geometry(previous_geometric) if previous_geometric is not None else None
            after = _geometry(current)
            if before is not None and after is not None:
                measurements["previous_geometric_observation_id"] = previous_geometric.fact_id
                measurements["next_geometric_observation_id"] = current.fact_id
                diagonal = max(before[2], after[2])
                distance = math.hypot(after[0] - before[0], after[1] - before[1])
                measurements["center_distance"] = distance
                if diagonal > 0:
                    ratio = distance / diagonal
                    measurements["larger_positive_bbox_diagonal"] = diagonal
                    measurements["center_jump_ratio"] = ratio
                    if ratio > policy.max_center_jump_ratio:
                        reasons.append("SPATIAL_DISCONTINUITY")
            conflicts = []
            label = _category(current)
            if label is not None and current.independence_root_ids:
                for earlier in current_episode:
                    earlier_label = _category(earlier)
                    if (earlier_label is not None and earlier.independence_root_ids
                            and earlier.source_family != current.source_family
                            and set(earlier.independence_root_ids).isdisjoint(current.independence_root_ids)
                            and canonical_category(earlier_label, policy) != canonical_category(label, policy)
                            and not _same_observation_context(earlier, current)):
                        conflicts.append({
                            "previous_observation_id": earlier.fact_id,
                            "next_observation_id": current.fact_id,
                            "previous_category": canonical_category(earlier_label, policy),
                            "next_category": canonical_category(label, policy),
                            "previous_source_family": earlier.source_family.value,
                            "next_source_family": current.source_family.value,
                            "previous_root_ids": list(earlier.independence_root_ids),
                            "next_root_ids": list(current.independence_root_ids),
                        })
            measurements["independent_category_conflicts"] = conflicts
            if conflicts:
                reasons.append("INDEPENDENT_CATEGORY_CONFLICT")
            if (gap is not None and cadence is not None
                    and gap > policy.long_empty_gap_cadence_multiplier * cadence):
                reasons.append("LONG_EMPTY_GAP")
            if reasons:
                episodes.append(_episode(
                    current_episode, video_id=video_id, raw_track_id=raw_track_id, policy=policy,
                    split_reasons=start_reasons, split_measurements=start_measurements, cadence=cadence,
                ))
                current_episode = []
                previous_geometric = None
                previous_temporal = None
                start_reasons, start_measurements = tuple(reasons), measurements
            current_episode.append(current)
            if after is not None:
                previous_geometric = current
            if is_continuity:
                previous_temporal = current
        episodes.append(_episode(
            current_episode, video_id=video_id, raw_track_id=raw_track_id, policy=policy,
            split_reasons=start_reasons, split_measurements=start_measurements, cadence=cadence,
        ))
        episodes[group_episode_start:] = _attach_geometryless_derivations(
            episodes[group_episode_start:], group,
        )
    return tuple(episodes)


def episode_for_fact(fact: EvidenceFact, episodes: Sequence[TrackEpisode]) -> str | None:
    """Resolve membership by observation ID and the original video/track reference."""
    for episode in episodes:
        if (episode.video_id == fact.video_id and episode.raw_track_id == _raw_track_id(fact)
                and fact.fact_id in episode.ordered_observation_ids):
            return episode.episode_id
    return None


def normalize_and_fuse(
    facts: Sequence[EvidenceFact], *, video_id: str,
    policy: NormalizationPolicy = DEFAULT_NORMALIZATION_POLICY,
) -> NormalizationResult:
    """Normalize one video's lineage, episodes, and source-aware identity."""
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a nonempty string")
    for fact in facts:
        if fact.video_id != video_id or (
            "video_id" in fact.provenance and fact.provenance["video_id"] != video_id
        ):
            raise ValueError("normalization facts must have one consistent video_id")
    # References are outputs of this boundary, never inputs to a later pass.
    inputs = tuple(replace(fact, provenance={key: deepcopy(value)
                                            for key, value in fact.provenance.items()
                                            if key not in _IDENTITY_REFERENCE_KEYS}) for fact in facts)
    normalized, failures = _resolve_lineage(inputs)
    episodes = build_track_episodes(normalized, policy=policy)
    episode_ids = {fact_id: episode.episode_id for episode in episodes
                   for fact_id in episode.ordered_observation_ids}
    episode_track_keys = {(episode.video_id, episode.raw_track_id) for episode in episodes}

    def assigned_episode(fact: EvidenceFact) -> str | None:
        if fact.fact_id in episode_ids:
            return episode_ids[fact.fact_id]
        raw_track_id = _raw_track_id(fact)
        if (_derivation_requires_observed_episode(fact) and raw_track_id is not None
                and (fact.video_id, raw_track_id) in episode_track_keys):
            # When this normalization input contains observed episodes for the
            # target, their lineage assignment is authoritative. A stale or
            # ambiguous preassigned ID cannot override them.
            return None
        return _preassigned_observed_episode(fact, policy)

    normalized = tuple(replace(
        fact, episode_id=assigned_episode(fact),
    ) for fact in normalized)
    context_by_fact = {fact.fact_id: _observation_context(fact) for fact in normalized}
    identity_context_by_fact = {
        fact.fact_id: (
            context_by_fact[fact.fact_id]
            if fact.episode_id is None else json.dumps(
                (fact.video_id, fact.provenance.get("scene_id"),
                 fact.provenance.get("scope_id")),
                separators=(",", ":"),
            )
        )
        for fact in normalized
    }
    identity_groups: dict[tuple[str, str | None, str], list[IdentityEvidence]] = {}
    for item in identity_evidence_from_facts(normalized):
        context = identity_context_by_fact[item.evidence_id]
        identity_groups.setdefault((item.local_target_id, item.episode_id, context), []).append(item)
    assessments = tuple(
        assess_identity(items, episode_id=episode_id, local_target_id=target_id, policy=policy)
        for (target_id, episode_id, _), items in sorted(
            identity_groups.items(), key=lambda entry: (entry[0][0], entry[0][1] or "", entry[0][2]))
    )
    assessment_by_fact = {item.evidence_id: assessment for assessment in assessments
                          for item in assessment.observed_labels}
    normalized = tuple(
        replace(fact, provenance={**fact.provenance,
                "identity_assessment_id": _identity_assessment_id(
                    assessment_by_fact[fact.fact_id], policy, video_id,
                    observation_context=identity_context_by_fact[fact.fact_id]),
                "identity_status": assessment_by_fact[fact.fact_id].status.value})
        if fact.fact_id in assessment_by_fact else fact for fact in normalized
    )
    groups: dict[str, dict[str, set[str]]] = {}
    for fact in normalized:
        if not _has_identity_label(fact):
            continue
        for root in fact.independence_root_ids:
            group = groups.setdefault(root, {"source_families": set(), "fact_ids": set()})
            group["source_families"].add(fact.source_family.value)
            group["fact_ids"].add(fact.fact_id)
    trace = {
        "policy_version": policy.policy_version,
        "policy": json.loads(json.dumps(asdict(policy))),
        "input_fact_count": len(facts),
        "output_fact_count": len(normalized),
        "lineage_failures": failures,
        "track_episodes": json.loads(json.dumps([asdict(episode) for episode in episodes])),
        "identity_assessments": json.loads(json.dumps([
            {"assessment_id": _identity_assessment_id(
                assessment, policy, video_id,
                observation_context=identity_context_by_fact[
                    assessment.observed_labels[0].evidence_id
                ]),
             **asdict(assessment)}
            for assessment in assessments
        ])),
        "independent_identity_source_count": len(groups),
        "independent_identity_source_groups": [
            {"root_id": root, "source_families": sorted(groups[root]["source_families"]),
             "fact_ids": sorted(groups[root]["fact_ids"])}
            for root in sorted(groups)
        ],
    }
    return NormalizationResult(normalized, episodes, assessments, trace)


_COMPONENT_ROWS = {
    RequirementType.EXISTENCE: "existence",
    RequirementType.TARGET_ATTRIBUTE: "attribute",
    RequirementType.COUNT: "count",
    RequirementType.SPATIAL_RELATION: "geometry",
    RequirementType.SCENE_REGION: "region",
    RequirementType.MOTION: "motion",
    RequirementType.ENTRY_EXIT: "entry_exit",
    RequirementType.INTERACTION: "interaction",
    RequirementType.BEFORE_STATE: "before_state",
    RequirementType.AFTER_STATE: "after_state",
    RequirementType.STATE_CHANGE: "state_change",
    RequirementType.COMPARISON: "comparison",
    RequirementType.EVENT: "event",
}
_TARGET_OBSERVATIONS = {"visual_object", "object", "target", "target_observation"}
_BEFORE_STATES = {"before_state", "scene_state_before"}
_AFTER_STATES = {"after_state", "scene_state_after"}
_ENTRY_EVENTS = {"entry", "exit", "entry_exit", "entry_exit_event"}
_QUEUE_OBSERVATIONS = {"explicit_queue_like_observation", "queue_like_observation"}


def _value(fact: EvidenceFact) -> dict[str, Any]:
    return fact.value if isinstance(fact.value, dict) else {}


def _predicate(fact: EvidenceFact) -> str:
    value = _value(fact)
    predicate = value.get("predicate") or value.get("event_type") or value.get("motion")
    if not isinstance(predicate, str):
        predicate = fact.fact_type
    return {"entry": "enter", "image_near": "near"}.get(predicate.casefold(), predicate.casefold())


def _scope(fact: EvidenceFact) -> tuple[Any, Any]:
    return fact.provenance.get("scene_id"), fact.provenance.get("scope_id")


def _has_scope(fact: EvidenceFact) -> bool:
    return any(isinstance(value, str) and value.strip() for value in _scope(fact))


def _scope_mismatch(left: EvidenceFact, right: EvidenceFact) -> str | None:
    left_scene, left_scope = _scope(left)
    right_scene, right_scope = _scope(right)
    if left_scene != right_scene:
        return "SCENE_MISMATCH"
    if left_scope != right_scope:
        return "SCOPE_MISMATCH"
    return None


def _endpoint(fact: EvidenceFact, role: RoleReference | None, generic: str) -> str | None:
    if role is None:
        return fact.role_bindings.get(generic)
    return fact.role_bindings.get(role.role_id) or fact.role_bindings.get(generic)


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)


def _anchor_time(fact: EvidenceFact) -> Any:
    return _value(fact).get("anchor_time", fact.provenance.get("anchor_time"))


def _candidate_gate(fact: EvidenceFact, normalized: EvidenceFact | None) -> str | None:
    """Only the boundary's immutable observation can be promoted by the matrix."""
    if normalized is None:
        return "FACT_NOT_NORMALIZED"
    if fact.video_id != normalized.video_id:
        return "VIDEO_MISMATCH"
    if (fact.provenance.get("video_id") != fact.video_id
            or not fact.source_evidence_ids):
        return "INVALID_PROVENANCE"
    if fact.time_span != normalized.time_span:
        return "TIME_MISMATCH"
    if reason := _scope_mismatch(fact, normalized):
        return reason
    if fact.role_bindings != normalized.role_bindings:
        return "ROLE_MISMATCH"
    if fact.episode_id is not None and fact.episode_id != normalized.episode_id:
        return "EPISODE_MISMATCH"
    if not normalized.independence_root_ids:
        return "UNTRACEABLE_LINEAGE"
    raw_provenance = {key: value for key, value in fact.provenance.items()
                      if key not in _IDENTITY_REFERENCE_KEYS}
    expected_provenance = {key: value for key, value in normalized.provenance.items()
                           if key not in _IDENTITY_REFERENCE_KEYS}
    if (fact.fact_type != normalized.fact_type or fact.value != normalized.value
            or fact.frame_ids != normalized.frame_ids
            or fact.source_evidence_ids != normalized.source_evidence_ids
            or fact.derived_from != normalized.derived_from
            or fact.derivation_rule != normalized.derivation_rule
            or fact.source_kind != normalized.source_kind
            or raw_provenance != expected_provenance):
        return "NORMALIZED_FACT_MISMATCH"
    return None


def _ambiguous_role(fact: EvidenceFact, role: RoleReference | RegionReference | None, generic: str) -> bool:
    return bool(role is not None and fact.role_bindings.get(role.role_id)
                and fact.role_bindings.get(generic)
                and fact.role_bindings[role.role_id] != fact.role_bindings[generic])


def _role_gate(requirement: QuestionRequirement, fact: EvidenceFact) -> str | None:
    roles = [(requirement.subject, "actor" if requirement.requirement_type is RequirementType.INTERACTION else "subject"),
             (requirement.reference_object, "patient" if requirement.requirement_type is RequirementType.INTERACTION else "reference"),
             (requirement.comparison_target, "comparison")]
    if any(role is not None and not _endpoint(fact, role, generic) for role, generic in roles):
        return "ROLE_MISMATCH"
    if (any(_ambiguous_role(fact, role, generic) for role, generic in roles)
            or _ambiguous_role(fact, requirement.reference_region, "region")):
        return "ROLE_MISMATCH"
    if any(role is not None and role.coreference_target
           and _endpoint(fact, role, generic) != role.coreference_target for role, generic in roles):
        return "ROLE_MISMATCH"
    return None


def _state_gate(requirement: QuestionRequirement, fact: EvidenceFact) -> str | None:
    value = _value(fact)
    if not (any(value.get(key) for key in ("state", "object_counts", "motion_summary", "queue_like_observation"))
            or _finite_number(value.get("traffic_density_proxy"))):
        return "OBSERVED_STATE_REQUIRED"
    if not _has_scope(fact):
        return "SCENE_SCOPE_REQUIRED"
    temporal = requirement.temporal_relation
    if ("anchor_role_id" in value and "anchor_role_id" in fact.provenance
            and value["anchor_role_id"] != fact.provenance["anchor_role_id"]):
        return "ANCHOR_MISMATCH"
    anchor_role = value.get("anchor_role_id") or fact.provenance.get("anchor_role_id")
    if temporal is None or anchor_role != temporal.anchor_role_id:
        return "ANCHOR_MISMATCH"
    directions = {
        RequirementType.BEFORE_STATE: {"before", "before_after"},
        RequirementType.AFTER_STATE: {"after", "before_after"},
        RequirementType.STATE_CHANGE: {"before_after"},
    }
    if temporal.direction not in directions[requirement.requirement_type]:
        return "TIME_MISMATCH"
    predicate = (requirement.predicate or "").casefold()
    if predicate and predicate not in {"state", "change"} and value.get("state") != predicate:
        return "PREDICATE_MISMATCH"
    anchor = _anchor_time(fact)
    if ("anchor_time" in value and "anchor_time" in fact.provenance
            and value["anchor_time"] != fact.provenance["anchor_time"]):
        return "ANCHOR_MISMATCH"
    if not _finite_number(anchor):
        return "ANCHOR_MISMATCH"
    if ((fact.fact_type in _BEFORE_STATES and fact.time_span.end > anchor)
            or (fact.fact_type in _AFTER_STATES and fact.time_span.start < anchor)):
        return "TIME_MISMATCH"
    if value.get("causality_claimed") or fact.provenance.get("causality_claimed"):
        return "TEMPORAL_ORDER_NOT_CAUSALITY"
    if requirement.subject and fact.episode_id is None:
        return "EPISODE_REQUIRED"
    return _role_gate(requirement, fact)


def _single_compatibility(
    requirement: QuestionRequirement, fact: EvidenceFact, normalization: NormalizationResult,
) -> str | None:
    """Return the first hard rejection, or None for an admissible typed fact."""
    kind, value = requirement.requirement_type, _value(fact)
    predicate = (requirement.predicate or "").casefold()
    actual = _predicate(fact)
    if actual == "enter":
        if kind is RequirementType.STATE_CHANGE:
            return "ENTER_NOT_STATE_CHANGE"
        if predicate == "congestion":
            return "ENTER_NOT_CONGESTION"
        if predicate == "transition":
            return "ENTER_NOT_TRANSITION"
    if kind in {RequirementType.EXISTENCE, RequirementType.TARGET_ATTRIBUTE}:
        if fact.fact_type not in _TARGET_OBSERVATIONS:
            return "EVIDENCE_TYPE_MISMATCH"
        if not fact.role_bindings or not fact.frame_ids:
            return "ROLE_MISMATCH"
        if kind is RequirementType.TARGET_ATTRIBUTE:
            visible = {str(value.get(key, "")).casefold() for key in ("color", "shape", "size")}
            attributes = value.get("attributes", [])
            if isinstance(attributes, (list, tuple)):
                visible.update(str(attribute).casefold() for attribute in attributes)
            if not requirement.attributes or any(attribute.casefold() not in visible for attribute in requirement.attributes):
                return "VISIBLE_ATTRIBUTE_REQUIRED"
    elif kind is RequirementType.COUNT:
        if fact.fact_type not in {"object_count", "object_count_observation", "scene_state"}:
            return "EVIDENCE_TYPE_MISMATCH"
        if value.get("complete_scope") is not True or not _has_scope(fact):
            return "COMPLETE_COUNT_SCOPE_REQUIRED"
        if not value.get("counting_rule") or (predicate and value["counting_rule"] != predicate):
            return "COUNTING_RULE_MISMATCH"
        count = value.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return "EXPLICIT_COUNT_REQUIRED"
    elif kind is RequirementType.SPATIAL_RELATION:
        if not predicate or actual != predicate:
            return "PREDICATE_MISMATCH"
        allowed_types = {predicate, "same_frame_relation"}
        if predicate == "near":
            allowed_types.add("image_near")
        if fact.fact_type not in allowed_types:
            return "EVIDENCE_TYPE_MISMATCH"
        if len(fact.frame_ids) != 1:
            return "SAME_FRAME_REQUIRED"
        endpoints = [_endpoint(fact, requirement.subject, "subject"),
                     _endpoint(fact, requirement.reference_object, "reference")]
        if predicate == "between":
            endpoints.append(fact.role_bindings.get("reference_2"))
        if not all(endpoints) or len(set(endpoints)) != len(endpoints):
            return "DISTINCT_ENDPOINTS_REQUIRED"
        if predicate == "between" and fact.provenance.get("predicate_test") != "between":
            return "BETWEEN_REQUIRES_EXPLICIT_GEOMETRY"
    elif kind is RequirementType.SCENE_REGION:
        if fact.fact_type not in {"located_in_region", "scene_region_observation"}:
            return "EVIDENCE_TYPE_MISMATCH"
        if len(fact.frame_ids) != 1:
            return "SAME_FRAME_REQUIRED"
        if not value.get("region_definition_version"):
            return "REGION_DEFINITION_REQUIRED"
        region = requirement.reference_region
        endpoint = fact.role_bindings.get(region.role_id if region else "region")
        if not endpoint or (region and region.label and value.get("region") != region.label):
            return "REGION_MISMATCH"
    elif kind is RequirementType.MOTION:
        if fact.fact_type not in {"motion", "track_motion", "motion_observation"}:
            return "EVIDENCE_TYPE_MISMATCH"
        if not value.get("motion") or (predicate and actual != predicate):
            return "PREDICATE_MISMATCH"
        if fact.episode_id is None:
            return "EPISODE_REQUIRED"
        observation_ids = value.get("observation_ids") or fact.derived_from
        observations = {item.fact_id: item for item in normalization.facts}
        if not isinstance(observation_ids, (list, tuple)) or len(set(observation_ids)) < 2:
            return "DISTINCT_OBSERVATIONS_REQUIRED"
        points = [observations.get(identifier) for identifier in observation_ids]
        if any(point is None or not point.independence_root_ids for point in points):
            return "UNTRACEABLE_LINEAGE"
        if any(point.episode_id != fact.episode_id for point in points):
            return "EPISODE_MISMATCH"
        if len({point.time_span.start for point in points}) < 2:
            return "DISTINCT_OBSERVATIONS_REQUIRED"
        if any(set(left.independence_root_ids) & set(right.independence_root_ids)
               for index, left in enumerate(points) for right in points[index + 1:]):
            return "DISTINCT_OBSERVATIONS_REQUIRED"
    elif kind is RequirementType.ENTRY_EXIT:
        if fact.fact_type not in _ENTRY_EVENTS:
            return "EVIDENCE_TYPE_MISMATCH"
        if not predicate or actual != predicate or actual not in {"enter", "exit"}:
            return "PREDICATE_MISMATCH"
        if not value.get("boundary_id"):
            return "BOUNDARY_REQUIRED"
        anchor = value.get("anchor_time")
        if not _finite_number(anchor) or not fact.time_span.start <= anchor <= fact.time_span.end:
            return "TIME_MISMATCH"
        if fact.episode_id is None:
            return "EPISODE_REQUIRED"
        region = requirement.reference_region
        if region and fact.role_bindings.get(region.role_id) != region.label:
            return "REGION_MISMATCH"
    elif kind is RequirementType.INTERACTION:
        if fact.fact_type not in {"interaction", "interaction_observation"}:
            return "EVIDENCE_TYPE_MISMATCH"
        if not predicate or actual != predicate:
            return "PREDICATE_MISMATCH"
        actor, patient = _endpoint(fact, requirement.subject, "actor"), _endpoint(fact, requirement.reference_object, "patient")
        if not actor or not patient or actor == patient:
            return "ROLE_MISMATCH"
        if len(fact.frame_ids) != 1:
            return "SAME_FRAME_REQUIRED"
    elif kind in {RequirementType.BEFORE_STATE, RequirementType.AFTER_STATE}:
        types = _BEFORE_STATES if kind is RequirementType.BEFORE_STATE else _AFTER_STATES
        if fact.fact_type not in types:
            return "EVIDENCE_TYPE_MISMATCH"
        return _state_gate(requirement, fact)
    elif kind is RequirementType.COMPARISON:
        if fact.fact_type != "comparison":
            return "EVIDENCE_TYPE_MISMATCH"
        if not predicate or actual != predicate:
            return "PREDICATE_MISMATCH"
        if not value.get("measure") or not value.get("measurement_semantics"):
            return "MEASUREMENT_MISMATCH"
        left, right = _endpoint(fact, requirement.subject, "subject"), _endpoint(fact, requirement.comparison_target, "comparison")
        if not left or not right or left == right:
            return "ROLE_MISMATCH"
    elif kind is RequirementType.EVENT:
        if fact.fact_type in _QUEUE_OBSERVATIONS and predicate != "queue_like":
            return "PREDICATE_MISMATCH"
        if fact.fact_type not in {"event", "visual_event", "event_observation", *_QUEUE_OBSERVATIONS}:
            return "EVIDENCE_TYPE_MISMATCH"
        if not predicate or actual != predicate:
            return "PREDICATE_MISMATCH"
        if value.get("causality_claimed") or fact.provenance.get("causality_claimed"):
            return "TEMPORAL_ORDER_NOT_CAUSALITY"
        if not _has_scope(fact):
            return "SCENE_SCOPE_REQUIRED"
    else:
        return "EVIDENCE_TYPE_MISMATCH"
    return _role_gate(requirement, fact)


def _pair_compatibility(requirement: QuestionRequirement, left: EvidenceFact, right: EvidenceFact) -> str | None:
    if left.video_id != right.video_id:
        return "VIDEO_MISMATCH"
    if not _has_scope(left) or not _has_scope(right):
        return "SCENE_SCOPE_REQUIRED"
    if reason := _scope_mismatch(left, right):
        return reason
    if requirement.subject and requirement.requirement_type is RequirementType.STATE_CHANGE:
        if left.episode_id is None or right.episode_id is None:
            return "EPISODE_REQUIRED"
        if (left.episode_id != right.episode_id
                or _endpoint(left, requirement.subject, "subject") != _endpoint(right, requirement.subject, "subject")):
            return "EPISODE_MISMATCH"
    if set(left.independence_root_ids) & set(right.independence_root_ids) or left.fact_id == right.fact_id:
        return "DISTINCT_OBSERVATIONS_REQUIRED"
    if requirement.requirement_type is RequirementType.STATE_CHANGE:
        for item in (left, right):
            reason = _state_gate(requirement, item)
            if reason:
                return reason
        if _anchor_time(left) != _anchor_time(right):
            return "ANCHOR_MISMATCH"
        if left.time_span.end >= right.time_span.start:
            return "TIME_MISMATCH"
    else:
        subject = _endpoint(left, requirement.subject, "subject")
        comparison = _endpoint(right, requirement.comparison_target, "comparison")
        if not subject or not comparison or subject == comparison:
            return "ROLE_MISMATCH"
        for role, item, generic in ((requirement.subject, left, "subject"),
                                    (requirement.comparison_target, right, "comparison")):
            if _ambiguous_role(item, role, generic):
                return "ROLE_MISMATCH"
            if role and role.coreference_target and _endpoint(item, role, generic) != role.coreference_target:
                return "ROLE_MISMATCH"
        before, after = _value(left), _value(right)
        if (not before.get("measure") or before.get("measure") != after.get("measure")
                or not before.get("measurement_semantics")
                or before.get("measurement_semantics") != after.get("measurement_semantics")):
            return "MEASUREMENT_MISMATCH"
        if left.time_span != right.time_span or left.frame_ids != right.frame_ids:
            return "TIME_MISMATCH"
        measurements = before.get("measurement"), after.get("measurement")
        if not all(_finite_number(value) for value in measurements):
            return "MEASUREMENT_MISMATCH"
        predicates = {"greater": measurements[0] > measurements[1],
                      "less": measurements[0] < measurements[1], "equal": measurements[0] == measurements[1]}
        if not predicates.get((requirement.predicate or "").casefold(), False):
            return "PREDICATE_MISMATCH"
    return None


def _decision(component_id: str, level: SupportLevel, supporting: Sequence[EvidenceFact],
              observations: Sequence[EvidenceFact], reasons: tuple[str, ...],
              refs: Sequence[str] = (), inspected: Sequence[EvidenceFact] = ()) -> SupportComponentDecision:
    support_ids = tuple(dict.fromkeys(fact.fact_id for fact in supporting))
    metadata = tuple(supporting) + tuple(inspected)
    return SupportComponentDecision(
        component_id, level, support_ids,
        tuple(dict.fromkeys(fact.fact_id for fact in observations if fact.fact_id not in support_ids)),
        tuple(sorted(set(refs))), tuple(sorted({fact.episode_id for fact in metadata if fact.episode_id})),
        tuple(sorted({root for fact in metadata for root in fact.independence_root_ids})), reasons,
    )


def _same_local_observation(left: EvidenceFact, right: EvidenceFact) -> bool:
    return (left.video_id == right.video_id
            and left.provenance.get("scene_id") == right.provenance.get("scene_id")
            and left.provenance.get("scope_id") == right.provenance.get("scope_id")
            and bool(set(left.frame_ids) & set(right.frame_ids))
            and left.time_span.start <= right.time_span.end
            and right.time_span.start <= left.time_span.end)


def _identity_at_endpoint(normalization: NormalizationResult, fact: EvidenceFact,
                          target_id: str) -> IdentityAssessment | None:
    by_id = {item.fact_id: item for item in normalization.facts}

    def matching_context(assessment: IdentityAssessment | None) -> IdentityAssessment | None:
        if (assessment is None or not assessment.observed_labels
                or any(item.evidence_id not in by_id
                       or _scope(fact) != _scope(by_id[item.evidence_id])
                       for item in assessment.observed_labels)):
            return None
        return assessment

    if fact.episode_id is not None:
        assessment = matching_context(identity_for_target(
            normalization, target_id, fact.episode_id,
        ))
        if assessment is not None:
            return assessment
    assessments = [assessment for assessment in normalization.identity_assessments
                   if assessment.local_target_id == target_id and assessment.episode_id is None
                   and assessment.observed_labels
                   and all(item.evidence_id in by_id
                           and _same_local_observation(by_id[item.evidence_id], fact)
                           for item in assessment.observed_labels)]
    # A same-frame relation does not have the subject's raw track reference.
    # Resolve its endpoint from the existing observed episode membership only;
    # raw IDs or labels in another frame/scope cannot supply the missing link.
    episodes = {item.episode_id for item in normalization.facts if item.episode_id is not None
                and _local_target_id(item) == target_id and _has_identity_label(item)
                and _same_local_observation(item, fact)}
    if len(episodes) == 1:
        assessment = matching_context(identity_for_target(
            normalization, target_id, next(iter(episodes)),
        ))
        if assessment is not None:
            assessments.append(assessment)
    elif episodes:
        return None
    return assessments[0] if len(assessments) == 1 else None


def compatible_components(
    requirement: QuestionRequirement, facts: Sequence[EvidenceFact], normalization: NormalizationResult,
) -> tuple[SupportComponentDecision, ...]:
    """Assess all 13 typed rows; requirement-level composition belongs to callers.

    Each result starts with an observation/relation component. Requested category
    roles add separate identity components, so retained geometry cannot silently
    promote a category-qualified claim. Scores never participate in hard gates.
    """
    by_id = {fact.fact_id: fact for fact in normalization.facts}
    admissible, failures = [], []
    for fact in facts:
        normalized = by_id.get(fact.fact_id)
        reason = _candidate_gate(fact, normalized)
        if reason:
            failures.append(reason)
        else:
            admissible.append(normalized)
    kind = requirement.requirement_type
    supporting, rejections = [], []
    if kind is RequirementType.STATE_CHANGE:
        before = [fact for fact in admissible if fact.fact_type in _BEFORE_STATES]
        after = [fact for fact in admissible if fact.fact_type in _AFTER_STATES]
        for left in before:
            for right in after:
                reason = _pair_compatibility(requirement, left, right)
                if reason:
                    rejections.append(reason)
                else:
                    supporting.extend((left, right))
        if not before or not after:
            rejections.append("ENTER_NOT_STATE_CHANGE" if any(_predicate(fact) == "enter" for fact in admissible)
                              else "BEFORE_AFTER_REQUIRED")
    else:
        for fact in admissible:
            reason = _single_compatibility(requirement, fact, normalization)
            if reason:
                rejections.append(reason)
            else:
                supporting.append(fact)
        if kind is RequirementType.COMPARISON:
            lefts = [fact for fact in admissible if fact.fact_type == "measurement"
                     and _endpoint(fact, requirement.subject, "subject")]
            rights = [fact for fact in admissible if fact.fact_type == "measurement"
                      and _endpoint(fact, requirement.comparison_target, "comparison")]
            pair_rejections = []
            for left in lefts:
                for right in rights:
                    reason = _pair_compatibility(requirement, left, right)
                    if reason:
                        pair_rejections.append(reason)
                    else:
                        supporting.extend((left, right))
            if pair_rejections:
                rejections = pair_rejections
    primary = _decision(
        f"{requirement.requirement_id}:{_COMPONENT_ROWS[kind]}",
        SupportLevel.FULL_SUPPORT if supporting else SupportLevel.NO_SUPPORT,
        supporting, facts, ("COMPATIBLE_EVIDENCE",) if supporting else
        tuple(sorted(set(failures or [reason for reason in rejections if reason != "EVIDENCE_TYPE_MISMATCH"]
                         or rejections or ["NO_COMPATIBLE_EVIDENCE"]))),
        inspected=admissible if not supporting else (),
    )
    if (supporting and requirement.subject is not None and kind in {
            RequirementType.BEFORE_STATE, RequirementType.AFTER_STATE, RequirementType.STATE_CHANGE,
    }):
        # Reporting certificate only: the existing matrix has already checked
        # this subject's episode constraint; scene-level rows have no such gate.
        primary = replace(primary, rationale_codes=primary.rationale_codes + ("TARGET_BOUND_COMPONENT",))
    decisions = [primary]
    policy_data = normalization.trace.get("policy", {})
    policy = NormalizationPolicy(**{**policy_data, "category_aliases": tuple(tuple(alias) for alias in policy_data.get("category_aliases", ()))})
    roles = [(requirement.subject, "actor" if kind is RequirementType.INTERACTION else "subject"),
             (requirement.reference_object, "patient" if kind is RequirementType.INTERACTION else "reference"),
             (requirement.comparison_target, "comparison")]
    for role, generic in roles:
        if role is None or not role.category:
            continue
        assessments = []
        missing_assessment = False
        for fact in supporting:
            endpoints = [_endpoint(fact, role, generic)]
            if (kind is RequirementType.SPATIAL_RELATION and (requirement.predicate or "").casefold() == "between"
                    and generic == "reference"):
                endpoints.append(fact.role_bindings.get("reference_2"))
            for endpoint in endpoints:
                if endpoint is None:
                    continue
                assessment = _identity_at_endpoint(normalization, fact, endpoint)
                if assessment and assessment not in assessments:
                    assessments.append(assessment)
                if assessment is None:
                    missing_assessment = True
        refs = [by_id[item.evidence_id].provenance["identity_assessment_id"]
                for assessment in assessments for item in assessment.observed_labels
                if item.evidence_id in by_id and "identity_assessment_id" in by_id[item.evidence_id].provenance]
        category = canonical_category(role.category, policy)
        if not supporting:
            level, reasons = SupportLevel.NO_SUPPORT, ("NO_COMPATIBLE_EVIDENCE",)
        elif any(item.status is IdentityStatus.CONFLICT for item in assessments):
            level, reasons = SupportLevel.NO_SUPPORT, ("IDENTITY_CONFLICT",)
        elif any(item.compatible_category is not None and item.compatible_category != category for item in assessments):
            level, reasons = SupportLevel.NO_SUPPORT, ("CATEGORY_MISMATCH",)
        elif missing_assessment or not assessments or any(item.status is IdentityStatus.UNVERIFIED for item in assessments):
            level, reasons = SupportLevel.CONDITIONAL_SUPPORT, ("IDENTITY_UNVERIFIED",)
        else:
            level, reasons = SupportLevel.FULL_SUPPORT, ("IDENTITY_CONFIRMED",)
        decisions.append(_decision(
            f"{requirement.requirement_id}:identity:{role.role_id}", level,
            supporting if level is not SupportLevel.NO_SUPPORT else (), facts, reasons, refs,
            inspected=[by_id[item.evidence_id] for assessment in assessments for item in assessment.observed_labels
                       if item.evidence_id in by_id] + (supporting if level is SupportLevel.NO_SUPPORT else []),
        ))
    return tuple(decisions)


def assess_requirement_support(
    requirement: QuestionRequirement, facts: Sequence[EvidenceFact], normalization: NormalizationResult,
) -> RequirementSupportDecision:
    """Compose the matrix's mandatory components without reassessing evidence."""
    components = compatible_components(requirement, facts, normalization)
    if any(component.level is SupportLevel.NO_SUPPORT for component in components):
        level = SupportLevel.NO_SUPPORT
    elif any(component.level is SupportLevel.CONDITIONAL_SUPPORT for component in components):
        level = SupportLevel.CONDITIONAL_SUPPORT
    else:
        level = SupportLevel.FULL_SUPPORT
    component_fact_ids = tuple(dict.fromkeys(
        identifier for component in components for identifier in component.fact_ids
    ))
    supporting = component_fact_ids if level is not SupportLevel.NO_SUPPORT else ()
    observations = set(identifier for component in components for identifier in component.observation_only_fact_ids)
    observations.update(component_fact_ids)
    observations.difference_update(supporting)
    return RequirementSupportDecision(
        requirement_id=requirement.requirement_id, level=level, supporting_fact_ids=supporting,
        observation_only_fact_ids=tuple(dict.fromkeys(fact.fact_id for fact in facts if fact.fact_id in observations)),
        component_decisions=components,
        identity_assessment_refs=tuple(sorted({identifier for component in components
                                              for identifier in component.identity_assessment_refs})),
        provenance_root_ids=tuple(sorted({identifier for component in components
                                         for identifier in component.provenance_root_ids})),
        rationale_codes=tuple(dict.fromkeys(reason for component in components for reason in component.rationale_codes)),
    )


def audit_support_safety(decisions: Sequence[RequirementSupportDecision]) -> dict[str, int]:
    """Diagnose impossible stored promotions without reassessing or changing evidence.

    A full typed component certifies the matrix gates already composed upstream.
    In particular, state_change certifies before/after, order, scope and independent
    observations. Identity certificates are separate where categories are requested.
    Episode checks remain local to continuity components, not relation endpoints.
    """
    audit = {
        "IDENTITY_OVER_SUPPORT": 0,
        "TYPING_MISS": 0,
        "CROSS_EPISODE_IDENTITY_MERGE": 0,
        "UNSUPPORTED_STATE_CHANGE": 0,
    }
    for decision in decisions:
        if decision.level is not SupportLevel.FULL_SUPPORT:
            continue
        identities = [component for component in decision.component_decisions
                      if ":identity:" in component.component_id]
        typed = [component for component in decision.component_decisions
                 if component.component_id.rsplit(":", 1)[-1] in _COMPONENT_ROWS.values()
                 and ":identity:" not in component.component_id]
        reasons = set(decision.rationale_codes)
        reasons.update(reason for component in decision.component_decisions for reason in component.rationale_codes)
        identity_failure = any(
            component.level is not SupportLevel.FULL_SUPPORT
            or "IDENTITY_CONFIRMED" not in component.rationale_codes
            for component in identities
        ) or bool(reasons & {"IDENTITY_UNVERIFIED", "IDENTITY_CONFLICT", "CATEGORY_MISMATCH"})
        audit["IDENTITY_OVER_SUPPORT"] += int(identity_failure)
        audit["TYPING_MISS"] += int(not any(
            component.level is SupportLevel.FULL_SUPPORT for component in typed
        ))
        continuity = [component for component in typed
                      if component.component_id.endswith((":motion", ":entry_exit"))
                      or "TARGET_BOUND_COMPONENT" in component.rationale_codes]
        audit["CROSS_EPISODE_IDENTITY_MERGE"] += int(any(
            len(set(component.episode_ids)) > 1 for component in continuity
        ) or bool(reasons & {"EPISODE_MISMATCH", "CROSS_EPISODE_TARGET"}))
        changes = [component for component in typed if component.component_id.endswith(":state_change")]
        state_rejections = {
            "BEFORE_AFTER_REQUIRED", "ENTER_NOT_STATE_CHANGE", "TIME_MISMATCH", "SCENE_MISMATCH",
            "SCOPE_MISMATCH",
            "SCENE_SCOPE_REQUIRED", "ANCHOR_MISMATCH", "EPISODE_REQUIRED", "EPISODE_MISMATCH",
            "CROSS_EPISODE_TARGET", "DISTINCT_OBSERVATIONS_REQUIRED", "OBSERVED_STATE_REQUIRED",
            "ROLE_MISMATCH", "TEMPORAL_ORDER_NOT_CAUSALITY",
        }
        audit["UNSUPPORTED_STATE_CHANGE"] += int(bool(changes) and (
            identity_failure or any(
                component.level is not SupportLevel.FULL_SUPPORT
                or "COMPATIBLE_EVIDENCE" not in component.rationale_codes
                or len(set(component.fact_ids)) < 2
                or len(set(component.provenance_root_ids)) < 2
                or bool(reasons & state_rejections)
                or ("TARGET_BOUND_COMPONENT" in component.rationale_codes
                    and len(set(component.episode_ids)) != 1)
                for component in changes
            )
        ))
    return audit
