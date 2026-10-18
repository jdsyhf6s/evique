from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

from .normalization import NormalizationResult, assess_requirement_support, normalize_and_fuse
from .role_consistency import _check_role_consistency_with_decisions, _selected_requirement_ids
from .support import _logical_support
from .schema import (
    EvidenceBundle,
    EvidenceFact,
    QuestionRequirement,
    QuestionRequirementSet,
    RequirementSupportDecision,
    SubjectBinding,
    SupportLevel,
    TimeRange,
)


@dataclass(frozen=True)
class RequirementSupport:
    requirement_id: str
    supported: bool
    fact_ids: tuple[str, ...]
    reason: str


def _supports(requirement: QuestionRequirement, fact: EvidenceFact) -> bool:
    """Legacy one-fact projection; bundle construction uses decision records."""
    if not fact.source_evidence_ids or fact.provenance.get("video_id") != fact.video_id:
        return False
    normalization = normalize_and_fuse([fact], video_id=fact.video_id)
    return assess_requirement_support(requirement, [fact], normalization).level is SupportLevel.FULL_SUPPORT


def requirement_support(requirement: QuestionRequirement, bundle: EvidenceBundle) -> RequirementSupport:
    decision = bundle.requirement_support_decisions.get(requirement.requirement_id)
    if decision is not None:
        return RequirementSupport(
            decision.requirement_id, decision.level is SupportLevel.FULL_SUPPORT,
            decision.supporting_fact_ids, ",".join(decision.rationale_codes),
        )
    if bundle.requirement_support_decisions:
        return RequirementSupport(
            requirement.requirement_id, False, (), "no_authoritative_support_decision",
        )
    mapping = bundle.provenance.get("requirement_support") or {}
    fact_ids = tuple(mapping.get(requirement.requirement_id) or ())
    return RequirementSupport(
        requirement.requirement_id, bool(fact_ids), fact_ids,
        "explicit_fact_support" if fact_ids else "no_compatible_fact",
    )


def _clusters(facts: list[EvidenceFact]) -> list[list[EvidenceFact]]:
    clusters: list[list[EvidenceFact]] = []
    for fact in sorted(facts, key=lambda item: (item.video_id, item.time_span.start, item.time_span.end, item.fact_id)):
        matching = []
        for cluster in clusters:
            if cluster[0].video_id != fact.video_id:
                continue
            shares_frame = bool(set(fact.frame_ids) & {frame for item in cluster for frame in item.frame_ids})
            overlaps = any(
                fact.time_span.start <= item.time_span.end and item.time_span.start <= fact.time_span.end
                for item in cluster
            )
            if shares_frame or overlaps:
                matching.append(cluster)
        if not matching:
            clusters.append([fact])
        else:
            merged = [fact]
            for cluster in matching:
                merged.extend(cluster)
                clusters.remove(cluster)
            clusters.append(merged)
    return clusters


def _bindings(requirements: QuestionRequirementSet, facts: list[EvidenceFact],
              decisions: dict[str, RequirementSupportDecision]) -> dict[str, SubjectBinding]:
    selected = _selected_requirement_ids(requirements, decisions, facts)
    full_ids = {
        identifier for identifier, decision in decisions.items()
        if decision.level is SupportLevel.FULL_SUPPORT
    }
    primary_ids = selected if selected is not None else full_ids
    identifiers_by_role: dict[str, list[str]] = {}

    def add_bindings(requirement, *, skip_existing_roles=False):
        decision = decisions[requirement.requirement_id]
        cited_fact_ids = set(decision.supporting_fact_ids)
        cited_fact_ids.update(
            identifier for component in decision.component_decisions for identifier in component.fact_ids
        )
        roles = ((requirement.subject, ("subject", "actor")),
                 (requirement.reference_object, ("reference", "patient")),
                 (requirement.comparison_target, ("comparison",)))
        for role, generic_keys in roles:
            if role is None or (skip_existing_roles and role.role_id in identifiers_by_role):
                continue
            identifiers = []
            for fact in facts:
                if fact.fact_id not in cited_fact_ids:
                    continue
                direct = fact.role_bindings.get(role.role_id)
                if direct:
                    identifiers.append(direct)
                identifiers.extend(fact.role_bindings[key] for key in generic_keys if fact.role_bindings.get(key))
            identifiers_by_role.setdefault(role.role_id, []).extend(identifiers)

    for requirement in requirements.requirements:
        if requirement.requirement_id in primary_ids:
            add_bindings(requirement)
    # Preserve bindings for distinct FULL optional/alternative roles without
    # allowing an unselected branch to widen a role selected by the root logic.
    for requirement in requirements.requirements:
        if requirement.requirement_id in full_ids - set(primary_ids):
            add_bindings(requirement, skip_existing_roles=True)
    return {
        role_id: SubjectBinding(role_id, tuple(dict.fromkeys(identifiers)))
        for role_id, identifiers in identifiers_by_role.items() if identifiers
    }


def _evidence_groups(facts: list[EvidenceFact]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {key: [] for key in (
        "scene_evidence", "object_evidence", "relation_evidence", "track_evidence",
        "temporal_evidence", "event_evidence",
    )}
    relation_types = {"left_of", "right_of", "above", "below", "overlap", "image_near", "located_in_region", "interaction"}
    temporal_types = {"before_state", "after_state", "scene_state_before", "scene_state_after", "state_change"}
    for fact in facts:
        if fact.fact_type in relation_types:
            group = "relation_evidence"
        elif fact.fact_type in temporal_types:
            group = "temporal_evidence"
        elif "track" in fact.fact_type or fact.fact_type == "motion":
            group = "track_evidence"
        elif "event" in fact.fact_type or fact.fact_type in {"entry", "exit", "entry_exit"}:
            group = "event_evidence"
        elif fact.fact_type in {"visual_object", "object", "target"}:
            group = "object_evidence"
        else:
            group = "scene_evidence"
        groups[group].append(fact.fact_id)
    return {key: tuple(value) for key, value in groups.items()}


def _report_payload(report) -> dict[str, Any]:
    def check(value):
        return {"consistent": value.consistent, "evidence_ids": list(value.evidence_ids), "warnings": list(value.warnings)}
    return {
        "role_checks": {key: check(value) for key, value in report.role_checks.items()},
        "relation_checks": {key: check(value) for key, value in report.relation_checks.items()},
    }


def _identity_witnesses_by_assessment(
    normalization: NormalizationResult,
) -> dict[str, frozenset[str]]:
    """Resolve each stored identity certificate to its observed fact witnesses."""
    by_id = {fact.fact_id: fact for fact in normalization.facts}
    witnesses: dict[str, set[str]] = {}
    for assessment in normalization.identity_assessments:
        evidence_ids = {item.evidence_id for item in assessment.observed_labels}
        refs = {
            by_id[identifier].provenance.get("identity_assessment_id")
            for identifier in evidence_ids if identifier in by_id
        }
        for ref in refs:
            if isinstance(ref, str) and ref:
                witnesses.setdefault(ref, set()).update(evidence_ids)
    return {ref: frozenset(identifiers) for ref, identifiers in witnesses.items()}


def _enforce_identity_witness_visibility(
    decision: RequirementSupportDecision,
    visible_fact_ids: set[str],
    witnesses_by_assessment: dict[str, frozenset[str]],
) -> RequirementSupportDecision:
    """Do not retain FULL identity support whose confirming labels are hidden."""
    components = []
    changed = False
    for component in decision.component_decisions:
        if component.level is not SupportLevel.FULL_SUPPORT or ":identity:" not in component.component_id:
            components.append(component)
            continue
        references = component.identity_assessment_refs
        witness_sets = [witnesses_by_assessment.get(ref) for ref in references]
        closed = bool(references) and all(
            witnesses is not None and witnesses and witnesses <= visible_fact_ids
            for witnesses in witness_sets
        )
        if closed:
            components.append(component)
            continue
        changed = True
        rationale = tuple(
            reason for reason in component.rationale_codes if reason != "IDENTITY_CONFIRMED"
        ) + ("IDENTITY_WITNESS_NOT_VISIBLE",)
        components.append(replace(
            component, level=SupportLevel.CONDITIONAL_SUPPORT,
            rationale_codes=tuple(dict.fromkeys(rationale)),
        ))
    if not changed:
        return decision
    components = tuple(components)
    if any(component.level is SupportLevel.NO_SUPPORT for component in components):
        level = SupportLevel.NO_SUPPORT
    elif any(component.level is SupportLevel.CONDITIONAL_SUPPORT for component in components):
        level = SupportLevel.CONDITIONAL_SUPPORT
    else:
        level = SupportLevel.FULL_SUPPORT
    supporting = tuple(dict.fromkeys(
        identifier for component in components for identifier in component.fact_ids
    )) if level is not SupportLevel.NO_SUPPORT else ()
    return replace(
        decision, level=level, supporting_fact_ids=supporting,
        component_decisions=components,
        rationale_codes=tuple(dict.fromkeys(
            reason for component in components for reason in component.rationale_codes
        )),
    )


def build_candidate_bundles(requirements: QuestionRequirementSet, facts: list[EvidenceFact], *,
                            max_bundles: int = 2,
                            normalization: NormalizationResult | None = None) -> list[EvidenceBundle]:
    if type(max_bundles) is not int or not 1 <= max_bundles <= 2:
        raise ValueError("max_bundles must be one or two")
    for fact in facts:
        if not fact.provenance or fact.provenance.get("video_id") != fact.video_id:
            raise ValueError("bundle fact provenance is missing or inconsistent")
        if not fact.source_evidence_ids:
            raise ValueError("bundle facts must cite source evidence")
    fact_ids = [fact.fact_id for fact in facts]
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("bundle fact IDs must be unique")
    if normalization is not None:
        normalized_ids = [fact.fact_id for fact in normalization.facts]
        if len(set(normalized_ids)) != len(normalized_ids) or set(normalized_ids) != set(fact_ids):
            raise ValueError("normalization fact IDs must exactly match bundle fact IDs")
        videos = {fact.video_id for fact in normalization.facts} | {fact.video_id for fact in facts}
        if len(videos) > 1:
            raise ValueError("supplied normalization must contain one consistent video_id")
        normalizations = {video_id: normalization for video_id in videos}
    else:
        normalizations = {
            video_id: normalize_and_fuse([fact for fact in facts if fact.video_id == video_id], video_id=video_id)
            for video_id in sorted({fact.video_id for fact in facts})
        }
    candidates = []
    for cluster in _clusters(facts):
        video_id = cluster[0].video_id
        if any(fact.video_id != video_id for fact in cluster):
            raise ValueError("bundle cannot mix video IDs")
        start = min(fact.time_span.start for fact in cluster)
        end = max(fact.time_span.end for fact in cluster)
        normalization_for_video = normalizations[video_id]
        visible_fact_ids = {fact.fact_id for fact in cluster}
        identity_witnesses = _identity_witnesses_by_assessment(normalization_for_video)
        decisions = {
            requirement.requirement_id: _enforce_identity_witness_visibility(
                assess_requirement_support(requirement, cluster, normalization_for_video),
                visible_fact_ids, identity_witnesses,
            )
            for requirement in requirements.requirements
        }
        supported = tuple(identifier for identifier, decision in decisions.items()
                          if decision.level is SupportLevel.FULL_SUPPORT)
        conditional = tuple(identifier for identifier, decision in decisions.items()
                            if decision.level is SupportLevel.CONDITIONAL_SUPPORT)
        no_support = tuple(identifier for identifier, decision in decisions.items()
                           if decision.level is SupportLevel.NO_SUPPORT)
        support_map = {identifier: list(decisions[identifier].supporting_fact_ids) for identifier in supported}
        _, missing = _logical_support(requirements, set(supported))
        bindings = _bindings(requirements, cluster, decisions)
        report = _check_role_consistency_with_decisions(requirements, cluster, bindings, decisions)
        fact_ids = tuple(fact.fact_id for fact in cluster)
        digest = hashlib.sha256(json.dumps([video_id, start, end, fact_ids]).encode()).hexdigest()[:16]
        serialized_cost = math.ceil(len(json.dumps([
            {"id": fact.fact_id, "type": fact.fact_type, "value": fact.value} for fact in cluster
        ], sort_keys=True)) / 4)
        evidence = _evidence_groups(cluster)
        bundle = EvidenceBundle(
            bundle_id=f"bundle:{digest}", video_id=video_id, time_window=TimeRange(start, end),
            focal_subjects=bindings, supported_requirement_ids=supported,
            requirement_support_decisions=decisions,
            conditional_requirement_ids=conditional, no_support_requirement_ids=no_support,
            missing_requirement_ids=missing, token_cost=serialized_cost,
            provenance={
                "video_id": video_id,
                "fact_refs": list(fact_ids),
                "fact_videos": {fact.fact_id: fact.video_id for fact in cluster},
                "fact_sources": {fact.fact_id: list(fact.source_evidence_ids) for fact in cluster},
                "requirement_support": support_map,
            },
            consistency=_report_payload(report), utility={}, **evidence,
        )
        # The schema's conservative default cannot see mandatory/root-OR logic.
        # Restore this builder's exact logical projection, including empty OR
        # and optional-only results, without changing authoritative decisions.
        object.__setattr__(bundle, "missing_requirement_ids", missing)
        candidates.append(bundle)
    candidates.sort(key=lambda bundle: (
        -len(bundle.supported_requirement_ids), bundle.time_window.start,
        bundle.time_window.end, bundle.video_id, bundle.bundle_id,
    ))
    return candidates[:max_bundles]
