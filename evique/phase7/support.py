from __future__ import annotations

from dataclasses import dataclass, field, replace

from .schema import (
    BooleanExpression,
    EvidenceBundle,
    EvidenceFact,
    LogicOperator,
    Optionality,
    ParseStatus,
    QuestionRequirementSet,
    RequirementSupportDecision,
    RequirementType,
    SupportLevel,
    from_dict,
)


@dataclass(frozen=True)
class BudgetStatus:
    bundle_fits: bool | None
    visible_bundle_ids: tuple[str, ...] = ()
    omitted_bundle_ids: tuple[str, ...] = ()
    actual_message_tokens: int | None = None
    estimated_message_tokens: int | None = None
    reason: str = "not_evaluated"


@dataclass(frozen=True)
class SupportAssessment:
    observations_available: bool
    mandatory_count: int
    supported_mandatory_count: int
    supported_requirement_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]
    all_mandatory_supported: bool
    parse_complete: bool
    role_consistent: bool
    temporal_consistent: bool
    provenance_complete: bool
    no_answer_support: bool
    budget_status: BudgetStatus
    complete: bool
    compatibility_status: str
    requirement_descriptions: dict[str, str]
    full_support_count: int = 0
    conditional_support_count: int = 0
    no_support_count: int = 0
    conditional_requirement_ids: tuple[str, ...] = ()
    no_support_requirement_ids: tuple[str, ...] = ()
    projection_consistent: bool = True
    _requirements: QuestionRequirementSet | None = field(default=None, repr=False)
    _required_support_present: bool = field(default=False, repr=False)


def _evaluate_mandatory_logic(expression: BooleanExpression, supported: set[str],
                              mandatory: set[str]) -> tuple[bool, tuple[str, ...]] | None:
    """Return required satisfaction/missing IDs, or None for optional-only logic."""
    children = []
    for child in expression.children:
        if isinstance(child, str):
            if child not in mandatory:
                continue
            decision = (True, ()) if child in supported else (False, (child,))
        else:
            decision = _evaluate_mandatory_logic(child, supported, mandatory)
        if decision is not None:
            children.append(decision)
    if not children:
        return None
    satisfied = all(child[0] for child in children) if expression.operator is LogicOperator.AND else any(
        child[0] for child in children
    )
    if satisfied:
        return True, ()
    return False, tuple(dict.fromkeys(identifier for _, missing in children for identifier in missing))


def _logical_support(requirements: QuestionRequirementSet,
                     supported: set[str]) -> tuple[bool, tuple[str, ...]]:
    mandatory = {item.requirement_id for item in requirements.requirements
                 if item.optionality is Optionality.MANDATORY}
    result = _evaluate_mandatory_logic(requirements.root_logic, supported, mandatory)
    return result if result is not None else (True, ())


def _nonempty_unique_ids(value) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    identifiers = tuple(value)
    if (any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        return None
    return identifiers


def _certificate_fact_ids(decision: RequirementSupportDecision) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        decision.supporting_fact_ids
        + decision.observation_only_fact_ids
        + tuple(
            fact_id for component in decision.component_decisions
            for fact_id in component.fact_ids + component.observation_only_fact_ids
        )
    ))


def project_legacy_support(bundle: EvidenceBundle) -> tuple[RequirementSupportDecision, ...]:
    """Adapt old boolean bundles only when no authoritative decisions exist."""
    if bundle.requirement_support_decisions:
        raise ValueError("legacy projection requires an absent support decision map")
    configured = bundle.provenance.get("requirement_support")
    support_map = configured if isinstance(configured, dict) else {}
    supported = set(bundle.supported_requirement_ids)
    identifiers = tuple(dict.fromkeys(bundle.supported_requirement_ids + bundle.missing_requirement_ids))
    projected = []
    for identifier in identifiers:
        citations = _nonempty_unique_ids(support_map.get(identifier)) if identifier in supported else None
        projected.append(RequirementSupportDecision(
            identifier,
            SupportLevel.FULL_SUPPORT if citations else SupportLevel.NO_SUPPORT,
            citations or (), (), (), (), (), ("LEGACY_BOOLEAN_PROJECTION",),
        ))
    return tuple(projected)


def _fact_ids(bundle: EvidenceBundle) -> tuple[str, ...]:
    return (
        bundle.scene_evidence + bundle.object_evidence + bundle.relation_evidence
        + bundle.track_evidence + bundle.temporal_evidence + bundle.event_evidence
    )


def _role_consistent(requirements: QuestionRequirementSet, bundle: EvidenceBundle,
                     supported: set[str]) -> bool:
    role_checks = bundle.consistency.get("role_checks") or {}
    required_roles = {
        role.role_id for requirement in requirements.requirements
        if requirement.requirement_id in supported
        for role in (requirement.subject, requirement.reference_object, requirement.comparison_target)
        if role is not None
    }
    if any(
        role_id not in bundle.focal_subjects
        or role_id not in role_checks
        or role_checks[role_id].get("consistent") is not True
        for role_id in required_roles
    ):
        return False
    relation_checks = bundle.consistency.get("relation_checks") or {}
    relation_requirements = {
        requirement.requirement_id for requirement in requirements.requirements
        if requirement.requirement_type in {RequirementType.SPATIAL_RELATION, RequirementType.INTERACTION}
        and requirement.requirement_id in supported
    }
    return all(
        relation_checks.get(identifier, {}).get("consistent") is True
        for identifier in relation_requirements
    )


def _temporal_consistent(requirements: QuestionRequirementSet, bundle: EvidenceBundle,
                         supported: set[str]) -> bool:
    temporal = {
        RequirementType.BEFORE_STATE, RequirementType.AFTER_STATE, RequirementType.STATE_CHANGE,
        RequirementType.ENTRY_EXIT,
    }
    has_supported_temporal = any(
        requirement.requirement_type in temporal and requirement.requirement_id in supported
        for requirement in requirements.requirements
    )
    return not has_supported_temporal or bundle.consistency.get("temporal_consistency") is True


def _provenance_complete(
    bundle: EvidenceBundle,
    decisions: dict[str, RequirementSupportDecision] | None = None,
) -> bool:
    facts = _fact_ids(bundle)
    sources = bundle.provenance.get("fact_sources")
    videos = bundle.provenance.get("fact_videos")
    return (
        bundle.provenance.get("video_id") == bundle.video_id
        and isinstance(sources, dict)
        and isinstance(videos, dict)
        and all(sources.get(fact_id) for fact_id in facts)
        and all(videos.get(fact_id) == bundle.video_id for fact_id in facts)
        and bundle.consistency.get("authoritative_visible_facts_consistent") is not False
        and _certificate_payload_complete(bundle, decisions)
    )


def _certificate_payload_complete(
    bundle: EvidenceBundle,
    decisions: dict[str, RequirementSupportDecision] | None = None,
) -> bool:
    """Validate every visible fact cited by a FULL support certificate."""
    authoritative = bool(bundle.requirement_support_decisions)
    if decisions is None:
        decisions = bundle.requirement_support_decisions or {
            decision.requirement_id: decision for decision in project_legacy_support(bundle)
        }
    full_decisions = tuple(
        decision for decision in decisions.values()
        if decision.level is SupportLevel.FULL_SUPPORT
    )
    certificates = tuple(_certificate_fact_ids(decision) for decision in full_decisions)
    if any(not fact_ids for fact_ids in certificates):
        return False
    needed = {fact_id for fact_ids in certificates for fact_id in fact_ids}
    if not full_decisions:
        return True
    sources = bundle.provenance.get("fact_sources")
    if not isinstance(sources, dict):
        return False
    configured = bundle.provenance.get("visible_facts")
    if not isinstance(configured, (list, tuple)):
        return False
    payloads = {fact_id: [] for fact_id in needed}
    for payload in configured:
        if not isinstance(payload, dict) or payload.get("fact_id") not in needed:
            continue
        payloads[payload["fact_id"]].append(payload)

    fact_fields = set(EvidenceFact.__dataclass_fields__)
    required_fields = fact_fields - {"video_id"}
    for fact_id, candidates in payloads.items():
        if not candidates:
            return False
        source_ids = _nonempty_unique_ids(sources.get(fact_id))
        if source_ids is None:
            return False
        if authoritative:
            canonical = []
            for payload in candidates:
                if not required_fields <= set(payload) or not isinstance(payload.get("provenance"), dict):
                    return False
                payload_source_ids = _nonempty_unique_ids(payload.get("source_evidence_ids"))
                if payload_source_ids is None or payload_source_ids != source_ids:
                    return False
                if (payload["provenance"].get("video_id") != bundle.video_id
                        or payload.get("video_id", bundle.video_id) != bundle.video_id):
                    return False
                value = {key: payload[key] for key in fact_fields if key in payload}
                value["video_id"] = (
                    payload.get("video_id")
                    or payload["provenance"].get("video_id")
                    or bundle.video_id
                )
                try:
                    fact = from_dict(EvidenceFact, value)
                except (AttributeError, KeyError, TypeError, ValueError):
                    return False
                if fact.fact_id != fact_id or fact.video_id != bundle.video_id:
                    return False
                canonical.append(fact)
            if any(fact != canonical[0] for fact in canonical[1:]):
                return False
            continue

        # Legacy boolean projections have no typed certificate.  Preserve them
        # only when their cited payload still carries an unambiguous source and
        # matching video provenance.
        first = candidates[0]
        if any(candidate != first for candidate in candidates[1:]):
            return False
        provenance = first.get("provenance")
        payload_source_ids = _nonempty_unique_ids(first.get("source_evidence_ids"))
        if (not isinstance(provenance, dict)
                or payload_source_ids is None or payload_source_ids != source_ids
                or provenance.get("video_id") != bundle.video_id
                or first.get("video_id", bundle.video_id) != bundle.video_id):
            return False
    return True


def _description(requirement) -> str:
    pieces = [requirement.requirement_type.value]
    if requirement.subject and requirement.subject.category:
        pieces.append(requirement.subject.category)
    if requirement.predicate:
        pieces.append(requirement.predicate)
    return ": ".join(pieces)


def _status(observations: bool, required_support_present: bool, complete: bool) -> str:
    if not observations:
        return "UNAVAILABLE"
    if complete:
        return "COMPLETE"
    if required_support_present:
        return "PARTIAL"
    return "INSUFFICIENT"


def assess_support(requirements: QuestionRequirementSet,
                   bundle: EvidenceBundle | None) -> SupportAssessment:
    mandatory = tuple(item for item in requirements.requirements if item.optionality is Optionality.MANDATORY)
    mandatory_ids = {item.requirement_id for item in mandatory}
    decisions = bundle.requirement_support_decisions if bundle else {}
    authoritative = bool(decisions)
    if bundle and not authoritative:
        decisions = {decision.requirement_id: decision for decision in project_legacy_support(bundle)}
    # A present but incomplete decision map never inherits legacy support.
    levels = {item.requirement_id: decisions[item.requirement_id].level
              if item.requirement_id in decisions else SupportLevel.NO_SUPPORT
              for item in requirements.requirements}
    supported = tuple(identifier for identifier, level in levels.items() if level is SupportLevel.FULL_SUPPORT)
    conditional = tuple(identifier for identifier, level in levels.items()
                        if level is SupportLevel.CONDITIONAL_SUPPORT)
    no_support = tuple(identifier for identifier, level in levels.items() if level is SupportLevel.NO_SUPPORT)
    supported_set = set(supported)
    all_supported, missing = _logical_support(requirements, supported_set)
    required_support_present = bool(mandatory_ids & (supported_set | set(conditional)))
    projection_consistent = not authoritative or (
        set(decisions) == set(levels)
        and bundle.supported_requirement_ids == supported
        and bundle.conditional_requirement_ids == conditional
        and bundle.no_support_requirement_ids == no_support
        and bundle.missing_requirement_ids == missing
        and bundle.provenance.get("requirement_support") == {
            identifier: list(decisions[identifier].supporting_fact_ids) for identifier in supported
        }
    )
    observations = bool(bundle and _fact_ids(bundle))
    role_consistent = bool(bundle and _role_consistent(requirements, bundle, supported_set))
    temporal_consistent = bool(bundle and _temporal_consistent(requirements, bundle, supported_set))
    provenance_complete = bool(bundle and _provenance_complete(bundle, decisions))
    budget = BudgetStatus(None)
    complete = False
    return SupportAssessment(
        observations_available=observations,
        mandatory_count=len(mandatory), supported_mandatory_count=len(mandatory_ids & supported_set),
        supported_requirement_ids=supported, missing_requirement_ids=missing,
        full_support_count=len(supported), conditional_support_count=len(conditional),
        no_support_count=len(no_support), conditional_requirement_ids=conditional,
        no_support_requirement_ids=no_support, projection_consistent=projection_consistent,
        all_mandatory_supported=all_supported,
        parse_complete=requirements.parse_status is ParseStatus.COMPLETE,
        role_consistent=role_consistent, temporal_consistent=temporal_consistent,
        provenance_complete=provenance_complete,
        no_answer_support=observations and not (mandatory_ids & supported_set),
        budget_status=budget, complete=complete,
        compatibility_status=_status(observations, required_support_present, complete),
        requirement_descriptions={item.requirement_id: _description(item) for item in requirements.requirements},
        _requirements=requirements, _required_support_present=required_support_present,
    )


def finalize_budget(assessment: SupportAssessment, budget_status: BudgetStatus, *,
                    visible_supported_ids: tuple[str, ...]) -> SupportAssessment:
    visible = set(visible_supported_ids)
    supported = tuple(identifier for identifier in assessment.supported_requirement_ids if identifier in visible)
    if assessment._requirements is not None:
        all_supported, missing = _logical_support(assessment._requirements, set(supported))
        supported_mandatory_count = sum(
            item.optionality is Optionality.MANDATORY and item.requirement_id in supported
            for item in assessment._requirements.requirements
        )
        required_support_present = assessment._required_support_present
    else:
        # Preserve finalization for assessments constructed through the old shape.
        missing = tuple(dict.fromkeys(
            assessment.missing_requirement_ids
            + tuple(identifier for identifier in assessment.supported_requirement_ids if identifier not in visible)
        ))
        all_supported = assessment.all_mandatory_supported and not missing
        supported_mandatory_count = len(supported)
        required_support_present = bool(assessment.supported_mandatory_count or assessment.conditional_support_count)
    complete = (
        assessment.observations_available and assessment.parse_complete and all_supported and assessment.role_consistent
        and assessment.temporal_consistent and assessment.provenance_complete
        and assessment.projection_consistent and budget_status.bundle_fits is True
    )
    return replace(
        assessment,
        supported_mandatory_count=supported_mandatory_count, supported_requirement_ids=supported,
        missing_requirement_ids=missing, all_mandatory_supported=all_supported,
        no_answer_support=assessment.observations_available and not supported_mandatory_count,
        budget_status=budget_status, complete=complete,
        compatibility_status=_status(assessment.observations_available, required_support_present, complete),
    )
