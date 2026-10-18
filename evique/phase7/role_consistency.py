from __future__ import annotations

from dataclasses import dataclass
import math
from .normalization import DEFAULT_NORMALIZATION_POLICY, assess_identity, canonical_category
from .schema import (
    EvidenceFact,
    IdentityAssessment,
    IdentityEvidence,
    IdentityStatus as SchemaIdentityStatus,
    LogicOperator,
    Optionality,
    QuestionRequirementSet,
    RequirementSupportDecision,
    RequirementType,
    SubjectBinding,
    SourceFamily,
    SupportLevel,
    TimeRange,
)


@dataclass(frozen=True)
class ConsistencyCheck:
    consistent: bool | None
    evidence_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleConsistencyReport:
    role_checks: dict[str, ConsistencyCheck]
    relation_checks: dict[str, ConsistencyCheck]


class IdentityStatus:
    """Deprecated import shim; every member is a canonical schema status."""
    CONFIRMED = SchemaIdentityStatus.CONFIRMED
    CONSISTENT = SchemaIdentityStatus.CONFIRMED
    CONFLICT = SchemaIdentityStatus.CONFLICT
    UNVERIFIED = SchemaIdentityStatus.UNVERIFIED


class IdentityCategoryCheck(IdentityAssessment):
    """Assessment with read-only properties for deprecated category callers."""

    @property
    def categories(self) -> dict[str, str]:
        return {item.evidence_id: canonical_category(item.observed_category, DEFAULT_NORMALIZATION_POLICY)
                for item in self.observed_labels}

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.rationale_codes


def assess_identity_categories(*, detector_category: object = None,
                               track_category: object = None,
                               role_category: object = None,
                               semantic_category: object = None) -> IdentityCategoryCheck:
    """Retain legacy strings without granting them source independence."""
    evidence = tuple(
        IdentityEvidence(name, value, family, (), (), None, "legacy-target", TimeRange(0, 0), {})
        for name, value, family in (
            ("detector_category", detector_category, SourceFamily.DETECTOR),
            ("track_category", track_category, SourceFamily.TRACK_DERIVED),
            ("role_category", role_category, SourceFamily.OTHER_INDEPENDENT),
            ("semantic_category", semantic_category, SourceFamily.VISUAL_SEMANTIC),
        ) if isinstance(value, str) and value.strip()
    )
    assessment = assess_identity(evidence, episode_id=None, local_target_id="legacy-target",
                                 policy=DEFAULT_NORMALIZATION_POLICY)
    return IdentityCategoryCheck(
        assessment.status, assessment.episode_id, assessment.local_target_id,
        assessment.observed_labels, assessment.independent_source_groups,
        assessment.compatible_category, assessment.rationale_codes,
    )


def identity_check_payload(check: IdentityAssessment) -> dict:
    return {
        "status": check.status.value,
        "categories": {item.evidence_id: canonical_category(item.observed_category, DEFAULT_NORMALIZATION_POLICY)
                       for item in check.observed_labels},
        "warnings": list(check.rationale_codes),
    }


_CATEGORY_DEPENDENT_TYPES = {
    RequirementType.EXISTENCE,
    RequirementType.TARGET_ATTRIBUTE,
    RequirementType.MOTION,
    RequirementType.ENTRY_EXIT,
    RequirementType.BEFORE_STATE,
    RequirementType.AFTER_STATE,
    RequirementType.STATE_CHANGE,
    RequirementType.EVENT,
}


def identity_allows_requirement_support(requirement, fact: EvidenceFact) -> bool:
    """Keep observations while limiting category-dependent support promotion."""
    payload = fact.provenance.get("identity_consistency")
    if not isinstance(payload, dict):
        return True
    status = payload.get("status")
    if status in {IdentityStatus.CONFLICT.value, "IDENTITY_CONFLICT"}:
        return requirement.requirement_type not in (
            _CATEGORY_DEPENDENT_TYPES
            | {RequirementType.SPATIAL_RELATION, RequirementType.INTERACTION}
        )
    # Category-only legacy payloads carry no proof of independent sources.
    # Authoritative normalization assessments are consumed by the later support layer.
    return requirement.requirement_type not in _CATEGORY_DEPENDENT_TYPES


def _fact_subject_ids(fact: EvidenceFact) -> set[str]:
    return set(fact.role_bindings.values())


def _observation_context(fact: EvidenceFact) -> tuple | None:
    frames = tuple(sorted(set(fact.frame_ids)))
    points = (fact.time_span.start, fact.time_span.end)
    if (not fact.video_id or not frames
            or any(not isinstance(frame, str) or not frame for frame in frames)
            or any(isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point)
                   for point in points)
            or fact.time_span.start > fact.time_span.end):
        return None
    return (
        fact.video_id, fact.provenance.get("scene_id"), fact.provenance.get("scope_id"),
        frames, fact.time_span.start, fact.time_span.end,
    )


def _role_keys(requirements: QuestionRequirementSet) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for requirement in requirements.requirements:
        interaction = requirement.requirement_type is RequirementType.INTERACTION
        for role, generic in (
            (requirement.subject, "actor" if interaction else "subject"),
            (requirement.reference_object, "patient" if interaction else "reference"),
            (requirement.comparison_target, "comparison"),
        ):
            if role is not None:
                result.setdefault(role.role_id, []).extend((role.role_id, generic))
    return {role_id: tuple(dict.fromkeys(keys)) for role_id, keys in result.items()}


def _role_is_category_free(requirements: QuestionRequirementSet, role_id: str,
                           selected: set[str] | None = None) -> bool:
    references = [
        role for requirement in requirements.requirements
        if selected is None or requirement.requirement_id in selected
        for role in (requirement.subject, requirement.reference_object, requirement.comparison_target)
        if role is not None and role.role_id == role_id
    ]
    return bool(references) and all(role.category is None for role in references)


def _role_endpoints(facts: list[EvidenceFact], role_id: str, binding: SubjectBinding,
                    keys: tuple[str, ...]) -> tuple[tuple[str, str | None, str | None, tuple | None, bool], ...]:
    rows = []
    targets = set(binding.subject_ids)
    for fact in facts:
        matching = [(key, target) for key in keys
                    if (target := fact.role_bindings.get(key)) in targets]
        for key, target in matching:
            endpoint_keys = tuple(dict.fromkeys(
                candidate for candidate in keys
                if fact.role_bindings.get(candidate) in (None, target)
            ))

            def explicit_value(suffix):
                values = {
                    value for endpoint_key in endpoint_keys
                    if isinstance((value := fact.provenance.get(f"{endpoint_key}_{suffix}")), str)
                    and value
                }
                return (next(iter(values)) if len(values) == 1 else None, len(values) <= 1)

            explicit_episode, episode_resolved = explicit_value("episode_id")
            explicit_assessment, assessment_resolved = explicit_value("identity_assessment_id")
            endpoint_markers = {target}
            for endpoint_key in endpoint_keys:
                endpoint_markers.update(
                    value for suffix in ("identity_target_id", "raw_track_id", "track_id")
                    if isinstance((value := fact.provenance.get(f"{endpoint_key}_{suffix}")), str)
                    and value
                )
            value = fact.value if isinstance(fact.value, dict) else {}
            general_markers = {
                candidate for candidate in (
                    value.get("local_target_id"), value.get("object_id"), value.get("track_id"),
                    fact.provenance.get("local_target_id"), fact.provenance.get("identity_target_id"),
                    fact.provenance.get("raw_track_id"), fact.provenance.get("track_id"),
                )
                if isinstance(candidate, str) and candidate
            }
            sole_target = set(fact.role_bindings.values()) == {target}
            general_owns_endpoint = sole_target or bool(general_markers & endpoint_markers)
            has_general_metadata = bool(
                fact.episode_id
                or (isinstance(fact.provenance.get("identity_assessment_id"), str)
                    and fact.provenance["identity_assessment_id"])
            )
            ambiguous_general_metadata = (
                has_general_metadata and not sole_target and not general_markers
            )
            episode = explicit_episode
            if episode is None and general_owns_endpoint:
                episode = fact.episode_id
            assessment = explicit_assessment
            if assessment is None and general_owns_endpoint:
                general_assessment = fact.provenance.get("identity_assessment_id")
                if isinstance(general_assessment, str) and general_assessment:
                    assessment = general_assessment
            rows.append((
                target, episode,
                assessment,
                _observation_context(fact),
                episode_resolved and assessment_resolved and not ambiguous_general_metadata,
            ))
            break
    return tuple(dict.fromkeys(rows))


def _safe_role_continuity(rows, *, category_free: bool) -> bool:
    if len(rows) <= 1:
        return True
    if not all(row[4] for row in rows):
        return False
    episodes = {row[1] for row in rows if row[1] is not None}
    assessments = {row[2] for row in rows if row[2] is not None}
    if len(episodes) > 1 or len(assessments) > 1:
        return False
    if episodes and all(row[1] is not None for row in rows):
        return True
    if assessments and all(row[2] is not None for row in rows):
        return True
    if episodes or assessments:
        return False
    contexts = {row[3] for row in rows}
    targets = {row[0] for row in rows}
    return category_free and None not in contexts and len(contexts) == 1 and len(targets) == 1


def _selected_requirement_ids(requirements: QuestionRequirementSet,
                              decisions: dict[str, RequirementSupportDecision],
                              facts: list[EvidenceFact]) -> set[str] | None:
    mandatory = {
        item.requirement_id for item in requirements.requirements
        if item.optionality is Optionality.MANDATORY
    }

    def select(expression):
        if isinstance(expression, str):
            if expression not in mandatory:
                return None
            decision = decisions.get(expression)
            full = decision is not None and decision.level is SupportLevel.FULL_SUPPORT
            return (frozenset((expression,)),) if full else ()
        children = [selected for child in expression.children
                    if (selected := select(child)) is not None]
        if not children:
            return None
        if expression.operator is LogicOperator.AND:
            if any(not child for child in children):
                return ()
            selections = (frozenset(),)
            for child in children:
                selections = tuple(left | right for left in selections for right in child)
            return tuple(sorted(set(selections), key=lambda identifiers: tuple(sorted(identifiers))))
        return tuple(sorted({identifiers for child in children for identifiers in child},
                            key=lambda identifiers: tuple(sorted(identifiers))))

    candidates = select(requirements.root_logic)
    if not candidates:
        return None

    role_keys = _role_keys(requirements)

    def compatible(selected):
        for role_id, keys in role_keys.items():
            fact_ids = _role_fact_ids(requirements, decisions, set(selected), role_id)
            role_facts = [fact for fact in facts if fact.fact_id in fact_ids]
            if not role_facts:
                continue
            targets = tuple(sorted({target for fact in role_facts for key in keys
                                    if (target := fact.role_bindings.get(key)) is not None}))
            if len(targets) != 1:
                return False
            rows = _role_endpoints(role_facts, role_id, SubjectBinding(role_id, targets), keys)
            if not rows or not _safe_role_continuity(
                    rows, category_free=_role_is_category_free(requirements, role_id, set(selected))):
                return False
            tracks = {str(fact.value.get("track_id")) for fact in role_facts
                      if isinstance(fact.value, dict) and fact.value.get("track_id")}
            if len(tracks) > 1:
                return False
        return True

    return set(next((candidate for candidate in candidates if compatible(candidate)), candidates[0]))


def _role_fact_ids(requirements: QuestionRequirementSet,
                   decisions: dict[str, RequirementSupportDecision],
                   selected: set[str], role_id: str) -> set[str]:
    identifiers = set()
    for requirement in requirements.requirements:
        if requirement.requirement_id not in selected or not any(
            role is not None and role.role_id == role_id
            for role in (requirement.subject, requirement.reference_object, requirement.comparison_target)
        ):
            continue
        decision = decisions.get(requirement.requirement_id)
        if decision is None:
            continue
        identifiers.update(decision.supporting_fact_ids)
        identifiers.update(
            fact_id for component in decision.component_decisions for fact_id in component.fact_ids
        )
    return identifiers


def _check_role_consistency_with_decisions(
    requirements: QuestionRequirementSet,
    facts: list[EvidenceFact],
    bindings: dict[str, SubjectBinding],
    decisions: dict[str, RequirementSupportDecision] | None,
) -> RoleConsistencyReport:
    if len({fact.video_id for fact in facts}) > 1:
        raise ValueError("role consistency cannot mix video IDs")
    subject_roles = {
        requirement.subject.role_id for requirement in requirements.requirements if requirement.subject
    }
    role_keys = _role_keys(requirements)
    selected = _selected_requirement_ids(requirements, decisions, facts) if decisions is not None else None
    role_checks: dict[str, ConsistencyCheck] = {}
    for role_id, binding in bindings.items():
        if binding.role_id != role_id:
            raise ValueError("binding key and role_id differ")
        role_facts = facts
        if selected is not None and decisions is not None:
            fact_ids = _role_fact_ids(requirements, decisions, selected, role_id)
            role_facts = [fact for fact in facts if fact.fact_id in fact_ids]
        relevant = [
            fact for fact in role_facts
            if fact.role_bindings.get(role_id) in binding.subject_ids
            or bool(_fact_subject_ids(fact) & set(binding.subject_ids))
        ]
        direct = [fact for fact in role_facts if fact.role_bindings.get(role_id) in binding.subject_ids]
        continuity_rows = _role_endpoints(
            role_facts, role_id, binding, role_keys.get(role_id, (role_id,)),
        )
        continuity_targets = {row[0] for row in continuity_rows}
        enforce_continuity = decisions is not None or requirements.root_logic.operator is not LogicOperator.OR
        target_bound = (role_id in role_keys if decisions is not None
                        else role_id in subject_roles or len(binding.subject_ids) == 1)
        warnings = []
        conflict = False
        if (
            target_bound
            and len(continuity_targets or set(binding.subject_ids)) > 1
            and enforce_continuity
        ):
            conflict = True
            warnings.append("identity_discontinuity")
        tracks = {
            str(fact.value.get("track_id")) for fact in relevant
            if isinstance(fact.value, dict) and fact.value.get("track_id")
        }
        if target_bound and len(tracks) > 1:
            conflict = True
            warnings.append("track_discontinuity")
        if (target_bound
                and enforce_continuity
                and not _safe_role_continuity(
                    continuity_rows,
                    category_free=_role_is_category_free(requirements, role_id, selected),
                )):
            conflict = True
            warnings.append("identity_discontinuity")
        if conflict:
            consistent: bool | None = False
        elif direct:
            consistent = True
        elif relevant and len(binding.subject_ids) == 1:
            # A generic explicit relation endpoint can establish the binding,
            # but label equality without an endpoint remains unknown.
            consistent = True
        else:
            consistent = None
            warnings.append("identity_unknown")
        role_checks[role_id] = ConsistencyCheck(
            consistent, tuple(sorted(fact.fact_id for fact in relevant)), tuple(dict.fromkeys(warnings))
        )

    relation_checks: dict[str, ConsistencyCheck] = {}
    for requirement in requirements.requirements:
        if requirement.requirement_type not in {RequirementType.INTERACTION, RequirementType.SPATIAL_RELATION}:
            continue
        subject_ids = set(bindings.get(requirement.subject.role_id, SubjectBinding("unknown", ())).subject_ids) \
            if requirement.subject else set()
        reference_ids = set(bindings.get(requirement.reference_object.role_id, SubjectBinding("unknown", ())).subject_ids) \
            if requirement.reference_object else set()
        matches = []
        for fact in facts:
            if requirement.requirement_type is RequirementType.INTERACTION:
                type_match = fact.fact_type in {"interaction", requirement.predicate}
                left = fact.role_bindings.get("actor") or fact.role_bindings.get("subject")
                right = fact.role_bindings.get("patient") or fact.role_bindings.get("reference")
            else:
                type_match = fact.fact_type == requirement.predicate or (
                    requirement.predicate == "near" and fact.fact_type == "image_near"
                )
                left = fact.role_bindings.get("subject")
                right = fact.role_bindings.get("reference")
            if type_match and left in subject_ids and right in reference_ids:
                matches.append(fact)
        relation_checks[requirement.requirement_id] = ConsistencyCheck(
            True if matches else None,
            tuple(sorted(fact.fact_id for fact in matches)),
            () if matches else ("explicit_relation_unknown",),
        )
    return RoleConsistencyReport(role_checks, relation_checks)


def check_role_consistency(requirements: QuestionRequirementSet, facts: list[EvidenceFact],
                           bindings: dict[str, SubjectBinding]) -> RoleConsistencyReport:
    return _check_role_consistency_with_decisions(requirements, facts, bindings, None)
