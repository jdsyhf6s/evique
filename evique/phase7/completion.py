from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Mapping, Protocol

from .bundle_builder import _evidence_groups
from .normalization import (
    DEFAULT_NORMALIZATION_POLICY, NormalizationPolicy, NormalizationResult, Normalizer,
    assess_identity, assess_requirement_support, identity_evidence_from_facts, normalize_and_fuse,
)
from .schema import (
    EvidenceBundle,
    EvidenceFact,
    IdentityStatus,
    LogicOperator,
    Optionality,
    QuestionRequirement,
    QuestionRequirementSet,
    RequirementType,
    RequirementSupportDecision,
    SubjectBinding,
    SupportLevel,
    from_dict,
)
from .support import _logical_support


TYPED_FETCH_LIMIT = 20
TYPED_ROUTES = {
    RequirementType.SPATIAL_RELATION: "same_frame_relation",
    RequirementType.AFTER_STATE: "post_anchor_scene_state",
    RequirementType.BEFORE_STATE: "pre_anchor_scene_state",
    RequirementType.STATE_CHANGE: "ordered_scene_state_pair",
    RequirementType.MOTION: "compact_track",
    RequirementType.ENTRY_EXIT: "compact_track",
}


class CompletionSource(Protocol):
    def fetch(self, requirement: QuestionRequirement, bundle: EvidenceBundle, *,
              limit: int) -> list[EvidenceFact]: ...


@dataclass(frozen=True)
class CompletionResult:
    bundles: tuple[EvidenceBundle, ...]
    rebuild_count: int
    stop_reason: str
    trace: dict


def _all_evidence(bundle: EvidenceBundle) -> set[str]:
    return set(
        bundle.scene_evidence + bundle.object_evidence + bundle.relation_evidence
        + bundle.track_evidence + bundle.temporal_evidence + bundle.event_evidence
    )


def _admit(requirement: QuestionRequirement, bundle: EvidenceBundle,
           facts: list[EvidenceFact], *,
           reserved_facts: dict[str, EvidenceFact] | None = None) -> tuple[list[EvidenceFact], list[dict]]:
    reserved_facts = reserved_facts if reserved_facts is not None else {}
    known_ids = _all_evidence(bundle) | set(bundle.provenance.get("fact_refs") or ())
    known_sources = {
        source_id
        for source_ids in (bundle.provenance.get("fact_sources") or {}).values()
        for source_id in source_ids
    }
    admitted = []
    rejected = []
    for fact in facts:
        reason = None
        if fact.video_id != bundle.video_id:
            reason = "wrong_video"
        elif not _time_compatible(requirement, bundle, fact):
            reason = "outside_time_window"
        elif fact.fact_id in known_ids:
            reason = "duplicate_fact"
        elif fact.fact_id in reserved_facts:
            reason = ("duplicate_fact" if reserved_facts[fact.fact_id] == fact
                      else "conflicting_duplicate_fact")
        elif set(fact.source_evidence_ids) and set(fact.source_evidence_ids) <= known_sources:
            reason = "duplicate_source"
        elif (not fact.source_evidence_ids or not fact.provenance
              or fact.provenance.get("video_id") != fact.video_id):
            reason = "invalid_provenance"
        if reason:
            rejected.append({"fact_id": fact.fact_id, "reason": reason})
            continue
        admitted.append(fact)
        known_ids.add(fact.fact_id)
        reserved_facts[fact.fact_id] = fact
        known_sources.update(fact.source_evidence_ids)
    return admitted, rejected


def _time_compatible(requirement: QuestionRequirement, bundle: EvidenceBundle,
                     fact: EvidenceFact) -> bool:
    inside = (
        fact.time_span.start >= bundle.time_window.start
        and fact.time_span.end <= bundle.time_window.end
    )
    if inside:
        return True
    if requirement.requirement_type is RequirementType.BEFORE_STATE:
        return fact.time_span.end <= bundle.time_window.start
    if requirement.requirement_type is RequirementType.AFTER_STATE:
        return fact.time_span.start >= bundle.time_window.end
    if requirement.requirement_type is RequirementType.STATE_CHANGE:
        return fact.time_span.start <= bundle.time_window.end and fact.time_span.end >= bundle.time_window.start
    return False


def _fact_payload(fact: EvidenceFact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "fact_type": fact.fact_type,
        "value": fact.value,
        "time_span": {"start": fact.time_span.start, "end": fact.time_span.end},
        "frame_ids": list(fact.frame_ids),
        "role_bindings": dict(fact.role_bindings),
        "source_evidence_ids": list(fact.source_evidence_ids),
        "source_kind": fact.source_kind.value,
        "source_family": fact.source_family.value if fact.source_family else None,
        "derived_from": list(fact.derived_from),
        "independence_root_ids": list(fact.independence_root_ids),
        "episode_id": fact.episode_id,
        "derivation_rule": fact.derivation_rule,
        "provenance": dict(fact.provenance),
    }


def _semantic_key(fact: EvidenceFact) -> str:
    value = [fact.fact_type, fact.video_id, fact.role_bindings, fact.value,
             round(fact.time_span.start, 1), round(fact.time_span.end, 1)]
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _completion_decision(requirement: QuestionRequirement,
                         normalization: NormalizationResult) -> RequirementSupportDecision:
    decision = assess_requirement_support(requirement, normalization.facts, normalization)
    if (requirement.requirement_type is RequirementType.STATE_CHANGE
            and requirement.subject and "EPISODE_MISMATCH" in decision.rationale_codes):
        decision = replace(decision, rationale_codes=decision.rationale_codes + ("CROSS_EPISODE_TARGET",))
    return decision


@dataclass(frozen=True)
class _IdentityEndpoint:
    target_id: str
    episode_id: str | None
    scene_id: str | None
    scope_id: str | None
    episode_known: bool
    role_key: str
    requirement_role_id: str | None
    observation_context: tuple | None = None


def _observation_context(payload):
    provenance = payload.get("provenance") or {}
    video = payload.get("video_id") or provenance.get("video_id")
    frames, span = payload.get("frame_ids"), payload.get("time_span") or {}
    start, end = span.get("start"), span.get("end")
    if (not isinstance(video, str) or not video or not isinstance(frames, (list, tuple)) or not frames
            or any(not isinstance(frame, str) or not frame for frame in frames)
            or any(isinstance(point, bool) or not isinstance(point, (int, float)) or not math.isfinite(point)
                   for point in (start, end)) or start > end):
        return None
    return (video, provenance.get("scene_id"), provenance.get("scope_id"), tuple(sorted(set(frames))), start, end)


def _explicit_endpoint_episode(provenance, *role_keys):
    values = {
        value for role_key in dict.fromkeys(role_keys)
        if role_key is not None
        and isinstance((value := provenance.get(f"{role_key}_episode_id")), str)
        and value
    }
    if len(values) > 1:
        return None, False
    if values:
        return next(iter(values)), True
    return None, None


def _endpoint_episode(fact, target, role_key, component, visible, requirement_role_id=None):
    provenance = fact.get("provenance") or {}
    explicit_episode, resolved = _explicit_endpoint_episode(
        provenance, requirement_role_id, role_key,
    )
    if resolved is not None:
        return explicit_episode, resolved
    bindings = fact.get("role_bindings") or {}
    value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
    # A fact's own episode belongs only to its own observed/track target.
    # Multi-endpoint relation episodes cannot stand in for a reference episode.
    own_target = (set(bindings.values()) == {target}
                  or (role_key in {"subject", "actor"} and value.get("track_id") == target))
    # Null stored episodes are unresolved; canonical untracked linkage requires
    # an exact authoritative assessment reference in the conflict matcher.
    if own_target and fact.get("episode_id") is not None:
        return fact["episode_id"], True
    scene = provenance.get("scene_id")
    scope = provenance.get("scope_id")
    episodes = set()
    for identifier in component.fact_ids + component.observation_only_fact_ids:
        observed = visible.get(identifier)
        if observed is None or observed.get("episode_id") is None:
            continue
        observed_roles = set((observed.get("role_bindings") or {}).values())
        if observed_roles != {target}:
            continue
        observed_provenance = observed.get("provenance") or {}
        assessment_ref = observed_provenance.get("identity_assessment_id")
        if assessment_ref not in component.identity_assessment_refs:
            continue
        observed_scene = observed_provenance.get("scene_id")
        observed_scope = observed_provenance.get("scope_id")
        if (scene is not None and observed_scene is not None and scene != observed_scene
                or scope is not None and observed_scope is not None and scope != observed_scope):
            continue
        episodes.add(observed["episode_id"])
    return (next(iter(episodes)), True) if len(episodes) == 1 else (None, False)


def _identity_rows(decision: RequirementSupportDecision, component, role, generic: str,
                   visible: Mapping[str, dict]) -> list[_IdentityEndpoint]:
    identifiers = decision.supporting_fact_ids or tuple(
        identifier for item in decision.component_decisions
        if ":identity:" not in item.component_id for identifier in item.fact_ids
    )
    rows = []
    for identifier in identifiers:
        fact = visible.get(identifier)
        if fact is None:
            continue
        bindings, provenance = fact.get("role_bindings") or {}, fact.get("provenance") or {}
        target = bindings.get(role.role_id) or bindings.get(generic)
        if target is None:
            continue
        endpoints = [(target, generic)]
        explicit_target = provenance.get(f"{generic}_identity_target_id")
        if explicit_target and explicit_target != target:
            endpoints.append((explicit_target, generic))
        if (generic == "reference" and bindings.get("reference_2")
                and (fact.get("fact_type") == "between" or provenance.get("predicate_test") == "between")):
            endpoints.append((bindings["reference_2"], "reference_2"))
        for target, role_key in endpoints:
            requirement_role_id = role.role_id if role_key == generic else None
            episode, known = _endpoint_episode(
                fact, target, role_key, component, visible, requirement_role_id,
            )
            rows.append(_IdentityEndpoint(
                target, episode, provenance.get("scene_id"), provenance.get("scope_id"),
                known, role_key, requirement_role_id, _observation_context(fact),
            ))
    return rows


def _same_target_scope(left, right) -> bool:
    return (left.target_id == right.target_id
            and not (left.scene_id is not None and right.scene_id is not None
                     and left.scene_id != right.scene_id)
            and not (left.scope_id is not None and right.scope_id is not None and left.scope_id != right.scope_id))


def _compatible_identity_scope(left, right) -> bool:
    return (_same_target_scope(left, right) and left.episode_known and right.episode_known
            and left.episode_id == right.episode_id)


def _retain_prior_conflict(prior, proposal, old_component, component, role, generic, visible) -> bool:
    old_rows = _identity_rows(prior, old_component, role, generic, visible)
    new_rows = _identity_rows(proposal, component, role, generic, visible)
    if old_rows and new_rows:
        return any(_same_target_scope(left, right)
                   and (not left.episode_known or not right.episode_known or left.episode_id == right.episode_id)
                   for left in old_rows for right in new_rows)
    # Missing stored target metadata cannot resolve a retained contradiction.
    return True


def _assessment_refs(normalization, assessment):
    evidence_ids = {label.evidence_id for label in assessment.observed_labels}
    return tuple(sorted({fact.provenance["identity_assessment_id"] for fact in normalization.facts
                         if fact.fact_id in evidence_ids and fact.provenance.get("identity_assessment_id")}))


def _conflicts_for_component(decision, component, role, generic, visible, additions):
    rows = _identity_rows(decision, component, role, generic, visible)
    matches = []
    for result in additions.values():
        for assessment in result.identity_assessments:
            if assessment.status is not IdentityStatus.CONFLICT:
                continue
            refs = _assessment_refs(result, assessment)
            exact_ref = bool(set(component.identity_assessment_refs) & set(refs))
            linked = False
            for row in rows:
                label_rows = []
                for label in assessment.observed_labels:
                    provenance = label.provenance
                    episode, known = _explicit_endpoint_episode(
                        provenance, row.requirement_role_id, row.role_key, generic,
                    )
                    if known is None:
                        episode, known = assessment.episode_id, True
                    label_rows.append(_IdentityEndpoint(
                        assessment.local_target_id, episode,
                        provenance.get("scene_id"), provenance.get("scope_id"), known, row.role_key,
                        row.requirement_role_id,
                    ))
                if all(_same_target_scope(row, observed)
                       and (_compatible_identity_scope(row, observed) or (not row.episode_known and exact_ref))
                       for observed in label_rows):
                    linked = True
                    break
            if not rows:
                # Opaque equality proves exact normalized target+episode linkage;
                # an unknown episode never acts as a wildcard.
                linked = exact_ref
            if linked:
                matches.append((assessment, refs))
    return matches


def _cross_batch_conflicts(prior, component, role, generic, visible, additions):
    """Consume only negative identity assessments; never corroborate batches."""
    rows = _identity_rows(prior, component, role, generic, visible)
    stored = []
    for payload in visible.values():
        provenance = payload.get("provenance") or {}
        if provenance.get("identity_assessment_id") not in component.identity_assessment_refs:
            continue
        fact = from_dict(EvidenceFact, {**payload, "video_id": provenance["video_id"]})
        for label in identity_evidence_from_facts((fact,)):
            for row in rows:
                episode, known = _endpoint_episode(
                    payload, label.local_target_id, row.role_key, component, visible,
                    row.requirement_role_id,
                )
                endpoint = _IdentityEndpoint(label.local_target_id, episode,
                                             provenance.get("scene_id"), provenance.get("scope_id"),
                                             known, row.role_key, row.requirement_role_id)
                if (_compatible_identity_scope(row, endpoint)
                        or (_same_target_scope(row, endpoint) and not row.episode_known and not known)):
                    stored.append((label, payload, endpoint))
                    break
    candidates = {}
    for result in additions.values():
        policy_data = result.trace.get("policy", {})
        policy = NormalizationPolicy(**{**policy_data, "category_aliases": tuple(
            tuple(alias) for alias in policy_data.get("category_aliases", ()))})
        new_visible = {fact.fact_id: _fact_payload(fact) for fact in result.facts}
        for assessment in result.identity_assessments:
            refs = _assessment_refs(result, assessment)
            exact_ref = bool(set(component.identity_assessment_refs) & set(refs))
            linked_labels = {}
            for label in assessment.observed_labels:
                payload = new_visible[label.evidence_id]
                for old_label, old_payload, endpoint in stored:
                    episode, known = _endpoint_episode(
                        payload, label.local_target_id, endpoint.role_key, component, new_visible,
                        endpoint.requirement_role_id,
                    )
                    observed = _IdentityEndpoint(label.local_target_id, episode,
                                                label.provenance.get("scene_id"),
                                                label.provenance.get("scope_id"), known, endpoint.role_key,
                                                endpoint.requirement_role_id)
                    if not (_compatible_identity_scope(endpoint, observed)
                            or (_same_target_scope(endpoint, observed) and exact_ref
                                and not endpoint.episode_known and not known)):
                        continue
                    # Stable target/episode refs alone do not prove the labels
                    # describe the same local observation context.
                    if (endpoint.scene_id != observed.scene_id
                            or endpoint.scope_id != observed.scope_id
                            or not set(old_payload.get("frame_ids") or ()) & set(payload.get("frame_ids") or ())
                            or old_label.time_span.start > label.time_span.end
                            or label.time_span.start > old_label.time_span.end):
                        continue
                    linked_labels[old_label.evidence_id] = replace(old_label, episode_id=episode)
                    linked_labels[label.evidence_id] = replace(label, episode_id=episode)
            if not linked_labels:
                continue
            episodes = {label.episode_id for label in linked_labels.values()}
            for episode in sorted(episodes, key=lambda value: value or ""):
                group = candidates.setdefault((assessment.local_target_id, episode),
                                              {"labels": {}, "refs": set(), "policy": policy})
                group["labels"].update({identifier: label for identifier, label in linked_labels.items()
                                        if label.episode_id == episode})
                group["refs"].update(refs)
    matches = []
    # Assess the complete linked root graph. A later addition can connect
    # groups that looked independent when inspected as isolated pairs.
    for (target, episode), group in sorted(candidates.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        conflict = assess_identity(tuple(group["labels"].values()), episode_id=episode,
                                   local_target_id=target, policy=group["policy"])
        if conflict.status is IdentityStatus.CONFLICT:
            matches.append((conflict, tuple(sorted(group["refs"]))))
    return matches


def _requirement_roles(requirement):
    interaction = requirement.requirement_type is RequirementType.INTERACTION
    return ((requirement.subject, "actor" if interaction else "subject"),
            (requirement.reference_object, "patient" if interaction else "reference"),
            (requirement.comparison_target, "comparison"))


def _role_support_rows(requirement, decision, role, generic, visible):
    component = next((item for item in decision.component_decisions
                      if item.component_id == f"{requirement.requirement_id}:identity:{role.role_id}"), None)
    if component is None:
        component = next((item for item in decision.component_decisions if ":identity:" not in item.component_id), None)
    return (_identity_rows(decision, component, role, generic, visible), component) if component else ([], None)


def _selected_or_requirements(requirements, decisions, prior_decisions, compatible):
    mandatory = {item.requirement_id for item in requirements.requirements if item.optionality is Optionality.MANDATORY}
    prior_full = {identifier for identifier, decision in prior_decisions.items() if decision.level is SupportLevel.FULL_SUPPORT}

    def select(expression):
        if isinstance(expression, str):
            if expression not in mandatory:
                return None
            full = decisions[expression].level is SupportLevel.FULL_SUPPORT
            return (frozenset((expression,)),) if full else ()
        children = [selected for child in expression.children if (selected := select(child)) is not None]
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

    candidates = select(requirements.root_logic) or ()
    if not candidates:
        return set()
    ordered = sorted(candidates, key=lambda identifiers: (
        -len(identifiers & prior_full), tuple(sorted(identifiers))))
    return set(next((identifiers for identifiers in ordered if compatible(identifiers)), ordered[0]))


def _contains_or(expression):
    return (not isinstance(expression, str)
            and (expression.operator is LogicOperator.OR
                 or any(_contains_or(child) for child in expression.children)))


def _role_rows_compatible(left_rows, right_rows, left_component, right_component):
    exact_ref = bool(left_component and right_component and
                     set(left_component.identity_assessment_refs) & set(right_component.identity_assessment_refs))
    category_free = left_component is not None and right_component is not None and all(
        ":identity:" not in item.component_id for item in (left_component, right_component))
    return bool(left_rows) and bool(right_rows) and all(
        _compatible_identity_scope(left, right)
        or (_same_target_scope(left, right) and not left.episode_known and not right.episode_known
            and (exact_ref or (category_free and left.observation_context is not None
                               and left.observation_context == right.observation_context)))
        for left in left_rows for right in right_rows
    )


def _role_support_compatible(support):
    identifiers = {row.target_id for _, rows, _ in support for row in rows}
    if len(identifiers) != 1:
        return False
    return all(
        _role_rows_compatible(left_rows, right_rows, left_component, right_component)
        for index, (_, left_rows, left_component) in enumerate(support)
        for _, right_rows, right_component in support[index + 1:]
    )


def _selected_role_support(requirements, decisions, selected, visible):
    roles = {}
    for requirement in requirements.requirements:
        if requirement.requirement_id not in selected:
            continue
        decision = decisions[requirement.requirement_id]
        for role, generic in _requirement_roles(requirement):
            if role is None:
                continue
            rows, component = _role_support_rows(requirement, decision, role, generic, visible)
            roles.setdefault(role.role_id, []).append((decision, rows, component))
    return roles


def _reconcile_role_support(bundle, requirements, decisions, additions, visible):
    subjects = dict(bundle.focal_subjects)
    consistency = {**bundle.consistency, "role_checks": dict(bundle.consistency.get("role_checks") or {}),
                   "relation_checks": dict(bundle.consistency.get("relation_checks") or {})}
    prior_roles, new_roles = {}, {}
    for requirement in requirements.requirements:
        identifier = requirement.requirement_id
        for role, generic in _requirement_roles(requirement):
            if role is None:
                continue
            prior = bundle.requirement_support_decisions.get(identifier)
            if prior and prior.supporting_fact_ids:
                rows, component = _role_support_rows(requirement, prior, role, generic, visible)
                prior_roles.setdefault(role.role_id, []).append((prior, rows, component))
            decision = decisions[identifier]
            if identifier in additions and decision.supporting_fact_ids:
                rows, component = _role_support_rows(requirement, decision, role, generic, visible)
                new_roles.setdefault(role.role_id, []).append((decision, rows, component))
    selected_roles = {}
    if _contains_or(requirements.root_logic):
        role_support = lambda identifiers: _selected_role_support(requirements, decisions, identifiers, visible)
        selected = _selected_or_requirements(
            requirements, decisions, bundle.requirement_support_decisions,
            lambda identifiers: all(_role_support_compatible(support)
                                    for support in role_support(identifiers).values()),
        )
        selected_roles = role_support(selected)
    else:
        selected = None
    for role_id, proposals in new_roles.items():
        prior_support = prior_roles.get(role_id, [])
        support = prior_support + proposals
        mismatch = False
        if selected is not None:
            full = [item for item in support if item[0].level is SupportLevel.FULL_SUPPORT
                    and decisions[item[0].requirement_id].level is SupportLevel.FULL_SUPPORT]
            alternatives = selected_roles.get(role_id, [])
            # Alternative decisions remain inspectable; only a selected branch
            # contributes targets to this shared role's consistency check.
            support = alternatives or sorted(full, key=lambda item: item[0].requirement_id)[:1] or support
            proposals = []
            mismatch = bool(alternatives) and not _role_support_compatible(alternatives)
        for proposal, rows, component in proposals:
            incompatible = False
            for other, old_rows, old_component in support:
                if other is proposal:
                    continue
                if not _role_rows_compatible(rows, old_rows, component, old_component):
                    incompatible = True
            if incompatible and requirements.root_logic.operator is not LogicOperator.OR:
                mismatch = True
                if proposal.level is not SupportLevel.FULL_SUPPORT:
                    continue
                failed = replace(component, level=SupportLevel.NO_SUPPORT, fact_ids=(),
                                 observation_only_fact_ids=tuple(sorted(set(component.fact_ids + component.observation_only_fact_ids))),
                                 rationale_codes=("ROLE_CONTINUITY_MISMATCH",))
                current = decisions[proposal.requirement_id]
                decisions[proposal.requirement_id] = _with_identity_conflict(current, component, failed)
        identifiers = tuple(sorted({row.target_id for _, rows, _ in support for row in rows}))
        if identifiers:
            subjects[role_id] = SubjectBinding(role_id, identifiers)
        prior_check = consistency["role_checks"].get(role_id) or {}
        evidence_ids = set(prior_check.get("evidence_ids") or ()) if selected is None else set()
        evidence_ids.update(identifier for decision, _, _ in support for identifier in decision.supporting_fact_ids)
        consistency["role_checks"][role_id] = {
            "consistent": not mismatch and len(identifiers) == 1,
            "evidence_ids": sorted(evidence_ids),
            "warnings": ["identity_discontinuity"] if mismatch or len(identifiers) != 1 else [],
        }
    return subjects, consistency


def _with_identity_conflict(decision, component, conflict_component):
    components = tuple(conflict_component if item.component_id == component.component_id else item
                       for item in decision.component_decisions)
    observations = set(decision.observation_only_fact_ids) | set(decision.supporting_fact_ids)
    observations.update(identifier for item in components
                        for identifier in item.fact_ids + item.observation_only_fact_ids)
    # The mandatory identity component is already NO from retained or newly
    # normalized authority. Preserve primary evidence; no primary reassessment.
    return replace(
        decision, level=SupportLevel.NO_SUPPORT, supporting_fact_ids=(),
        observation_only_fact_ids=tuple(sorted(observations)), component_decisions=components,
        identity_assessment_refs=tuple(sorted({ref for item in components for ref in item.identity_assessment_refs})),
        provenance_root_ids=tuple(sorted({root for item in components for root in item.provenance_root_ids})),
        rationale_codes=tuple(dict.fromkeys(reason for item in components for reason in item.rationale_codes)),
    )


def _rebuild(bundle: EvidenceBundle, additions: dict[str, NormalizationResult],
             requirements: QuestionRequirementSet, *,
             reconciliation_additions: Mapping[str, NormalizationResult] | None = None) -> EvidenceBundle:
    reconciliation_additions = reconciliation_additions or additions
    owned_new = {fact.fact_id: fact for result in additions.values() for fact in result.facts}
    prior_visible = {row["fact_id"]: row for row in bundle.provenance.get("visible_facts") or ()}
    known_ids = _all_evidence(bundle) | set(bundle.provenance.get("fact_refs") or ()) | set(owned_new)
    owned_results = {id(result) for result in additions.values()}
    reconciliation_facts = dict(owned_new)
    reconciliation_rejections = []
    for result in reconciliation_additions.values():
        if id(result) in owned_results:
            continue
        for fact in result.facts:
            payload = _fact_payload(fact)
            if fact.fact_id in known_ids:
                if prior_visible.get(fact.fact_id) != payload:
                    reconciliation_rejections.append(
                        {"fact_id": fact.fact_id, "reason": "conflicting_duplicate_fact"}
                    )
                continue
            existing = reconciliation_facts.get(fact.fact_id)
            if existing is not None:
                if existing != fact:
                    reconciliation_rejections.append(
                        {"fact_id": fact.fact_id, "reason": "conflicting_duplicate_fact"}
                    )
                continue
            reconciliation_facts[fact.fact_id] = fact
    invalid_reconciliation_ids = {
        row["fact_id"] for row in reconciliation_rejections
        if row["reason"] == "conflicting_duplicate_fact"
    }
    analysis_visible = {
        **prior_visible,
        **{fact.fact_id: _fact_payload(fact) for fact in reconciliation_facts.values()},
    }
    conflict_fact_ids = set()
    decisions = {}
    for requirement in requirements.requirements:
        identifier = requirement.requirement_id
        prior = bundle.requirement_support_decisions.get(identifier)
        if identifier in additions:
            decisions[identifier] = _completion_decision(requirement, additions[identifier])
        elif prior is not None:
            decisions[identifier] = prior
        else:
            decisions[identifier] = RequirementSupportDecision(
                identifier, SupportLevel.NO_SUPPORT, (), (), (), (), (),
                rationale_codes=("NO_AUTHORITATIVE_SUPPORT_DECISION",),
            )
        roles = ((requirement.subject, "actor" if requirement.requirement_type is RequirementType.INTERACTION else "subject"),
                 (requirement.reference_object, "patient" if requirement.requirement_type is RequirementType.INTERACTION else "reference"),
                 (requirement.comparison_target, "comparison"))
        for role, generic in roles:
            if role is None or not role.category:
                continue
            decision = decisions[identifier]
            component_id = f"{identifier}:identity:{role.role_id}"
            component = next((item for item in decision.component_decisions if item.component_id == component_id), None)
            if component is None:
                continue
            old_component = next((item for item in prior.component_decisions if item.component_id == component_id), None) if prior else None
            if (old_component is not None and "IDENTITY_CONFLICT" in old_component.rationale_codes
                    and _retain_prior_conflict(prior, decision, old_component, component, role, generic,
                                               analysis_visible)):
                decisions[identifier] = _with_identity_conflict(decision, component, old_component)
                decision = decisions[identifier]
                component = old_component
            stored_identity = old_component is not None and any(
                (payload.get("provenance") or {}).get("identity_assessment_id") in old_component.identity_assessment_refs
                for payload in prior_visible.values())
            # Stored normalized labels permit the stricter local-context check.
            # Retain the legacy exact-reference path only when labels are absent.
            matches = [] if stored_identity else _conflicts_for_component(
                decision, component, role, generic, analysis_visible, reconciliation_additions)
            if prior is not None and old_component is not None:
                matches.extend(_cross_batch_conflicts(
                    prior, old_component, role, generic, prior_visible, reconciliation_additions))
            if identifier in additions:
                own_visible = {fact.fact_id: _fact_payload(fact) for fact in additions[identifier].facts}
                own_result = additions[identifier]
                others = {key: result for key, result in reconciliation_additions.items()
                          if result is not own_result}
                matches.extend(_cross_batch_conflicts(decision, component, role, generic, own_visible, others))
            if matches:
                labels = [label for assessment, _ in matches for label in assessment.observed_labels]
                valid_labels = [label for label in labels
                                if label.evidence_id not in invalid_reconciliation_ids]
                invalid_match = any(label.evidence_id in invalid_reconciliation_ids for label in labels)
                conflict_fact_ids.update(label.evidence_id for label in valid_labels)
                conflict_component = replace(
                    component, level=SupportLevel.NO_SUPPORT, fact_ids=(),
                    observation_only_fact_ids=tuple(sorted(set(component.fact_ids + component.observation_only_fact_ids)
                                                          | {label.evidence_id for label in valid_labels})),
                    identity_assessment_refs=tuple(sorted(set(component.identity_assessment_refs)
                                                        | {ref for assessment, refs in matches for ref in refs
                                                           if all(label.evidence_id not in invalid_reconciliation_ids
                                                                  for label in assessment.observed_labels)})),
                    episode_ids=tuple(sorted(set(component.episode_ids)
                                             | {label.episode_id for label in valid_labels if label.episode_id})),
                    provenance_root_ids=tuple(sorted(set(component.provenance_root_ids)
                                                     | {root for label in valid_labels
                                                        for root in label.independence_root_ids})),
                    rationale_codes=(("IDENTITY_CONFLICT", "CONFLICTING_RECONCILIATION_FACT_ID")
                                     if invalid_match else ("IDENTITY_CONFLICT",)),
                )
                decisions[identifier] = _with_identity_conflict(decision, component, conflict_component)
    all_new = list({
        **owned_new,
        **{identifier: reconciliation_facts[identifier] for identifier in conflict_fact_ids
           if identifier in reconciliation_facts},
    }.values())
    externally_changed = any(
        identifier in bundle.requirement_support_decisions
        and decision != bundle.requirement_support_decisions[identifier]
        for identifier, decision in decisions.items()
    )
    if not additions and not externally_changed and not reconciliation_rejections:
        return bundle
    grouped = _evidence_groups(all_new)
    visible = {**prior_visible, **{fact.fact_id: _fact_payload(fact) for fact in all_new}}
    subjects, consistency = _reconcile_role_support(bundle, requirements, decisions, additions, visible)
    supported = tuple(key for key, decision in decisions.items() if decision.level is SupportLevel.FULL_SUPPORT)
    conditional = tuple(key for key, decision in decisions.items() if decision.level is SupportLevel.CONDITIONAL_SUPPORT)
    no_support = tuple(key for key, decision in decisions.items() if decision.level is SupportLevel.NO_SUPPORT)
    _, missing = _logical_support(requirements, set(supported))
    provenance = dict(bundle.provenance)
    fact_refs = list(dict.fromkeys(list(provenance.get("fact_refs") or ()) + [fact.fact_id for fact in all_new]))
    provenance.update({
        "fact_refs": fact_refs,
        "fact_videos": {**(provenance.get("fact_videos") or {}), **{fact.fact_id: fact.video_id for fact in all_new}},
        "fact_sources": {**(provenance.get("fact_sources") or {}), **{fact.fact_id: list(fact.source_evidence_ids) for fact in all_new}},
        "fact_semantic_keys": {**(provenance.get("fact_semantic_keys") or {}), **{fact.fact_id: _semantic_key(fact) for fact in all_new}},
        "requirement_support": {key: list(decisions[key].supporting_fact_ids) for key in supported},
        "visible_facts": list(visible.values()), "completion_rounds": 1,
        "identity_demoted_requirement_ids": sorted(set(bundle.supported_requirement_ids) - set(supported)),
    })
    if reconciliation_rejections:
        provenance["completion_reconciliation_rejections"] = list({
            (row["fact_id"], row["reason"]): row
            for row in [*(provenance.get("completion_reconciliation_rejections") or ()),
                        *reconciliation_rejections]
        }.values())
    for requirement in requirements.requirements:
        decision = decisions[requirement.requirement_id]
        if requirement.requirement_id not in additions:
            continue
        if requirement.requirement_type in {RequirementType.SPATIAL_RELATION, RequirementType.INTERACTION}:
            consistency["relation_checks"][requirement.requirement_id] = {
                "consistent": decision.level is SupportLevel.FULL_SUPPORT,
                "evidence_ids": list(decision.supporting_fact_ids),
                "warnings": [] if decision.level is SupportLevel.FULL_SUPPORT else list(decision.rationale_codes),
            }
    temporal_facts = [fact for fact in all_new if fact.fact_type in {
        "before_state", "after_state", "scene_state_before", "scene_state_after", "state_change"}]
    if temporal_facts:
        consistency["temporal_consistency"] = bundle.consistency.get("temporal_consistency") is not False and all(
            fact.derivation_rule and fact.provenance.get("video_id") == bundle.video_id for fact in temporal_facts)
    rebuilt = replace(
        bundle, time_window=type(bundle.time_window)(
            min([bundle.time_window.start, *(fact.time_span.start for fact in all_new)]),
            max([bundle.time_window.end, *(fact.time_span.end for fact in all_new)])),
        focal_subjects=subjects, requirement_support_decisions=decisions,
        supported_requirement_ids=supported, conditional_requirement_ids=conditional,
        no_support_requirement_ids=no_support, missing_requirement_ids=missing,
        provenance=provenance, consistency=consistency,
        token_cost=bundle.token_cost + math.ceil(len(json.dumps([_fact_payload(fact) for fact in all_new],
                                                    ensure_ascii=False, sort_keys=True)) / 4),
        **{key: tuple(dict.fromkeys(getattr(bundle, key) + value)) for key, value in grouped.items()},
    )
    object.__setattr__(rebuilt, "missing_requirement_ids", missing)
    return rebuilt


def complete_bundles(requirements: QuestionRequirementSet, bundles: list[EvidenceBundle],
                     sources: Mapping[RequirementType, CompletionSource], *,
                     max_rounds: int = 1, normalizer: Normalizer = normalize_and_fuse,
                     normalization_policy: NormalizationPolicy = DEFAULT_NORMALIZATION_POLICY) -> CompletionResult:
    if len(bundles) > 2:
        raise ValueError("completion accepts at most two bundles")
    if type(max_rounds) is not int or max_rounds not in {0, 1}:
        raise ValueError("completion permits at most one round")
    trace = {"max_rounds": max_rounds, "fetch_limit": TYPED_FETCH_LIMIT, "fetches": [], "normalization_calls": 0}
    if max_rounds == 0 or not bundles:
        return CompletionResult(tuple(bundles), 0, "disabled" if max_rounds == 0 else "no_bundles", trace)

    by_id = {item.requirement_id: item for item in requirements.requirements}
    fetched_requirement_ids = set()
    additions_by_bundle: dict[int, dict[str, NormalizationResult]] = {}
    reserved_facts_by_bundle: dict[int, dict[str, EvidenceFact]] = {
        index: {} for index in range(len(bundles))
    }
    for bundle_index, bundle in enumerate(bundles):
        supported = set(bundle.supported_requirement_ids)
        for requirement_id in bundle.missing_requirement_ids:
            requirement = by_id.get(requirement_id)
            if (
                requirement is None
                or requirement_id in supported
                or requirement.optionality is Optionality.OPTIONAL
                or requirement_id in fetched_requirement_ids
            ):
                continue
            fetched_requirement_ids.add(requirement_id)
            source = sources.get(requirement.requirement_type)
            entry = {
                "bundle_id": bundle.bundle_id,
                "requirement_id": requirement_id,
                "requirement_type": requirement.requirement_type.value,
                "source_kind": requirement.requirement_type.value,
                "constraints": {
                    "video_id": bundle.video_id,
                    "time_window": {"start": bundle.time_window.start, "end": bundle.time_window.end},
                    "typed_only": True,
                    "route": TYPED_ROUTES.get(requirement.requirement_type, "typed_offline_fact"),
                },
                "candidate_ids": [], "admitted_new_fact_ids": [], "rejected": [],
            }
            if source is None:
                entry["rejected"].append({"fact_id": None, "reason": "no_typed_source"})
                trace["fetches"].append(entry)
                continue
            try:
                candidates = source.fetch(requirement, bundle, limit=TYPED_FETCH_LIMIT)
                if not isinstance(candidates, list) or any(not isinstance(fact, EvidenceFact) for fact in candidates):
                    raise TypeError("typed completion source must return EvidenceFact list")
            except Exception as exc:
                entry["rejected"].append({"fact_id": None, "reason": "source_error", "error_type": type(exc).__name__})
                trace["fetches"].append(entry)
                continue
            entry["candidate_ids"] = [fact.fact_id for fact in candidates]
            admitted, rejected = _admit(
                requirement, bundle, candidates,
                reserved_facts=reserved_facts_by_bundle[bundle_index],
            )
            entry["admitted_new_fact_ids"] = [fact.fact_id for fact in admitted]
            entry["rejected"] = rejected
            if admitted:
                normalization = normalizer(admitted, video_id=bundle.video_id, policy=normalization_policy)
                trace["normalization_calls"] += 1
                normalized_ids = [fact.fact_id for fact in normalization.facts]
                if len(set(normalized_ids)) != len(admitted) or set(normalized_ids) != {fact.fact_id for fact in admitted}:
                    raise ValueError("completion normalization must preserve admitted fact IDs")
                decision = _completion_decision(requirement, normalization)
                entry["support_level"] = decision.level.value
                entry["support_eligible_fact_ids"] = list(decision.supporting_fact_ids)
                entry["observation_only_fact_ids"] = [fact.fact_id for fact in normalization.facts
                    if fact.fact_id not in entry["support_eligible_fact_ids"]]
                entry["identity_status_by_fact"] = {fact.fact_id: fact.provenance.get("identity_status")
                                                    for fact in normalization.facts}
                entry["normalization"] = normalization.trace
                additions_by_bundle.setdefault(bundle_index, {})[requirement_id] = normalization
            else:
                entry["support_eligible_fact_ids"] = []
                entry["observation_only_fact_ids"] = []
            trace["fetches"].append(entry)

    if not additions_by_bundle:
        trace["rebuild_count"] = 0
        trace["stop_reason"] = "no_new_facts"
        return CompletionResult(tuple(bundles), 0, "no_new_facts", trace)
    reconciliation_additions = {
        f"{bundle_index}:{requirement_id}": result
        for bundle_index, additions in additions_by_bundle.items()
        for requirement_id, result in additions.items()
    }
    rebuilt = []
    for index, bundle in enumerate(bundles):
        same_video = {
            key: result for key, result in reconciliation_additions.items()
            if result.facts and all(fact.video_id == bundle.video_id for fact in result.facts)
        }
        rebuilt.append(_rebuild(
            bundle, additions_by_bundle.get(index, {}), requirements,
            reconciliation_additions=same_video,
        ))
    trace["identity_demoted_requirement_ids_by_bundle"] = {
        bundle.bundle_id: list(bundle.provenance.get("identity_demoted_requirement_ids") or ())
        for bundle in rebuilt
        if bundle.provenance.get("identity_demoted_requirement_ids")
    }
    trace["rebuild_count"] = 1
    trace["stop_reason"] = "round_limit_reached"
    return CompletionResult(tuple(rebuilt), 1, "round_limit_reached", trace)
