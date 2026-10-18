from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Callable, Any

from .normalization import audit_support_safety
from .role_consistency import _check_role_consistency_with_decisions, _selected_requirement_ids
from .schema import (
    EvidenceBundle,
    EvidenceFact,
    ObservationSource,
    Optionality,
    QuestionRequirementSet,
    RequirementType,
    SourceFamily,
    SubjectBinding,
    SupportLevel,
    from_dict,
)
from .support import (
    BudgetStatus,
    SupportAssessment,
    _logical_support,
    _provenance_complete,
    _role_consistent,
    finalize_budget,
    project_legacy_support,
)


ANSWER_INSTRUCTION = (
    "State every concrete fact directly supported by the visible evidence. "
    "Answer each supported part, then state which requested aspects remain unknown. "
    "Do not infer missing spatial, temporal, interaction, or causal facts."
)


@dataclass(frozen=True)
class VisibleContext:
    messages: tuple[dict[str, str], ...]
    user_content: str
    visible_bundle_ids: tuple[str, ...]
    omitted_bundle_ids: tuple[str, ...]
    budget_status: BudgetStatus
    assessment: SupportAssessment
    trace: dict[str, Any]


def _safe_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "video_id", "source", "frame_id", "frame_ids", "camera_id", "segment_id",
        "scope_id", "scene_id", "model_id", "prompt_version", "region_id", "derivation_rule",
        "identity_status", "identity_assessment_id", "identity_target_id", "raw_track_id",
        "track_id", "subject_identity_target_id", "subject_raw_track_id",
        "anchor_event_id", "anchor_object_id", "anchor_identity_target_id",
        "anchor_episode_id", "anchor_role_id", "linked_fact_id", "parent_fact_id",
        "source_event_id",
    }
    return {key: item for key, item in value.items() if key in allowed}


def _safe_value(value: Any) -> Any:
    forbidden = ("confidence", "score", "utility", "routing", "annotation", "hidden", "candidate",
                 "root", "derived_from")
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item) for key, item in value.items()
            if not any(token in str(key).casefold() for token in forbidden)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return value


def _safe_fact(value: Any, fallback_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"evidence_id": fallback_id}
    result = {"evidence_id": str(value.get("fact_id") or value.get("evidence_id") or fallback_id)}
    for key in ("fact_type", "text", "value", "time_span", "frame_ids", "role_bindings",
                "source_evidence_ids", "derived_from", "episode_id"):
        if value.get(key) is not None:
            result[key] = _safe_value(value[key])
    provenance = _safe_provenance(value.get("provenance"))
    if provenance:
        result["provenance"] = provenance
    return result


def _observed_label(fact: dict[str, Any]) -> str | None:
    value = fact.get("value")
    if isinstance(value, dict):
        for key in ("observed_category", "category", "label"):
            label = value.get(key)
            if isinstance(label, str) and label.strip():
                return label
    return None


def _identity_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    provenance = fact.get("provenance") or {}
    value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
    target = next((candidate for candidate in (
        value.get("local_target_id"), provenance.get("local_target_id"), value.get("object_id"),
        value.get("track_id"), provenance.get("raw_track_id"), provenance.get("track_id"),
    ) if isinstance(candidate, str) and candidate.strip()), None)
    if target is None:
        endpoints = set((fact.get("role_bindings") or {}).values())
        target = next(iter(endpoints)) if len(endpoints) == 1 else None
    return provenance.get("identity_assessment_id"), fact.get("episode_id"), target


def _pure_identity_vote_family(fact: dict[str, Any]) -> str | None:
    """Return one trustworthy observed family, or fail closed for wrappers."""
    family = fact.get("source_family")
    source_kind = fact.get("source_kind")
    eligible_kinds = {
        SourceFamily.DETECTOR.value: {ObservationSource.DETECTOR_OBSERVATION.value},
        SourceFamily.SCOPE_SEMANTIC.value: {
            ObservationSource.OFFLINE_CAPTION.value,
            ObservationSource.TRANSCRIPT.value,
        },
        SourceFamily.VISUAL_SEMANTIC.value: {
            ObservationSource.QUERY_CONDITIONED_OBSERVATION.value,
        },
        SourceFamily.OTHER_INDEPENDENT.value: {
            ObservationSource.RULE_DERIVED_OBSERVATION.value,
        },
    }
    if (not isinstance(family, str) or not isinstance(source_kind, str)
            or family not in eligible_kinds or source_kind not in eligible_kinds[family]):
        return None
    derived_from = fact.get("derived_from")
    if not isinstance(derived_from, (list, tuple)) or derived_from or fact.get("derivation_rule"):
        return None
    fact_type = fact.get("fact_type")
    if not isinstance(fact_type, str):
        return None
    normalized_type = fact_type.casefold()
    if (normalized_type.startswith(("track_", "event_"))
            or normalized_type in {
                "track", "visual_track", "motion", "event", "visual_event",
                "entry", "exit", "entry_exit", "entry_exit_event", "state_change",
            }):
        return None
    source_ids = fact.get("source_evidence_ids")
    roots = fact.get("independence_root_ids")
    if (not isinstance(source_ids, (list, tuple)) or not source_ids
            or any(not isinstance(item, str) or not item for item in source_ids)
            or not isinstance(roots, (list, tuple)) or not roots
            or any(not isinstance(item, str) or not item for item in roots)):
        return None
    if family == SourceFamily.OTHER_INDEPENDENT.value:
        provenance = fact.get("provenance")
        if not isinstance(provenance, dict):
            return None
        record_id = provenance.get("source_record_id")
        if (provenance.get("observed") is not True
                or provenance.get("is_derived") is not False
                or record_id not in source_ids
                or record_id not in roots):
            return None
    return family


def _retained_identity_confirmed(component, by_id: dict[str, dict[str, Any]], retained: set[str]) -> bool:
    """Preserve confirmation only with independent explicit retained label sources.

    The prior FULL decision supplies category compatibility. This checks its
    remaining source sufficiency without normalizing, resolving aliases, or
    pooling evidence across assessment/target/episode boundaries.
    """
    groups = {}
    for identifier in component.fact_ids + component.observation_only_fact_ids:
        fact = by_id.get(identifier) or {}
        key = _identity_key(fact)
        if _observed_label(fact) is None or key[0] not in component.identity_assessment_refs:
            continue
        groups.setdefault(key, []).append((identifier, fact))
    if not groups:
        return False
    for labels in groups.values():
        source_groups: list[tuple[set[str], list[dict[str, Any]]]] = []
        for identifier, fact in labels:
            if identifier not in retained:
                continue
            declared_roots = fact.get("independence_root_ids")
            if (not isinstance(declared_roots, (list, tuple)) or not declared_roots
                    or any(not isinstance(root, str) or not root for root in declared_roots)):
                continue
            roots = set(declared_roots)
            members = [fact]
            separate = []
            for prior_roots, prior_members in source_groups:
                if roots.isdisjoint(prior_roots):
                    separate.append((prior_roots, prior_members))
                else:
                    roots.update(prior_roots)
                    members.extend(prior_members)
            source_groups = separate + [(roots, members)]
        voting_families = []
        for _, members in source_groups:
            families = [_pure_identity_vote_family(fact) for fact in members]
            if None not in families and len(set(families)) == 1:
                voting_families.append(families[0])
        if not any(left != right for index, left in enumerate(voting_families)
                   for right in voting_families[index + 1:]):
            return False
    return True


def _bundle_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    fact_ids = (
        bundle.scene_evidence + bundle.object_evidence + bundle.relation_evidence
        + bundle.track_evidence + bundle.temporal_evidence + bundle.event_evidence
    )
    configured = bundle.provenance.get("visible_facts") or []
    by_id = {
        str(item.get("fact_id") or item.get("evidence_id")): item
        for item in configured if isinstance(item, dict) and (item.get("fact_id") or item.get("evidence_id"))
    }
    sources = bundle.provenance.get("fact_sources") or {}
    facts = []
    for fact_id in fact_ids:
        safe = _safe_fact(by_id.get(fact_id), fact_id)
        if "source_evidence_ids" not in safe and sources.get(fact_id):
            safe["source_evidence_ids"] = sources[fact_id]
        facts.append(safe)
    decisions = bundle.requirement_support_decisions or {
        item.requirement_id: item for item in project_legacy_support(bundle)
    }
    uncertainty = []
    for decision in decisions.values():
        if decision.level is SupportLevel.FULL_SUPPORT:
            continue
        reasons = decision.rationale_codes
        status = ("CONFLICT" if "IDENTITY_CONFLICT" in reasons else
                  "UNVERIFIED" if "IDENTITY_UNVERIFIED" in reasons else
                  "CONFIRMED" if "IDENTITY_CONFIRMED" in reasons else None)
        refs = decision.identity_assessment_refs
        cited = set(decision.supporting_fact_ids + decision.observation_only_fact_ids)
        labels = []
        for fact_id in fact_ids:
            fact = by_id.get(fact_id) or {}
            provenance = fact.get("provenance") or {}
            if fact_id not in cited or provenance.get("identity_assessment_id") not in refs:
                continue
            label = _observed_label(fact)
            if label is None:
                continue
            row = {"evidence_id": fact_id, "label": label}
            if isinstance(fact.get("source_family"), str) and fact["source_family"] in SourceFamily._value2member_map_:
                row["source_family"] = fact["source_family"]
            labels.append(row)
        uncertainty.append({
            "requirement_id": decision.requirement_id, "support_level": decision.level.value,
            "identity_status": status, "identity_assessment_refs": list(refs),
            "episode_ids": sorted({episode for component in decision.component_decisions
                                   for episode in component.episode_ids}),
            "observed_labels": labels, "reason_codes": list(reasons),
        })
    return {
        "bundle_id": bundle.bundle_id,
        "video_id": bundle.video_id,
        "time_window": {"start": bundle.time_window.start, "end": bundle.time_window.end},
        "supported_requirement_ids": [identifier for identifier, decision in decisions.items()
                                      if decision.level is SupportLevel.FULL_SUPPORT],
        "conditional_requirement_ids": [identifier for identifier, decision in decisions.items()
                                        if decision.level is SupportLevel.CONDITIONAL_SUPPORT],
        "no_support_requirement_ids": [identifier for identifier, decision in decisions.items()
                                       if decision.level is SupportLevel.NO_SUPPORT],
        "missing_requirement_ids": list(bundle.missing_requirement_ids),
        "uncertainty": uncertainty,
        "facts": facts,
    }


def serialize_bundle(bundle: EvidenceBundle) -> str:
    return json.dumps(_bundle_payload(bundle), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _fact_ids(bundle: EvidenceBundle) -> tuple[str, ...]:
    return (
        bundle.scene_evidence + bundle.object_evidence + bundle.relation_evidence
        + bundle.track_evidence + bundle.temporal_evidence + bundle.event_evidence
    )


_ANCHOR_GATED_EVENT_TYPES = {
    "event", "visual_event", "entry", "exit", "entry_exit", "entry_exit_event",
}


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, str) and item}
    return set()


def _fact_tracks(fact: dict[str, Any]) -> set[str]:
    value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
    provenance = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else {}
    tracks = set()
    for container in (value, provenance):
        for key in (
            "track_id", "raw_track_id", "identity_target_id", "subject_raw_track_id",
            "subject_identity_target_id",
        ):
            tracks.update(_strings(container.get(key)))
    tracks.update(
        target for target in (fact.get("role_bindings") or {}).values()
        if isinstance(target, str) and target.startswith("track:")
    )
    return tracks


def _fact_episodes(fact: dict[str, Any]) -> set[str]:
    provenance = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else {}
    episodes = _strings(fact.get("episode_id"))
    for key in ("episode_id", "subject_episode_id", "anchor_episode_id"):
        episodes.update(_strings(provenance.get(key)))
    return episodes


def _fact_scopes(fact: dict[str, Any]) -> set[str]:
    value = fact.get("value") if isinstance(fact.get("value"), dict) else {}
    provenance = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else {}
    return _strings(value.get("scope_id")) | _strings(provenance.get("scope_id"))


def _fact_linkage_refs(fact: dict[str, Any]) -> set[str]:
    provenance = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else {}
    refs = _strings(fact.get("source_evidence_ids")) | _strings(fact.get("derived_from"))
    for key in (
        "anchor_event_id", "anchor_object_id", "linked_fact_id", "parent_fact_id",
        "source_event_id",
    ):
        refs.update(_strings(provenance.get(key)))
    return refs


def _anchor_relevant_fact_ids(bundle: EvidenceBundle) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gate event candidates against explicit anchors without changing retrieval."""
    all_ids = _fact_ids(bundle)
    payloads = {
        str(fact.get("fact_id") or fact.get("evidence_id")): fact
        for fact in bundle.provenance.get("visible_facts") or ()
        if isinstance(fact, dict) and (fact.get("fact_id") or fact.get("evidence_id"))
    }
    anchors = []
    for fact_id, fact in payloads.items():
        provenance = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else {}
        anchor_event_ids = _strings(provenance.get("anchor_event_id"))
        if not anchor_event_ids:
            continue
        anchors.append({
            "ids": {fact_id} | anchor_event_ids | _strings(provenance.get("anchor_object_id")),
            "tracks": (
                _strings(provenance.get("anchor_identity_target_id"))
                | _strings(provenance.get("identity_target_id"))
                | _strings(provenance.get("raw_track_id"))
                | _strings(provenance.get("subject_identity_target_id"))
                | _strings(provenance.get("subject_raw_track_id"))
            ),
            "episodes": _strings(provenance.get("anchor_episode_id")) | _fact_episodes(fact),
            "scopes": _fact_scopes(fact),
        })
    if not anchors:
        return all_ids, ()

    event_ids = set(bundle.event_evidence)
    event_ids.update(
        fact_id for fact_id, fact in payloads.items()
        if str(fact.get("fact_type") or "").casefold() in _ANCHOR_GATED_EVENT_TYPES
    )
    refs_by_id = {fact_id: _fact_linkage_refs(fact) for fact_id, fact in payloads.items()}

    def linked_to(anchor_ids: set[str], fact_id: str) -> bool:
        pending = [fact_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            refs = refs_by_id.get(current, set())
            if refs & anchor_ids:
                return True
            pending.extend(ref for ref in refs if ref in refs_by_id and ref not in visited)
        return False

    def relevant(fact_id: str) -> bool:
        fact = payloads.get(fact_id) or {}
        tracks = _fact_tracks(fact)
        episodes = _fact_episodes(fact)
        scopes = _fact_scopes(fact)
        return any(
            (bool(tracks & anchor["tracks"]))
            or (bool(scopes & anchor["scopes"]) and bool(episodes & anchor["episodes"]))
            or linked_to(anchor["ids"], fact_id)
            for anchor in anchors
        )

    excluded = tuple(fact_id for fact_id in all_ids if fact_id in event_ids and not relevant(fact_id))
    excluded_set = set(excluded)
    return tuple(fact_id for fact_id in all_ids if fact_id not in excluded_set), excluded


def _visible_bindings(
    requirements: QuestionRequirementSet,
    decisions: dict,
    facts: list[EvidenceFact],
) -> dict[str, SubjectBinding]:
    by_id = {fact.fact_id: fact for fact in facts}
    targets: dict[str, set[str]] = {}
    for requirement in requirements.requirements:
        decision = decisions.get(requirement.requirement_id)
        if decision is None or decision.level is not SupportLevel.FULL_SUPPORT:
            continue
        cited = set(decision.supporting_fact_ids)
        cited.update(fact_id for component in decision.component_decisions for fact_id in component.fact_ids)
        interaction = requirement.requirement_type is RequirementType.INTERACTION
        for role, generic in (
            (requirement.subject, "actor" if interaction else "subject"),
            (requirement.reference_object, "patient" if interaction else "reference"),
            (requirement.comparison_target, "comparison"),
        ):
            if role is None:
                continue
            for fact_id in sorted(cited):
                fact = by_id.get(fact_id)
                if fact is None:
                    continue
                target = fact.role_bindings.get(role.role_id) or fact.role_bindings.get(generic)
                if target:
                    targets.setdefault(role.role_id, set()).add(target)
    return {
        role_id: SubjectBinding(role_id, tuple(sorted(identifiers)))
        for role_id, identifiers in sorted(targets.items())
    }


def _rechecked_consistency(
    bundle: EvidenceBundle,
    requirements: QuestionRequirementSet,
    decisions: dict,
    visible_payloads: list[dict[str, Any]],
    prior: dict[str, Any],
) -> tuple[dict[str, Any], dict]:
    """Rebuild role bindings and checks from complete authoritative visible facts."""
    full_decisions = {
        identifier: decision for identifier, decision in decisions.items()
        if decision.level is SupportLevel.FULL_SUPPORT
    }
    needed = {
        fact_id
        for decision in full_decisions.values()
        for fact_id in (
            decision.supporting_fact_ids
            + tuple(item for component in decision.component_decisions for item in component.fact_ids)
        )
    }
    by_id = {
        str(payload.get("fact_id") or payload.get("evidence_id")): payload
        for payload in visible_payloads
        if isinstance(payload, dict) and (payload.get("fact_id") or payload.get("evidence_id"))
    }
    fact_fields = set(EvidenceFact.__dataclass_fields__)
    required_fields = fact_fields - {"video_id"}
    facts = []
    complete = bool(needed)
    for fact_id in sorted(needed):
        payload = by_id.get(fact_id)
        if payload is None or not required_fields <= set(payload):
            complete = False
            break
        value = {key: payload[key] for key in fact_fields if key in payload}
        value["video_id"] = (
            payload.get("video_id")
            or (payload.get("provenance") or {}).get("video_id")
            or bundle.video_id
        )
        try:
            facts.append(from_dict(EvidenceFact, value))
        except (TypeError, ValueError):
            complete = False
            break

    consistency = dict(prior)
    if complete:
        subjects = _visible_bindings(requirements, decisions, facts)
        report = _check_role_consistency_with_decisions(requirements, facts, subjects, decisions)
        consistency["role_checks"] = {
            identifier: {
                "consistent": check.consistent,
                "evidence_ids": list(check.evidence_ids),
                "warnings": list(check.warnings),
            }
            for identifier, check in report.role_checks.items()
        }
        consistency["relation_checks"] = {
            identifier: {
                "consistent": check.consistent,
                "evidence_ids": list(check.evidence_ids),
                "warnings": list(check.warnings),
            }
            for identifier, check in report.relation_checks.items()
        }
        return consistency, subjects

    required_roles = {
        role.role_id
        for requirement in requirements.requirements
        if requirement.requirement_id in full_decisions
        for role in (requirement.subject, requirement.reference_object, requirement.comparison_target)
        if role is not None
    }
    consistency["role_checks"] = {
        role_id: {
            "consistent": False,
            "evidence_ids": [],
            "warnings": ["visible_fact_payload_incomplete"],
        }
        for role_id in sorted(required_roles)
    }
    return consistency, dict(bundle.focal_subjects)


def _with_visible_facts(bundle: EvidenceBundle, retained: tuple[str, ...],
                        requirements: QuestionRequirementSet | None = None) -> EvidenceBundle:
    """Return a provenance-preserving bundle containing only complete retained facts."""
    retained_set = set(retained) & set(_fact_ids(bundle))
    retained = tuple(identifier for identifier in retained if identifier in retained_set)
    provenance = dict(bundle.provenance)
    by_id = {
        str(fact.get("fact_id") or fact.get("evidence_id")): fact
        for fact in provenance.get("visible_facts") or () if isinstance(fact, dict)
    }
    support = {
        requirement_id: [fact_id for fact_id in fact_ids if fact_id in retained_set]
        for requirement_id, fact_ids in (provenance.get("requirement_support") or {}).items()
    }
    support = {key: value for key, value in support.items() if value}
    supported = tuple(identifier for identifier in bundle.supported_requirement_ids if identifier in support)
    missing = tuple(dict.fromkeys(
        bundle.missing_requirement_ids
        + tuple(identifier for identifier in bundle.supported_requirement_ids if identifier not in supported)
    ))
    decisions = {}
    for identifier, decision in bundle.requirement_support_decisions.items():
        cited = set(decision.supporting_fact_ids + decision.observation_only_fact_ids)
        cited.update(fact_id for component in decision.component_decisions
                     for fact_id in component.fact_ids + component.observation_only_fact_ids)
        if cited <= retained_set:
            decisions[identifier] = decision
            continue
        components = []
        for component in decision.component_decisions:
            fact_ids = tuple(fact_id for fact_id in component.fact_ids if fact_id in retained_set)
            observations = tuple(fact_id for fact_id in component.observation_only_fact_ids
                                 if fact_id in retained_set)
            if fact_ids == component.fact_ids and observations == component.observation_only_fact_ids:
                components.append(component)
                continue
            level = component.level if fact_ids else SupportLevel.NO_SUPPORT
            comparison_alternative = component.component_id.endswith(":comparison") and any(
                (by_id.get(identifier) or {}).get("fact_type") == "comparison"
                and len(set(((by_id.get(identifier) or {}).get("role_bindings") or {}).values())) >= 2
                for identifier in fact_ids
            )
            if (component.component_id.endswith((":state_change", ":comparison"))
                    and fact_ids != component.fact_ids and not comparison_alternative):
                # These component citations describe a mandatory pair. Retaining
                # one endpoint cannot preserve the original paired decision.
                level = SupportLevel.NO_SUPPORT
            reasons = component.rationale_codes
            if (":identity:" in component.component_id and level is SupportLevel.FULL_SUPPORT
                    and not _retained_identity_confirmed(component, by_id, retained_set)):
                level = SupportLevel.CONDITIONAL_SUPPORT
                reasons = tuple(reason for reason in reasons if reason != "IDENTITY_CONFIRMED") + (
                    "IDENTITY_UNVERIFIED",
                )
            components.append(replace(
                component, level=level, fact_ids=fact_ids, observation_only_fact_ids=observations,
                rationale_codes=reasons + (
                    ("EVIDENCE_REMOVED_FOR_BUDGET",) if level is not component.level else ()),
            ))
        rank = {SupportLevel.NO_SUPPORT: 0, SupportLevel.CONDITIONAL_SUPPORT: 1, SupportLevel.FULL_SUPPORT: 2}
        level = min((component.level for component in components), key=rank.get) if components else decision.level
        # Compaction cannot strengthen an existing no/conditional decision.
        level = min((level, decision.level), key=rank.get)
        fact_ids = tuple(dict.fromkeys(fact_id for component in components for fact_id in component.fact_ids)) \
            if components else tuple(fact_id for fact_id in decision.supporting_fact_ids if fact_id in retained_set)
        if not fact_ids:
            level = SupportLevel.NO_SUPPORT
        supporting_ids = fact_ids if level is not SupportLevel.NO_SUPPORT else ()
        observations = tuple(dict.fromkeys(
            tuple(fact_id for fact_id in decision.supporting_fact_ids + decision.observation_only_fact_ids
                  if fact_id in retained_set and fact_id not in supporting_ids)
            + tuple(fact_id for component in components
                    for fact_id in component.observation_only_fact_ids + component.fact_ids
                    if fact_id not in supporting_ids)
        ))
        reasons = tuple(dict.fromkeys(reason for component in components for reason in component.rationale_codes)) \
            if components else decision.rationale_codes + (("EVIDENCE_REMOVED_FOR_BUDGET",)
                                                          if level is not decision.level else ())
        decisions[identifier] = replace(
            decision, level=level, supporting_fact_ids=supporting_ids, observation_only_fact_ids=observations,
            component_decisions=tuple(components), rationale_codes=reasons,
        )
    conditional = bundle.conditional_requirement_ids
    no_support = bundle.no_support_requirement_ids
    if decisions:
        supported = tuple(identifier for identifier, decision in decisions.items()
                          if decision.level is SupportLevel.FULL_SUPPORT)
        conditional = tuple(identifier for identifier, decision in decisions.items()
                            if decision.level is SupportLevel.CONDITIONAL_SUPPORT)
        no_support = tuple(identifier for identifier, decision in decisions.items()
                           if decision.level is SupportLevel.NO_SUPPORT)
        support = {identifier: list(decisions[identifier].supporting_fact_ids) for identifier in supported}
        if requirements is None:
            missing = tuple(dict.fromkeys(
                bundle.missing_requirement_ids
                + tuple(identifier for identifier in bundle.supported_requirement_ids if identifier not in supported)
            ))
    if requirements is not None:
        _, missing = _logical_support(requirements, set(supported))
    provenance["fact_refs"] = [fact_id for fact_id in provenance.get("fact_refs", ()) if fact_id in retained_set]
    provenance["visible_facts"] = [
        fact for fact in provenance.get("visible_facts", ())
        if str(fact.get("fact_id") or fact.get("evidence_id")) in retained_set
    ]
    for key in ("fact_sources", "fact_videos", "fact_semantic_keys"):
        provenance[key] = {
            fact_id: value for fact_id, value in (provenance.get(key) or {}).items()
            if fact_id in retained_set
        }
    provenance["requirement_support"] = support
    provenance["serialization_compaction"] = {
        "retained_fact_ids": list(retained),
        "omitted_fact_ids": [fact_id for fact_id in _fact_ids(bundle) if fact_id not in retained_set],
    }
    consistency = dict(bundle.consistency)
    consistency["role_checks"] = {
        role_id: {
            **check,
            "evidence_ids": [fact_id for fact_id in check.get("evidence_ids", ()) if fact_id in retained_set],
        }
        for role_id, check in (consistency.get("role_checks") or {}).items()
    }
    consistency["relation_checks"] = {
        requirement_id: {
            **check,
            "evidence_ids": [fact_id for fact_id in check.get("evidence_ids", ()) if fact_id in retained_set],
        }
        for requirement_id, check in (consistency.get("relation_checks") or {}).items()
    }
    focal_subjects = dict(bundle.focal_subjects)
    if requirements is not None and decisions:
        consistency, focal_subjects = _rechecked_consistency(
            bundle,
            requirements,
            decisions,
            provenance["visible_facts"],
            consistency,
        )
    groups = {
        name: tuple(fact_id for fact_id in getattr(bundle, name) if fact_id in retained_set)
        for name in (
            "scene_evidence", "object_evidence", "relation_evidence", "track_evidence",
            "temporal_evidence", "event_evidence",
        )
    }
    token_cost = math.ceil(len(json.dumps(provenance["visible_facts"], ensure_ascii=False,
                                          sort_keys=True, separators=(",", ":"))) / 4)
    compact = replace(
        bundle, supported_requirement_ids=supported, missing_requirement_ids=missing,
        requirement_support_decisions=decisions, conditional_requirement_ids=conditional,
        no_support_requirement_ids=no_support,
        focal_subjects=focal_subjects, provenance=provenance, consistency=consistency,
        token_cost=token_cost, **groups,
    )
    # The schema cannot infer satisfied OR or optional-only logic.
    object.__setattr__(compact, "missing_requirement_ids", missing)
    return compact


def _apply_anchor_relevance_gate(
    bundle: EvidenceBundle,
    requirements: QuestionRequirementSet | None,
) -> tuple[EvidenceBundle, tuple[str, ...]]:
    retained, excluded = _anchor_relevant_fact_ids(bundle)
    if not excluded:
        return bundle, ()
    gated = _with_visible_facts(bundle, retained, requirements)
    provenance = dict(gated.provenance)
    provenance.pop("serialization_compaction", None)
    provenance["anchor_relevance_gate"] = {
        "retained_fact_ids": list(retained),
        "excluded_fact_ids": list(excluded),
    }
    return replace(gated, provenance=provenance), excluded


def _candidate_user(question: str, bundles: list[EvidenceBundle], assessment: SupportAssessment) -> str:
    supported = tuple(dict.fromkeys(
        identifier for bundle in bundles for identifier in bundle.supported_requirement_ids
    ))
    missing = tuple(dict.fromkeys(
        assessment.missing_requirement_ids
        + tuple(identifier for identifier in assessment.supported_requirement_ids
                if identifier not in supported)
    ))
    return json.dumps(
        _user_payload(question, [_bundle_payload(bundle) for bundle in bundles], assessment,
                      supported, missing),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _atomic_fact_group(fact_id: str, by_id: dict[str, dict[str, Any]], known: set[str]) -> tuple[str, ...]:
    fact = by_id.get(fact_id) or {}
    kind = str(fact.get("fact_type") or "")
    if kind not in {
        "left_of", "right_of", "above", "below", "overlap", "image_near",
        "located_in_region", "interaction", "before_state", "after_state",
        "scene_state_before", "scene_state_after", "state_change",
    }:
        return (fact_id,)
    dependencies = [str(value) for value in fact.get("source_evidence_ids") or () if str(value) in known]
    return tuple(dict.fromkeys((*dependencies, fact_id)))


def _compact_bundle(question: str, prior: list[EvidenceBundle], bundle: EvidenceBundle,
                    assessment: SupportAssessment, context_limit_chars: int) -> EvidenceBundle | None:
    all_ids = _fact_ids(bundle)
    known = set(all_ids)
    by_id = {
        str(fact.get("fact_id") or fact.get("evidence_id")): fact
        for fact in bundle.provenance.get("visible_facts") or ()
        if isinstance(fact, dict) and (fact.get("fact_id") or fact.get("evidence_id"))
    }
    support = bundle.provenance.get("requirement_support") or {}
    support_lists = [
        [fact_id for fact_id in support.get(requirement_id, ()) if fact_id in known]
        for requirement_id in bundle.supported_requirement_ids
    ]
    # Cover every supported requirement once before spending budget on additional
    # observations for an already covered requirement.  This keeps compact
    # temporal/relation evidence from being crowded out by detector repetition.
    required = [
        fact_ids[offset]
        for offset in range(max((len(fact_ids) for fact_ids in support_lists), default=0))
        for fact_ids in support_lists
        if offset < len(fact_ids)
    ]
    priority_groups = (
        (bundle.relation_evidence, bundle.temporal_evidence, bundle.event_evidence,
         bundle.object_evidence, bundle.scene_evidence, bundle.track_evidence)
    )
    ordered = list(dict.fromkeys(required + [fact_id for group in priority_groups for fact_id in group]))
    semantic = bundle.provenance.get("fact_semantic_keys") or {}
    retained: list[str] = []
    seen_semantic: set[str] = set()
    required_set = set(required)
    for fact_id in ordered:
        semantic_key = str(semantic.get(fact_id) or fact_id)
        if fact_id not in required_set and semantic_key in seen_semantic:
            continue
        group = _atomic_fact_group(fact_id, by_id, known)
        candidate_ids = tuple(dict.fromkeys((*retained, *group)))
        candidate = _with_visible_facts(bundle, candidate_ids, assessment._requirements)
        if len(_candidate_user(question, prior + [candidate], assessment)) > context_limit_chars:
            continue
        retained = list(candidate_ids)
        seen_semantic.add(semantic_key)
    return _with_visible_facts(bundle, tuple(retained), assessment._requirements) if retained else None


def _merged_full_decisions(bundles: list[EvidenceBundle]) -> dict:
    grouped = {}
    for bundle in bundles:
        for identifier, decision in bundle.requirement_support_decisions.items():
            if decision.level is SupportLevel.FULL_SUPPORT:
                grouped.setdefault(identifier, []).append(decision)
    merged = {}
    for identifier, candidates in grouped.items():
        candidates = sorted(
            candidates,
            key=lambda item: (
                item.supporting_fact_ids,
                item.observation_only_fact_ids,
                tuple((component.component_id, component.fact_ids)
                      for component in item.component_decisions),
            ),
        )
        merged[identifier] = replace(
            candidates[0],
            supporting_fact_ids=tuple(sorted({
                fact_id for item in candidates for fact_id in item.supporting_fact_ids
            })),
            observation_only_fact_ids=tuple(sorted({
                fact_id for item in candidates for fact_id in item.observation_only_fact_ids
            })),
            component_decisions=tuple(sorted(
                {component for item in candidates for component in item.component_decisions},
                key=lambda component: (
                    component.component_id, component.fact_ids,
                    component.observation_only_fact_ids,
                ),
            )),
            identity_assessment_refs=tuple(sorted({
                ref for item in candidates for ref in item.identity_assessment_refs
            })),
            provenance_root_ids=tuple(sorted({
                root for item in candidates for root in item.provenance_root_ids
            })),
            rationale_codes=tuple(sorted({
                code for item in candidates for code in item.rationale_codes
            })),
        )
    return merged


def _authoritative_union_role_consistent(
    requirements: QuestionRequirementSet,
    bundles: list[EvidenceBundle],
) -> bool | None:
    """Recheck one Boolean selection over every retained authoritative fact."""
    mandatory = {
        item.requirement_id for item in requirements.requirements
        if item.optionality is Optionality.MANDATORY
    }
    decisions = {
        identifier: decision
        for identifier, decision in _merged_full_decisions(bundles).items()
        if identifier in mandatory
    }
    if not decisions or not _logical_support(requirements, set(decisions))[0]:
        return None
    needed = {
        fact_id
        for decision in decisions.values()
        for fact_id in (
            decision.supporting_fact_ids
            + tuple(fact_id for component in decision.component_decisions
                    for fact_id in component.fact_ids)
        )
    }
    payloads = {}
    for bundle in bundles:
        for payload in bundle.provenance.get("visible_facts") or ():
            if not isinstance(payload, dict):
                continue
            fact_id = payload.get("fact_id") or payload.get("evidence_id")
            if not isinstance(fact_id, str) or not fact_id:
                continue
            row = (payload, bundle.video_id)
            if fact_id in payloads and payloads[fact_id] != row:
                return False
            payloads[fact_id] = row

    fact_fields = set(EvidenceFact.__dataclass_fields__)
    required_fields = fact_fields - {"video_id"}
    facts = []
    for fact_id in sorted(needed):
        row = payloads.get(fact_id)
        if row is None:
            return False
        payload, bundle_video_id = row
        if not required_fields <= set(payload):
            return False
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            return False
        value = {key: payload[key] for key in fact_fields if key in payload}
        value["video_id"] = (
            payload.get("video_id")
            or provenance.get("video_id")
            or bundle_video_id
        )
        try:
            facts.append(from_dict(EvidenceFact, value))
        except (AttributeError, TypeError, ValueError):
            return False

    selected = _selected_requirement_ids(requirements, decisions, facts)
    if selected is None:
        return False
    bindings = _visible_bindings(requirements, decisions, facts)
    try:
        report = _check_role_consistency_with_decisions(requirements, facts, bindings, decisions)
    except ValueError:
        return False
    required_roles = {
        role.role_id
        for requirement in requirements.requirements
        if requirement.requirement_id in selected
        for role in (requirement.subject, requirement.reference_object, requirement.comparison_target)
        if role is not None
    }
    if any(
        role_id not in bindings
        or report.role_checks.get(role_id) is None
        or report.role_checks[role_id].consistent is not True
        for role_id in required_roles
    ):
        return False
    relation_ids = {
        requirement.requirement_id
        for requirement in requirements.requirements
        if requirement.requirement_id in selected
        and requirement.requirement_type in {RequirementType.SPATIAL_RELATION, RequirementType.INTERACTION}
    }
    return all(
        report.relation_checks.get(identifier) is not None
        and report.relation_checks[identifier].consistent is True
        for identifier in relation_ids
    )


def _compacted_assessment(assessment: SupportAssessment, bundles: list[EvidenceBundle]) -> SupportAssessment:
    """Refresh decision counts from the bundles that remain visible."""
    rank = {SupportLevel.NO_SUPPORT: 0, SupportLevel.CONDITIONAL_SUPPORT: 1, SupportLevel.FULL_SUPPORT: 2}
    levels = {}
    for bundle in bundles:
        decisions = bundle.requirement_support_decisions.values() if bundle.requirement_support_decisions \
            else project_legacy_support(bundle)
        for decision in decisions:
            prior = levels.get(decision.requirement_id, SupportLevel.NO_SUPPORT)
            levels[decision.requirement_id] = max((prior, decision.level), key=rank.get)
    requirements = assessment._requirements
    identifiers = tuple(item.requirement_id for item in requirements.requirements) if requirements else \
        tuple(assessment.requirement_descriptions)
    full = tuple(identifier for identifier in identifiers if levels.get(identifier) is SupportLevel.FULL_SUPPORT)
    conditional = tuple(identifier for identifier in identifiers
                        if levels.get(identifier) is SupportLevel.CONDITIONAL_SUPPORT)
    none = tuple(identifier for identifier in identifiers if identifier not in full + conditional)
    mandatory = {item.requirement_id for item in requirements.requirements
                 if item.optionality is Optionality.MANDATORY} if requirements else set(identifiers)
    all_supported, missing = _logical_support(requirements, set(full)) if requirements else (
        not none and not conditional, none + conditional,
    )
    compacted = [bundle for bundle in bundles if "serialization_compaction" in bundle.provenance]
    role_consistent = assessment.role_consistent
    if requirements is not None and compacted:
        role_consistent = role_consistent and all(
            _role_consistent(requirements, bundle, set(bundle.supported_requirement_ids))
            for bundle in compacted
        )
        union_consistent = _authoritative_union_role_consistent(requirements, bundles)
        if union_consistent is not None:
            role_consistent = role_consistent and union_consistent
    return replace(
        assessment, supported_requirement_ids=full, conditional_requirement_ids=conditional,
        no_support_requirement_ids=none, full_support_count=len(full), conditional_support_count=len(conditional),
        no_support_count=len(none), supported_mandatory_count=len(mandatory & set(full)),
        all_mandatory_supported=all_supported, missing_requirement_ids=missing,
        role_consistent=role_consistent,
        _required_support_present=bool(mandatory & set(full + conditional)),
    )


def _user_payload(question: str, bundles: list[dict[str, Any]], assessment: SupportAssessment,
                  supported_ids: tuple[str, ...], missing_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "question": question,
        "instruction": ANSWER_INSTRUCTION,
        "supported_requirements": [
            {"id": identifier, "description": assessment.requirement_descriptions.get(identifier, identifier)}
            for identifier in supported_ids
        ],
        "missing_requirements": [
            {"id": identifier, "description": assessment.requirement_descriptions.get(identifier, identifier)}
            for identifier in missing_ids
        ],
        "evidence_bundles": bundles,
    }


def build_visible_context(question: str, bundles: list[EvidenceBundle],
                          assessment: SupportAssessment, *, context_limit_chars: int,
                          token_counter: Callable[[list[dict]], int]) -> VisibleContext:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if type(context_limit_chars) is not int or context_limit_chars < 0:
        raise ValueError("context_limit_chars must be nonnegative")
    visible_bundles: list[EvidenceBundle] = []
    visible_payloads: list[dict[str, Any]] = []
    omitted: list[EvidenceBundle] = []
    compacted = []
    omitted_facts: dict[str, list[str]] = {}
    anchor_excluded_facts: dict[str, list[str]] = {}
    for index, bundle in enumerate(bundles):
        bundle, anchor_excluded = _apply_anchor_relevance_gate(bundle, assessment._requirements)
        if anchor_excluded:
            anchor_excluded_facts[bundle.bundle_id] = list(anchor_excluded)
        compacted.append(bundle.bundle_id)
        candidate_payload = _bundle_payload(bundle)
        candidate_bundles = visible_payloads + [candidate_payload]
        candidate_supported = tuple(dict.fromkeys(
            identifier for item in visible_bundles + [bundle]
            for identifier in item.supported_requirement_ids
        ))
        candidate_missing = tuple(identifier for identifier in assessment.missing_requirement_ids
                                  if identifier not in candidate_supported)
        candidate_user = json.dumps(
            _user_payload(question, candidate_bundles, assessment, candidate_supported, candidate_missing),
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        if len(candidate_user) > context_limit_chars:
            compact = _compact_bundle(
                question, visible_bundles, bundle, assessment, context_limit_chars,
            )
            if compact is None:
                omitted.extend(bundles[index:])
                break
            bundle = compact
            candidate_payload = _bundle_payload(bundle)
            if len(_candidate_user(question, visible_bundles + [bundle], assessment)) > context_limit_chars:
                omitted.extend(bundles[index:])
                break
            omitted_facts[bundle.bundle_id] = list(
                bundle.provenance.get("serialization_compaction", {}).get("omitted_fact_ids", ())
            )
        visible_bundles.append(bundle)
        visible_payloads.append(candidate_payload)
    if anchor_excluded_facts or omitted_facts or omitted:
        assessment = _compacted_assessment(assessment, visible_bundles)
    visible_supported = tuple(dict.fromkeys(
        identifier for bundle in visible_bundles for identifier in bundle.supported_requirement_ids
    ))
    visible_missing = tuple(dict.fromkeys(
        assessment.missing_requirement_ids
        + tuple(identifier for identifier in assessment.supported_requirement_ids
                if identifier not in visible_supported)
    ))
    user_content = json.dumps(
        _user_payload(question, visible_payloads, assessment, visible_supported, visible_missing),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    messages = (
        {"role": "system", "content": ANSWER_INSTRUCTION},
        {"role": "user", "content": user_content},
    )
    actual_tokens = token_counter(list(messages))
    if type(actual_tokens) is not int or actual_tokens < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    estimated = math.ceil(sum(len(message["content"]) for message in messages) / 4)
    fits = not omitted and len(user_content) <= context_limit_chars
    reason = "budget_infeasible" if omitted else ("compacted_to_fit" if omitted_facts else "fits")
    budget = BudgetStatus(
        bundle_fits=fits,
        visible_bundle_ids=tuple(bundle.bundle_id for bundle in visible_bundles),
        omitted_bundle_ids=tuple(bundle.bundle_id for bundle in omitted),
        actual_message_tokens=actual_tokens,
        estimated_message_tokens=estimated,
        reason=reason,
    )
    assessment = replace(
        assessment,
        provenance_complete=(
            assessment.provenance_complete
            and all(_provenance_complete(bundle) for bundle in visible_bundles)
        ),
    )
    final_assessment = finalize_budget(
        assessment, budget_status=budget, visible_supported_ids=visible_supported,
    )
    # Report the same final visible decisions for counts and audit, after repair.
    # Multiple bundles may offer alternatives for one requirement; count it once.
    visible_decisions = {}
    all_visible_decisions = []
    rank = {SupportLevel.NO_SUPPORT: 0, SupportLevel.CONDITIONAL_SUPPORT: 1, SupportLevel.FULL_SUPPORT: 2}
    for bundle in visible_bundles:
        decisions = bundle.requirement_support_decisions.values() if bundle.requirement_support_decisions \
            else project_legacy_support(bundle)
        # Legacy boolean projections have no authoritative component certificates.
        all_visible_decisions.extend(bundle.requirement_support_decisions.values())
        for decision in decisions:
            prior = visible_decisions.get(decision.requirement_id)
            if prior is None or rank[decision.level] > rank[prior.level]:
                visible_decisions[decision.requirement_id] = decision
    identifiers = tuple(assessment.requirement_descriptions)
    levels = [visible_decisions[identifier].level if identifier in visible_decisions else SupportLevel.NO_SUPPORT
              for identifier in identifiers]
    support_counts = {
        "full_support_count": levels.count(SupportLevel.FULL_SUPPORT),
        "conditional_support_count": levels.count(SupportLevel.CONDITIONAL_SUPPORT),
        "no_support_count": levels.count(SupportLevel.NO_SUPPORT),
    }
    return VisibleContext(
        messages=messages, user_content=user_content,
        visible_bundle_ids=budget.visible_bundle_ids, omitted_bundle_ids=budget.omitted_bundle_ids,
        budget_status=budget, assessment=final_assessment,
        trace={
            "support_counts": support_counts,
            "safety_audit": audit_support_safety(all_visible_decisions),
            "context_limit_chars": context_limit_chars,
            "context_chars_visible": len(user_content),
            "actual_message_tokens": actual_tokens,
            "estimated_message_tokens": estimated,
            "compacted_once": compacted,
            "visible_bundle_ids": list(budget.visible_bundle_ids),
            "omitted_bundle_ids": list(budget.omitted_bundle_ids),
            "visible_fact_ids_by_bundle": {
                bundle.bundle_id: list(_fact_ids(bundle)) for bundle in visible_bundles
            },
            "omitted_fact_ids_by_bundle": omitted_facts,
            "anchor_relevance_excluded_fact_ids_by_bundle": anchor_excluded_facts,
            "stop_reason": budget.reason,
        },
    )
