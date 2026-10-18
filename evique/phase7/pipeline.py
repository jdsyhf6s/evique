from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .bundle_builder import build_candidate_bundles
from .completion import CompletionResult, CompletionSource, complete_bundles
from .config import Phase7Config
from .normalization import (
    DEFAULT_NORMALIZATION_POLICY, NormalizationPolicy, NormalizationResult, Normalizer,
    normalize_and_fuse,
)
from .requirement_parser import RequirementParseClient, parse_requirements
from .role_consistency import _check_role_consistency_with_decisions
from .schema import (
    EvidenceBundle,
    EvidenceFact,
    QuestionRequirementSet,
    RequirementSupportDecision,
    SubjectBinding,
    SupportLevel,
    TimeRange,
    from_dict,
)
from .serializer import VisibleContext, build_visible_context
from .support import SupportAssessment, _logical_support, assess_support
from .utility import select_bundles


LEGACY_FLAGS = ("legacy_raw_packing", "legacy_graph_expansion", "legacy_cost_pruning")
FORBIDDEN_RUNTIME_KEYS = {"reference_requirements", "annotation_path", "ground_truth", "answer", "answers"}


class FactProvider(Protocol):
    def get_facts(self, requirements: QuestionRequirementSet, video_id: str,
                  shared_index: Any) -> list[EvidenceFact]: ...


class VisualCompletionAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class Phase7Package:
    query: str
    video_id: str
    requirements: QuestionRequirementSet
    initial_fact_ids: tuple[str, ...]
    normalization: NormalizationResult
    bundles: tuple[EvidenceBundle, ...]
    completion: CompletionResult
    support: SupportAssessment
    visible_context: VisibleContext
    trace: dict[str, Any]


def _forbidden_keys(value: Any, path: str = "metadata") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_RUNTIME_KEYS or "annotation" in normalized or "ground_truth" in normalized:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, f"{path}[{index}]"))
    return found


def _unique(values):
    return tuple(dict.fromkeys(values))


def _nonempty_unique_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    identifiers = tuple(value)
    if (any(not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        return None
    return identifiers


def _merge_consistency(requirements: QuestionRequirementSet, bundles: list[EvidenceBundle],
                       subjects: dict[str, SubjectBinding],
                       decisions: dict[str, RequirementSupportDecision]) -> dict[str, Any]:
    role_checks: dict[str, dict] = {}
    relation_checks: dict[str, dict] = {}
    role_contributors: dict[str, set[str]] = {}
    role_evidence: dict[str, set[str]] = {}
    role_warnings: dict[str, set[str]] = {}
    for bundle in bundles:
        for requirement in requirements.requirements:
            decision = bundle.requirement_support_decisions.get(requirement.requirement_id)
            if decision is None or decision.level is not SupportLevel.FULL_SUPPORT:
                continue
            evidence_ids = set(decision.supporting_fact_ids)
            evidence_ids.update(
                fact_id for component in decision.component_decisions
                for fact_id in component.fact_ids
            )
            for role in (
                requirement.subject, requirement.reference_object, requirement.comparison_target,
            ):
                if role is None:
                    continue
                role_contributors.setdefault(role.role_id, set()).add(bundle.bundle_id)
                role_evidence.setdefault(role.role_id, set()).update(evidence_ids)
                role_warnings.setdefault(role.role_id, set())
        for destination, key in ((role_checks, "role_checks"), (relation_checks, "relation_checks")):
            for identifier, check in (bundle.consistency.get(key) or {}).items():
                if key == "role_checks":
                    role_contributors.setdefault(identifier, set()).add(bundle.bundle_id)
                    role_evidence.setdefault(identifier, set()).update(check.get("evidence_ids") or ())
                    role_warnings.setdefault(identifier, set()).update(check.get("warnings") or ())
                previous = destination.get(identifier)
                if previous is None or previous.get("consistent") is None:
                    destination[identifier] = check
                elif check.get("consistent") is False:
                    destination[identifier] = check
    temporal_values = [bundle.consistency.get("temporal_consistency") for bundle in bundles
                       if "temporal_consistency" in bundle.consistency]
    visible = {}
    full_decisions = [
        decision for bundle in bundles
        for decision in bundle.requirement_support_decisions.values()
        if decision.level is SupportLevel.FULL_SUPPORT
    ]
    certificates = [
        tuple(dict.fromkeys(
            decision.supporting_fact_ids
            + decision.observation_only_fact_ids
            + tuple(identifier for component in decision.component_decisions
                    for identifier in component.fact_ids + component.observation_only_fact_ids)
        ))
        for decision in full_decisions
    ]
    recheckable = not any(not certificate for certificate in certificates)
    required_fact_fields = {
        "fact_id", "fact_type", "time_span", "frame_ids", "role_bindings", "value",
        "source_evidence_ids", "source_kind", "source_family", "derived_from",
        "independence_root_ids", "episode_id", "derivation_rule", "provenance",
    }
    fact_fields = set(EvidenceFact.__dataclass_fields__)
    authoritative_fact_ids = {
        fact_id
        for bundle in bundles
        for decision in bundle.requirement_support_decisions.values()
        if decision.level is SupportLevel.FULL_SUPPORT
        for fact_id in (
            *decision.supporting_fact_ids,
            *decision.observation_only_fact_ids,
            *(identifier for component in decision.component_decisions
              for identifier in component.fact_ids + component.observation_only_fact_ids),
        )
    }
    payloads_by_id: dict[str, list[dict]] = {identifier: [] for identifier in authoritative_fact_ids}
    for bundle in sorted(bundles, key=lambda item: item.bundle_id):
        payloads = bundle.provenance.get("visible_facts")
        fact_sources = bundle.provenance.get("fact_sources")
        if not isinstance(payloads, (list, tuple)):
            recheckable = False
            continue
        for payload in payloads:
            if not isinstance(payload, dict) or not isinstance(payload.get("fact_id"), str):
                continue
            if payload["fact_id"] not in authoritative_fact_ids:
                continue
            if (not required_fact_fields <= set(payload) or not set(payload) <= fact_fields
                    or not isinstance(payload.get("provenance"), dict)
                    or payload["provenance"].get("video_id") != bundle.video_id
                    or payload.get("video_id", bundle.video_id) != bundle.video_id):
                recheckable = False
                continue
            payload_sources = _nonempty_unique_ids(payload.get("source_evidence_ids"))
            mapped_sources = (
                _nonempty_unique_ids(fact_sources.get(payload["fact_id"]))
                if isinstance(fact_sources, dict) else None
            )
            if payload_sources is None or mapped_sources is None or payload_sources != mapped_sources:
                recheckable = False
                continue
            canonical = {
                **payload,
                "video_id": payload.get("video_id")
                or (payload.get("provenance") or {}).get("video_id")
                or bundle.video_id,
            }
            payloads_by_id[payload["fact_id"]].append(canonical)
    for identifier, payloads in payloads_by_id.items():
        if not payloads or any(payload != payloads[0] for payload in payloads[1:]):
            recheckable = False
            continue
        visible[identifier] = payloads[0]
    if visible and recheckable:
        try:
            facts = [from_dict(EvidenceFact, visible[identifier]) for identifier in sorted(visible)]
            report = _check_role_consistency_with_decisions(requirements, facts, subjects, decisions)
        except (AttributeError, KeyError, TypeError, ValueError):
            recheckable = False
        else:
            for identifier, check in report.role_checks.items():
                prior = role_checks.get(identifier) or {}
                role_checks[identifier] = {
                    "consistent": False if prior.get("consistent") is False else check.consistent,
                    "evidence_ids": sorted(set(prior.get("evidence_ids") or ()) | set(check.evidence_ids)),
                    "warnings": list(dict.fromkeys(
                        list(prior.get("warnings") or ()) + list(check.warnings)
                    )),
                }
            relation_checks.update({
                identifier: {
                    "consistent": check.consistent,
                    "evidence_ids": list(check.evidence_ids),
                    "warnings": list(check.warnings),
                }
                for identifier, check in report.relation_checks.items()
            })
    if not recheckable:
        for identifier in role_contributors:
            prior = role_checks.get(identifier, {})
            role_checks[identifier] = {
                "consistent": False if prior.get("consistent") is False else None,
                "evidence_ids": sorted(role_evidence[identifier]),
                "warnings": sorted(role_warnings[identifier] | {"aggregate_visibility_incomplete"}),
            }
    return {
        "role_checks": role_checks,
        "relation_checks": relation_checks,
        "temporal_consistency": all(value is True for value in temporal_values) if temporal_values else None,
        "authoritative_visible_facts_consistent": recheckable,
    }


def _merge_requirement_decisions(requirements: QuestionRequirementSet,
                                 bundles: list[EvidenceBundle]) -> dict[str, RequirementSupportDecision]:
    merged = {}
    for requirement in requirements.requirements:
        identifier = requirement.requirement_id
        children = [bundle.requirement_support_decisions[identifier] for bundle in bundles
                    if identifier in bundle.requirement_support_decisions]
        levels = {decision.level for decision in children}
        if SupportLevel.FULL_SUPPORT in levels:
            level = SupportLevel.FULL_SUPPORT
        elif SupportLevel.CONDITIONAL_SUPPORT in levels:
            level = SupportLevel.CONDITIONAL_SUPPORT
        else:
            level = SupportLevel.NO_SUPPORT

        def union(field):
            return tuple(sorted({value for decision in children for value in getattr(decision, field)}))

        # Retain the child component records, including their original levels.
        # Co-occurrence never composes components into a stronger requirement.
        components = tuple(sorted({component for decision in children for component in decision.component_decisions},
                                  key=lambda component: (
                                      component.component_id, component.level.value, component.fact_ids,
                                      component.observation_only_fact_ids, component.identity_assessment_refs,
                                      component.episode_ids, component.provenance_root_ids, component.rationale_codes,
                                  )))
        merged[identifier] = RequirementSupportDecision(
            requirement_id=identifier, level=level,
            supporting_fact_ids=union("supporting_fact_ids"),
            observation_only_fact_ids=union("observation_only_fact_ids"),
            component_decisions=components,
            identity_assessment_refs=union("identity_assessment_refs"),
            provenance_root_ids=union("provenance_root_ids"),
            rationale_codes=union("rationale_codes") if children else ("NO_SELECTED_SUPPORT_DECISION",),
        )
    return merged


def _aggregate(requirements: QuestionRequirementSet,
               bundles: list[EvidenceBundle]) -> EvidenceBundle | None:
    if not bundles:
        return None
    videos = {bundle.video_id for bundle in bundles}
    if len(videos) != 1:
        raise ValueError("selected Phase 7 bundles cannot mix videos")
    video_id = next(iter(videos))
    decisions = _merge_requirement_decisions(requirements, bundles)
    supported = tuple(identifier for identifier, decision in decisions.items()
                      if decision.level is SupportLevel.FULL_SUPPORT)
    conditional = tuple(identifier for identifier, decision in decisions.items()
                        if decision.level is SupportLevel.CONDITIONAL_SUPPORT)
    no_support = tuple(identifier for identifier, decision in decisions.items()
                       if decision.level is SupportLevel.NO_SUPPORT)
    _, missing = _logical_support(requirements, set(supported))
    subjects: dict[str, SubjectBinding] = {}
    for bundle in bundles:
        for role_id, binding in bundle.focal_subjects.items():
            if role_id in subjects:
                prior = subjects[role_id]
                subjects[role_id] = SubjectBinding(
                    role_id, _unique(prior.subject_ids + binding.subject_ids),
                    prior.track_id if prior.track_id == binding.track_id else None,
                )
            else:
                subjects[role_id] = binding
    provenance = {"video_id": video_id, "fact_refs": [], "fact_videos": {},
                  "fact_sources": {}, "visible_facts": [],
                  "requirement_support": {identifier: list(decisions[identifier].supporting_fact_ids)
                                          for identifier in supported}}
    for bundle in bundles:
        provenance["fact_refs"].extend(bundle.provenance.get("fact_refs") or ())
        provenance["fact_videos"].update(bundle.provenance.get("fact_videos") or {})
        provenance["fact_sources"].update(bundle.provenance.get("fact_sources") or {})
        for payload in bundle.provenance.get("visible_facts") or ():
            if not isinstance(payload, dict):
                continue
            payload_provenance = payload.get("provenance")
            canonical = {
                **payload,
                "video_id": payload.get("video_id")
                or (payload_provenance.get("video_id")
                    if isinstance(payload_provenance, dict) else None)
                or bundle.video_id,
            }
            try:
                fact = from_dict(EvidenceFact, canonical)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            provenance["visible_facts"].append(_fact_payload(fact))
    provenance["fact_refs"] = list(dict.fromkeys(provenance["fact_refs"]))
    digest = hashlib.sha256(json.dumps([bundle.bundle_id for bundle in bundles]).encode()).hexdigest()[:16]
    aggregate = EvidenceBundle(
        bundle_id=f"aggregate:{digest}", video_id=video_id,
        time_window=TimeRange(min(bundle.time_window.start for bundle in bundles),
                              max(bundle.time_window.end for bundle in bundles)),
        focal_subjects=subjects,
        scene_evidence=_unique(fact for bundle in bundles for fact in bundle.scene_evidence),
        object_evidence=_unique(fact for bundle in bundles for fact in bundle.object_evidence),
        relation_evidence=_unique(fact for bundle in bundles for fact in bundle.relation_evidence),
        track_evidence=_unique(fact for bundle in bundles for fact in bundle.track_evidence),
        temporal_evidence=_unique(fact for bundle in bundles for fact in bundle.temporal_evidence),
        event_evidence=_unique(fact for bundle in bundles for fact in bundle.event_evidence),
        supported_requirement_ids=supported, missing_requirement_ids=missing,
        requirement_support_decisions=decisions,
        conditional_requirement_ids=conditional, no_support_requirement_ids=no_support,
        provenance=provenance, token_cost=sum(bundle.token_cost for bundle in bundles),
        consistency=_merge_consistency(requirements, bundles, subjects, decisions), utility={},
    )
    # As in bundle construction, the schema's conservative missing default
    # cannot see requirement optionality or a satisfied root OR.
    object.__setattr__(aggregate, "missing_requirement_ids", missing)
    return aggregate


def _fact_payload(fact: EvidenceFact) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id, "fact_type": fact.fact_type, "value": fact.value,
        "time_span": {"start": fact.time_span.start, "end": fact.time_span.end},
        "frame_ids": list(fact.frame_ids), "source_evidence_ids": list(fact.source_evidence_ids),
        "role_bindings": dict(fact.role_bindings),
        "source_kind": fact.source_kind.value,
        "source_family": fact.source_family.value if fact.source_family else None,
        "derived_from": list(fact.derived_from),
        "independence_root_ids": list(fact.independence_root_ids),
        "episode_id": fact.episode_id,
        "derivation_rule": fact.derivation_rule,
        "provenance": fact.provenance,
    }


def _semantic_key(fact: EvidenceFact) -> str:
    payload = [
        fact.fact_type, fact.video_id, fact.role_bindings, fact.value,
        round(fact.time_span.start, 1), round(fact.time_span.end, 1),
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Phase7Pipeline:
    def __init__(self, shared_index: Any, fact_provider: FactProvider, *, config: Phase7Config,
                 token_counter: Callable[[list[dict]], int],
                 completion_sources: Mapping[Any, CompletionSource] | None = None,
                 parser_client: RequirementParseClient | None = None,
                 legacy_adapters: Mapping[str, Callable[[Phase7Package], Any]] | None = None,
                 visual_source: Any = None, visual_authorization_token: object | None = None,
                 runtime_metadata: dict[str, Any] | None = None,
                 normalization_policy: NormalizationPolicy = DEFAULT_NORMALIZATION_POLICY,
                 normalizer: Normalizer = normalize_and_fuse):
        if not isinstance(config, Phase7Config):
            raise TypeError("config must be Phase7Config")
        forbidden = _forbidden_keys(config.metadata) + _forbidden_keys(runtime_metadata or {}, "runtime_metadata")
        if forbidden:
            raise ValueError("runtime-forbidden fields: " + ", ".join(forbidden))
        if config.max_candidate_bundles != 2 or config.completion_rounds != 1:
            raise ValueError("Phase 7 bundle and completion bounds differ from protocol")
        self.shared_index = shared_index
        self.fact_provider = fact_provider
        self.config = config
        self.token_counter = token_counter
        self.completion_sources = dict(completion_sources or {})
        self.parser_client = parser_client
        self.legacy_adapters = dict(legacy_adapters or {})
        self.visual_source = visual_source
        self._visual_authorization_token = visual_authorization_token
        self.runtime_metadata = dict(runtime_metadata or {})
        self.normalization_policy = normalization_policy
        self.normalizer = normalizer

    def _flags(self) -> dict[str, bool]:
        return {name: bool(getattr(self.config, name)) for name in (*LEGACY_FLAGS, "visual_completion")}

    def _execute(self, query: str, video_id: str, *, extra_facts: list[EvidenceFact] | None = None,
                 visual_ablation: bool = False) -> Phase7Package:
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("video_id must be nonempty")
        requirements = parse_requirements(query, self.parser_client)
        facts = self.fact_provider.get_facts(requirements, video_id, self.shared_index)
        if not isinstance(facts, list) or any(not isinstance(fact, EvidenceFact) for fact in facts):
            raise TypeError("fact provider must return EvidenceFact list")
        facts = list(facts) + list(extra_facts or ())
        if any(not isinstance(fact, EvidenceFact) for fact in facts):
            raise TypeError("initial candidates must be EvidenceFact instances")
        if any(fact.video_id != video_id for fact in facts):
            raise ValueError("Phase 7 fact provider returned a different video_id")
        normalization = self.normalizer(facts, video_id=video_id, policy=self.normalization_policy)
        facts = list(normalization.facts)
        bundles = build_candidate_bundles(
            requirements, facts, max_bundles=self.config.max_candidate_bundles,
            normalization=normalization,
        )
        by_id = {fact.fact_id: fact for fact in facts}
        for bundle in bundles:
            bundle.provenance["fact_semantic_keys"] = {
                fact_id: _semantic_key(by_id[fact_id]) for fact_id in bundle.provenance.get("fact_refs", ())
                if fact_id in by_id
            }
            bundle.provenance["visible_facts"] = [
                _fact_payload(by_id[fact_id]) for fact_id in bundle.provenance.get("fact_refs", ())
                if fact_id in by_id
            ]
            temporal_types = {"before_state", "after_state", "state_change"}
            temporal_facts = [by_id[fact_id] for fact_id in bundle.provenance.get("fact_refs", ())
                              if fact_id in by_id and by_id[fact_id].fact_type in temporal_types]
            if temporal_facts:
                bundle.consistency["temporal_consistency"] = all(
                    fact.derivation_rule and fact.provenance.get("video_id") == video_id
                    for fact in temporal_facts
                )
        completion = complete_bundles(
            requirements, bundles, self.completion_sources,
            max_rounds=self.config.completion_rounds,
            normalizer=self.normalizer, normalization_policy=self.normalization_policy,
        )
        selected = select_bundles(list(completion.bundles), max_bundles=self.config.max_candidate_bundles) \
            if completion.bundles else []
        aggregate = _aggregate(requirements, selected)
        assessment = assess_support(requirements, aggregate)
        visible = build_visible_context(
            query, selected, assessment,
            context_limit_chars=self.config.context_limit_chars,
            token_counter=self.token_counter,
        )
        flags = self._flags()
        trace = {
            "protocol_version": self.config.protocol_version,
            "scope_sha256": self.shared_index.manifest.get("scope_sha256"),
            "embedding_manifest_sha256": self.shared_index.manifest.get("embedding_manifest_sha256"),
            "embedding_model": self.shared_index.manifest.get("model_id"),
            "flags": flags,
            "visual_ablation": visual_ablation,
            "initial_fact_ids": [fact.fact_id for fact in facts],
            "normalization": normalization.trace,
            "episode_ids": [episode.episode_id for episode in normalization.track_episodes],
            "identity_statuses": [assessment.status.value for assessment in normalization.identity_assessments],
            "lineage_failures": normalization.trace.get("lineage_failures", []),
            "candidate_support_counts": {
                f"{level.value.lower()}_count": sum(
                    decision.level is level for bundle in bundles
                    for decision in bundle.requirement_support_decisions.values()
                ) for level in SupportLevel
            },
            "candidate_bundle_ids": [bundle.bundle_id for bundle in bundles],
            "selected_bundle_ids": [bundle.bundle_id for bundle in selected],
            "completion": completion.trace,
            "support_status": visible.assessment.compatibility_status,
            "full_support_count": visible.assessment.full_support_count,
            "conditional_support_count": visible.assessment.conditional_support_count,
            "no_support_count": visible.assessment.no_support_count,
            "projection_consistent": visible.assessment.projection_consistent,
        }
        finalize_trace = getattr(self.fact_provider, "finalize_trace", None)
        if callable(finalize_trace):
            phase_d_trace = finalize_trace(
                normalization=normalization,
                candidate_bundles=tuple(completion.bundles),
                selected_bundles=tuple(selected),
                visible_context=visible,
            )
            json.dumps(phase_d_trace, sort_keys=True, allow_nan=False)
            trace["phase_d"] = phase_d_trace
        package = Phase7Package(
            query=query, video_id=video_id, requirements=requirements,
            initial_fact_ids=tuple(fact.fact_id for fact in facts), bundles=tuple(selected),
            normalization=normalization,
            completion=completion, support=visible.assessment, visible_context=visible, trace=trace,
        )
        for flag in LEGACY_FLAGS:
            if flags[flag]:
                adapter = self.legacy_adapters.get(flag)
                if adapter is None:
                    raise ValueError(f"enabled ablation has no adapter: {flag}")
                adapter(package)
                trace.setdefault("legacy_ablation_routes", []).append(flag)
        return package

    def run(self, query: str, video_id: str) -> Phase7Package:
        return self._execute(query, video_id)

    def run_visual_completion(self, query: str, video_id: str, *,
                              authorization_token: object) -> Phase7Package:
        if not self.config.visual_completion or self.visual_source is None:
            raise VisualCompletionAuthorizationError("visual completion ablation is not enabled")
        if authorization_token is not self._visual_authorization_token or authorization_token is None:
            raise VisualCompletionAuthorizationError("visual completion authorization capability is invalid")
        requirements = parse_requirements(query, self.parser_client)
        facts = self.visual_source.fetch(query, video_id, requirements)
        if not isinstance(facts, list) or any(not isinstance(fact, EvidenceFact) for fact in facts):
            raise TypeError("visual completion source must return EvidenceFact list")
        return self._execute(query, video_id, extra_facts=facts, visual_ablation=True)


class RepairedFullAdapter:
    def __init__(self, shared_index: Any, *, retriever: Any = None, **retriever_kwargs):
        self.shared_index = shared_index
        if retriever is None:
            from evique.retriever import EvidenceRetriever
            retriever = EvidenceRetriever(
                shared_index.index_dir, embedding_index=shared_index.dense_index, **retriever_kwargs
            )
        if Path(retriever.index_dir).resolve() != Path(shared_index.index_dir).resolve():
            raise ValueError("Arm B retriever does not use the shared repaired index")
        if retriever.embedding_index is not shared_index.dense_index:
            raise ValueError("Arm B retriever does not use the shared dense identity")
        self.retriever = retriever

    def retrieve(self, query: str, video_id: str) -> dict:
        result = self.retriever.retrieve(query, query_metadata={"video_id": video_id})
        if not isinstance(result, dict):
            raise TypeError("retained EvidenceRetriever must return an object")
        return {
            **result,
            "phase7_shared_scope_sha256": self.shared_index.manifest.get("scope_sha256"),
            "phase7_shared_embedding_sha256": self.shared_index.manifest.get("embedding_manifest_sha256"),
        }
