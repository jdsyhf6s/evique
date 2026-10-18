from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Collection, Sequence

from .bridge_trace import RELATION_FUNNEL_STAGES, RelationFunnelTrace
from .config import TemporalRelationBridgeConfig
from .local_completion import FrozenIndexCompletionSource
from .normalization import compatible_components, normalize_and_fuse
from .schema import EvidenceBundle, EvidenceFact, QuestionRequirementSet, RequirementType
from .temporal_refinement import refine_temporal_windows


@dataclass(frozen=True)
class CandidateAdmissionResult:
    facts: tuple[EvidenceFact, ...]
    admitted_fact_ids: tuple[str, ...]
    displaced_fact_ids: tuple[str, ...]
    rejected_fact_ids: tuple[str, ...]


def _semantic_key(fact: EvidenceFact) -> str:
    payload = {
        "video_id": fact.video_id,
        "fact_type": fact.fact_type,
        "time": [fact.time_span.start, fact.time_span.end],
        "roles": fact.role_bindings,
        "value": fact.value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def admit_phase_d_candidates(
    base_facts: Sequence[EvidenceFact],
    derived_facts: Sequence[EvidenceFact],
    *,
    limit: int,
    protected_fact_ids: Collection[str],
) -> CandidateAdmissionResult:
    if type(limit) is not int or limit <= 0:
        raise ValueError("candidate limit must be a positive integer")
    if len(base_facts) > limit:
        raise ValueError("base facts already exceed the fixed candidate limit")
    facts = list(base_facts)
    protected = set(protected_fact_ids)
    admitted: list[str] = []
    displaced: list[str] = []
    rejected: list[str] = []
    for candidate in derived_facts:
        if any(existing.fact_id == candidate.fact_id for existing in facts):
            rejected.append(candidate.fact_id)
            continue
        if len(facts) < limit:
            facts.append(candidate)
            admitted.append(candidate.fact_id)
            continue
        replace_index = next(
            (index for index in range(len(facts) - 1, -1, -1)
             if facts[index].fact_id not in protected),
            None,
        )
        if replace_index is None:
            rejected.append(candidate.fact_id)
            continue
        displaced.append(facts[replace_index].fact_id)
        facts[replace_index] = candidate
        admitted.append(candidate.fact_id)
    return CandidateAdmissionResult(tuple(facts), tuple(admitted), tuple(displaced), tuple(rejected))


class TemporalRelationFactProvider:
    def __init__(
        self,
        base_provider: Any,
        *,
        query: str,
        source_factory: Callable[[QuestionRequirementSet], FrozenIndexCompletionSource],
        config: TemporalRelationBridgeConfig,
        candidate_limit: int,
    ):
        if not config.enabled:
            raise ValueError("TemporalRelationFactProvider requires enabled config")
        if type(candidate_limit) is not int or candidate_limit <= 0:
            raise ValueError("candidate_limit must be a positive integer")
        self.base_provider = base_provider
        self.query = query
        self.source_factory = source_factory
        self.config = config
        self.candidate_limit = candidate_limit
        self._reset()

    def _reset(self) -> None:
        self._artifact_rows: set[str] = set()
        self._window_rows: set[str] = set()
        self._role_rows: set[str] = set()
        self._candidate_rows: set[str] = set()
        self._fact_to_row: dict[str, str] = {}
        self._row_reasons: dict[str, str] = {}
        self._refined_window_count = 0
        self._temporal_fact_count = 0

    @staticmethod
    def _row_id(row: dict[str, Any], index: int) -> str:
        for name in ("relation_id", "id"):
            value = row.get(name)
            if isinstance(value, str) and value:
                return value
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return "relation-row:" + hashlib.sha256(f"{index}:{payload}".encode()).hexdigest()[:20]

    @staticmethod
    def _protected_base_ids(
        base_facts: Sequence[EvidenceFact], requirements: QuestionRequirementSet,
    ) -> set[str]:
        if not base_facts:
            return set()
        normalization = normalize_and_fuse(
            list(base_facts), video_id=base_facts[0].video_id
        )
        protected: set[str] = set()
        covered_pairs: set[tuple[str, str]] = set()
        empty_contribution_protected = False
        for fact in base_facts:
            contribution = {
                (requirement.requirement_id, component.component_id)
                for requirement in requirements.requirements
                for component in compatible_components(requirement, (fact,), normalization)
                if fact.fact_id in component.fact_ids
                or fact.fact_id in component.observation_only_fact_ids
            }
            if contribution - covered_pairs or (not contribution and not empty_contribution_protected):
                protected.add(fact.fact_id)
                covered_pairs.update(contribution)
                empty_contribution_protected = empty_contribution_protected or not contribution
        parent_ids = {
            identifier for fact in base_facts
            for identifier in (*fact.derived_from, *fact.source_evidence_ids)
        }
        protected.update(fact.fact_id for fact in base_facts if fact.fact_id in parent_ids)
        protected.update(
            fact.fact_id for fact in base_facts
            if fact.fact_type in {"entry", "event", "visual_event", "anchor_state"}
            or fact.provenance.get("identity_assessment_id")
            or fact.provenance.get("identity_evidence_ids")
        )
        return protected

    def get_facts(self, requirements: QuestionRequirementSet, video_id: str,
                  shared_index: Any) -> list[EvidenceFact]:
        self._reset()
        base_facts = self.base_provider.get_facts(requirements, video_id, shared_index)
        if not isinstance(base_facts, list) or any(not isinstance(fact, EvidenceFact) for fact in base_facts):
            raise TypeError("base fact provider must return an EvidenceFact list")
        windows = refine_temporal_windows(
            self.query, requirements, base_facts,
            max_windows=self.config.max_refined_windows,
        )
        self._refined_window_count = len(windows)
        source = self.source_factory(requirements)
        for index, row in enumerate(getattr(source, "relations", ())):
            declared = row.get("video_id") or row.get("video_name")
            if str(declared or "") == video_id:
                self._artifact_rows.add(self._row_id(row, index))

        object_ids = tuple(dict.fromkeys(
            identifier for fact in base_facts
            for identifier in (
                *fact.role_bindings.values(),
                *((fact.value.get("object_id"),) if isinstance(fact.value, dict) else ()),
            )
            if isinstance(identifier, str) and identifier
        ))
        source_ids = tuple(dict.fromkeys(
            identifier for fact in base_facts for identifier in fact.source_evidence_ids
        ))
        derived: list[EvidenceFact] = []
        for requirement in requirements.requirements:
            if requirement.requirement_type is RequirementType.SPATIAL_RELATION:
                result = source.convert_relations(
                    requirement,
                    video_id=video_id,
                    subject_object_ids=object_ids,
                    windows=tuple(window.time_range for window in windows),
                )
                derived.extend(result.facts)
                self._fact_to_row.update(result.fact_to_row_id)
                self._window_rows.update(result.window_matched_row_ids)
                self._role_rows.update(result.role_matched_row_ids)
                self._row_reasons.update({item.row_id: item.reason for item in result.rejected})
            elif requirement.requirement_type in {
                RequirementType.ENTRY_EXIT,
                RequirementType.BEFORE_STATE,
                RequirementType.AFTER_STATE,
                RequirementType.STATE_CHANGE,
            }:
                result = source.convert_temporal(
                    requirement,
                    video_id=video_id,
                    source_evidence_ids=source_ids,
                    windows=windows,
                    limit=self.candidate_limit,
                )
                derived.extend(result.facts)

        derived = list({fact.fact_id: fact for fact in derived}.values())
        protected = self._protected_base_ids(base_facts, requirements)
        lineage = {
            identifier for fact in derived
            for identifier in (*fact.derived_from, *fact.source_evidence_ids)
        }
        protected.update(
            fact.fact_id for fact in base_facts
            if {fact.fact_id, *fact.source_evidence_ids} & lineage
        )
        endpoint_ids = {
            identifier for fact in derived for identifier in fact.role_bindings.values()
        }
        for fact in base_facts:
            represented = set(fact.role_bindings.values())
            if isinstance(fact.value, dict):
                represented.update(
                    value for key in ("object_id", "track_id")
                    if isinstance((value := fact.value.get(key)), str) and value
                )
            if represented & endpoint_ids:
                protected.add(fact.fact_id)
        admission = admit_phase_d_candidates(
            base_facts, derived, limit=self.candidate_limit, protected_fact_ids=protected,
        )
        admitted = set(admission.admitted_fact_ids)
        self._candidate_rows = {
            row_id for fact_id, row_id in self._fact_to_row.items() if fact_id in admitted
        }
        for fact_id in admission.rejected_fact_ids:
            row_id = self._fact_to_row.get(fact_id)
            if row_id:
                self._row_reasons[row_id] = "CANDIDATE_BUDGET_REJECTED"
        temporal_types = {"entry", "before_state", "anchor_state", "after_state", "state_change"}
        self._temporal_fact_count = sum(
            fact.fact_id in admitted and fact.fact_type in temporal_types for fact in derived
        )
        return list(admission.facts)

    @staticmethod
    def _bundle_fact_ids(bundles: Sequence[EvidenceBundle]) -> set[str]:
        result = set()
        for bundle in bundles:
            result.update(bundle.provenance.get("fact_refs") or ())
            result.update(bundle.scene_evidence)
            result.update(bundle.object_evidence)
            result.update(bundle.relation_evidence)
            result.update(bundle.track_evidence)
            result.update(bundle.temporal_evidence)
            result.update(bundle.event_evidence)
        return result

    def finalize_trace(
        self,
        *,
        normalization: Any,
        candidate_bundles: Sequence[EvidenceBundle],
        selected_bundles: Sequence[EvidenceBundle],
        visible_context: Any,
    ) -> dict[str, Any]:
        normalized_ids = {fact.fact_id for fact in normalization.facts}
        supported_ids = {
            fact_id for bundle in candidate_bundles
            for decision in bundle.requirement_support_decisions.values()
            for fact_id in (
                *decision.supporting_fact_ids,
                *(item for component in decision.component_decisions for item in component.fact_ids),
            )
        }
        selected_ids = self._bundle_fact_ids(selected_bundles)
        visible_ids = {
            fact_id
            for identifiers in (visible_context.trace.get("visible_fact_ids_by_bundle") or {}).values()
            for fact_id in identifiers
        }
        stages = {
            "RELATION_ARTIFACT_ROWS": set(self._artifact_rows),
            "RELATION_WINDOW_MATCH": set(self._window_rows) & self._artifact_rows,
            "RELATION_ROLE_MATCH": set(self._role_rows) & self._window_rows,
            "RELATION_CANDIDATES": set(self._candidate_rows) & self._role_rows,
            "RELATION_NORMALIZED": {
                row for fact, row in self._fact_to_row.items()
                if fact in normalized_ids and row in self._candidate_rows
            },
            "RELATION_SUPPORTED": {
                row for fact, row in self._fact_to_row.items()
                if fact in supported_ids and row in self._candidate_rows
            },
            "RELATION_RETRIEVED": {
                row for fact, row in self._fact_to_row.items()
                if fact in selected_ids and row in self._candidate_rows
            },
            "RELATION_VISIBLE": {
                row for fact, row in self._fact_to_row.items()
                if fact in visible_ids and row in self._candidate_rows
            },
        }
        trace = RelationFunnelTrace.start(tuple(sorted(stages[RELATION_FUNNEL_STAGES[0]])))
        defaults = {
            "RELATION_WINDOW_MATCH": "OUTSIDE_REFINED_WINDOW",
            "RELATION_ROLE_MATCH": "ROLE_MISMATCH",
            "RELATION_CANDIDATES": "CANDIDATE_BUDGET_REJECTED",
            "RELATION_NORMALIZED": "NORMALIZATION_REJECTED",
            "RELATION_SUPPORTED": "NO_REQUIREMENT_SUPPORT",
            "RELATION_RETRIEVED": "BUNDLE_NOT_SELECTED",
            "RELATION_VISIBLE": "VISIBLE_BUDGET_REJECTED",
        }
        previous = stages[RELATION_FUNNEL_STAGES[0]]
        for stage in RELATION_FUNNEL_STAGES[1:]:
            current = stages[stage] & previous
            trace.advance(stage, tuple(sorted(current)))
            for row_id in sorted(previous - current):
                trace.reject(
                    row_id,
                    stage=stage,
                    reason=self._row_reasons.get(row_id, defaults[stage]),
                )
            previous = current
        funnel = trace.finalize()
        return {
            "relation_funnel": funnel,
            "REFINED_WINDOW_COUNT": self._refined_window_count,
            "TEMPORAL_FACT_COUNT": self._temporal_fact_count,
            "VISIBLE_CONTEXT_TOKENS": visible_context.trace.get("actual_message_tokens", 0),
        }
