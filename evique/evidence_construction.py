"""Bounded evidence construction after retrieval and before existing packaging."""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Any

from .candidate_fusion import CandidateEvidence, reciprocal_rank_fusion
from .evidence_graph_executor import EvidenceGraphExecutor
from .evidence_validator import EvidenceValidator
from .schema import normalize_evidence


class EvidenceConstructionEngine:
    def __init__(self, records: list[dict], relations=(), *, max_hops: int = 3,
                 max_nodes: int = 100, max_retrieval_rounds: int = 2):
        if not 0 <= max_retrieval_rounds <= 3:
            raise ValueError('max_retrieval_rounds must be between 0 and 3')
        self.records = records
        self.relations = relations
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_retrieval_rounds = max_retrieval_rounds
        self.validator = EvidenceValidator()

    @staticmethod
    def _identity(item):
        return item.get('video_id'), item.get('view'), item.get('evidence_id') or item.get('id')

    @staticmethod
    def _retrieval_query(requirements: dict, missing: list[str], original: str) -> str:
        terms = list(requirements.get('target_requirement', {}).get('categories', []))
        terms += list(requirements.get('anchor_requirement', {}).get('event_types', []))
        temporal = requirements.get('temporal_requirement', {})
        for side in ('before', 'after'):
            if f'{side}_state' in missing:
                terms += [side, str(temporal.get(side) or 'state')]
        if 'event_evidence' in missing:
            terms += list(requirements.get('event_requirement', {}).get('event_types', []))
        if 'transcript' in missing:
            terms.append('speech transcript')
        return ' '.join(dict.fromkeys(terms)) or original

    def construct(self, query: str, plan, initial_candidates: list[dict], *, video_id: str,
                  retrieve_more: Callable[..., list[CandidateEvidence]] | None = None) -> dict[str, Any]:
        requirement = plan.requirements
        requirements = requirement.to_dict() if hasattr(requirement, 'to_dict') else deepcopy(requirement or {})
        seeds = [normalize_evidence(item) for item in initial_candidates]
        initial = deepcopy(seeds)
        catalog = list(self.records)
        expansions, retrieval_rounds = [], []
        stop_reason = 'retrieval_budget_exhausted'
        for round_number in range(self.max_retrieval_rounds + 1):
            executor = EvidenceGraphExecutor(catalog, self.relations, max_hops=self.max_hops, max_nodes=self.max_nodes)
            expansion = executor.expand(seeds, video_id=video_id)
            expansions.append({'round': round_number, **expansion})
            expanded = expansion['expanded_evidence']
            validation = self.validator.validate(plan, expanded, video_id=video_id)
            if validation['complete']:
                stop_reason = 'sufficient_evidence'
                break
            if retrieve_more is None or round_number == self.max_retrieval_rounds:
                stop_reason = 'retrieval_unavailable' if retrieve_more is None else 'retrieval_budget_exhausted'
                break
            supplemental_query = self._retrieval_query(requirements, validation['missing'], query)
            top_k = min(100, 40 * (round_number + 1))
            candidates = retrieve_more(supplemental_query, {'video_id': video_id}, top_k=top_k)
            channels = {source: [row for row in candidates if row.source == source]
                        for source in ('keyword', 'embedding', 'visual', 'event')}
            fused = reciprocal_rank_fusion(channels, video_ids={video_id})
            known = {self._identity(row) for row in seeds}
            additions = []
            for row in fused:
                evidence = normalize_evidence({**row['evidence'], 'score': row['fusion_score'],
                    'fusion_score': row['fusion_score'], 'original_ranks': row['original_ranks'],
                    'original_scores': row['original_scores'], 'retrieval_sources': row['sources']})
                if self._identity(evidence) not in known:
                    known.add(self._identity(evidence))
                    additions.append(evidence)
            retrieval_rounds.append({'round': round_number + 1, 'query': supplemental_query,
                'video_id': video_id, 'top_k': top_k, 'missing_before': list(validation['missing']),
                'candidates': [row.to_dict() for row in candidates], 'fusion_result': fused,
                'new_evidence_ids': [row['evidence_id'] for row in additions]})
            if not additions:
                stop_reason = 'no_new_evidence'
                break
            seeds.extend(additions)
            # Retry candidates may come from newly available evidence, but each
            # expansion restarts at the original/retrieved seeds: hops never
            # accumulate beyond the configured cap across retry rounds.
            catalog_ids = {self._identity(normalize_evidence(row)) for row in catalog}
            catalog.extend(row for row in additions if self._identity(row) not in catalog_ids)
        annotated = []
        for row in expanded:
            item = deepcopy(row)
            item['requirement_roles'] = [role for role, ids in validation.get('satisfied', {}).items()
                                         if item['evidence_id'] in ids]
            annotated.append(item)
        return {'requirements': requirements, 'required_evidence': list(plan.required_evidence),
                'initial_candidates': initial, 'graph_expansion': expansions,
                'expanded_evidence': annotated, 'validation_result': validation,
                'retrieval_rounds': retrieval_rounds, 'stop_reason': stop_reason,
                'video_id': video_id, 'max_hops': self.max_hops,
                'max_retrieval_rounds': self.max_retrieval_rounds}
