from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schema import EvidenceBundle


UTILITY_WEIGHTS = {
    "mandatory_requirement_coverage": 0.28,
    "new_fact_gain": 0.18,
    "role_consistency": 0.14,
    "temporal_consistency": 0.12,
    "provenance_completeness": 0.12,
    "semantic_density": 0.10,
    "normalized_token_cost": -0.03,
    "redundancy": -0.09,
}


@dataclass(frozen=True)
class UtilityBreakdown:
    mandatory_requirement_coverage: float
    new_fact_gain: float
    role_consistency: float
    temporal_consistency: float
    provenance_completeness: float
    semantic_density: float
    normalized_token_cost: float
    redundancy: float
    total: float


def _facts(bundle: EvidenceBundle) -> tuple[str, ...]:
    return (
        bundle.scene_evidence + bundle.object_evidence + bundle.relation_evidence
        + bundle.track_evidence + bundle.temporal_evidence + bundle.event_evidence
    )


def _hard_valid(bundle: EvidenceBundle) -> None:
    facts = _facts(bundle)
    provenance = bundle.provenance
    sources = provenance.get("fact_sources") or {}
    videos = provenance.get("fact_videos") or {}
    if provenance.get("video_id") != bundle.video_id:
        raise ValueError("bundle provenance has wrong video")
    if any(not sources.get(fact_id) for fact_id in facts):
        raise ValueError("bundle fact lacks provenance")
    if any(videos.get(fact_id) != bundle.video_id for fact_id in facts):
        raise ValueError("bundle fact has wrong video")


def _consistency(bundle: EvidenceBundle, key: str) -> float:
    if key == "temporal_consistency":
        value = bundle.consistency.get(key)
        return 1.0 if value is True else 0.0 if value is False or value is None else float(bool(value))
    checks = (bundle.consistency.get("role_checks") or {}).values()
    values = [item.get("consistent") for item in checks if isinstance(item, dict)]
    if not values:
        return 1.0
    return sum(value is True for value in values) / len(values)


def _semantic_keys(bundle: EvidenceBundle) -> dict[str, str]:
    configured = bundle.provenance.get("fact_semantic_keys") or {}
    return {fact_id: str(configured.get(fact_id) or fact_id) for fact_id in _facts(bundle)}


def score_bundle(bundle: EvidenceBundle, selected: Sequence[EvidenceBundle]) -> UtilityBreakdown:
    _hard_valid(bundle)
    for prior in selected:
        _hard_valid(prior)
    mandatory_total = len(bundle.supported_requirement_ids) + len(bundle.missing_requirement_ids)
    coverage = len(bundle.supported_requirement_ids) / mandatory_total if mandatory_total else 0.0
    support = bundle.provenance.get("requirement_support") or {}
    keys = _semantic_keys(bundle)
    pairs = {(requirement_id, keys.get(fact_id, fact_id))
             for requirement_id, fact_ids in support.items() for fact_id in fact_ids}
    selected_pairs = set()
    selected_semantics = set()
    for prior in selected:
        prior_keys = _semantic_keys(prior)
        selected_semantics.update(prior_keys.values())
        for requirement_id, fact_ids in (prior.provenance.get("requirement_support") or {}).items():
            selected_pairs.update((requirement_id, prior_keys.get(fact_id, fact_id)) for fact_id in fact_ids)
    new_pairs = pairs - selected_pairs
    # Protocol v1 normalizes five distinct marginal requirement/fact pairs to
    # full gain; a single novel pair remains useful without equating it to a
    # bundle that covers several new requested aspects.
    new_fact_gain = min(1.0, len(new_pairs) / 5)
    fact_keys = list(keys.values())
    redundant = sum(
        key in selected_semantics or key in fact_keys[:index]
        for index, key in enumerate(fact_keys)
    )
    redundancy = redundant / len(fact_keys) if fact_keys else 0.0
    provenance = 1.0 if all(
        bundle.provenance.get("fact_sources", {}).get(fact_id) for fact_id in _facts(bundle)
    ) else 0.0
    density = min(1.0, len(pairs | {(None, key) for key in set(fact_keys)}) * 100 / max(bundle.token_cost, 1))
    token_cost = min(1.0, bundle.token_cost / 4096)
    values = {
        "mandatory_requirement_coverage": coverage,
        "new_fact_gain": new_fact_gain,
        "role_consistency": _consistency(bundle, "role_consistency"),
        "temporal_consistency": _consistency(bundle, "temporal_consistency"),
        "provenance_completeness": provenance,
        "semantic_density": density,
        "normalized_token_cost": token_cost,
        "redundancy": redundancy,
    }
    total = sum(UTILITY_WEIGHTS[key] * value for key, value in values.items())
    return UtilityBreakdown(**values, total=total)


def select_bundles(bundles: list[EvidenceBundle], *, max_bundles: int = 2) -> list[EvidenceBundle]:
    if type(max_bundles) is not int or not 1 <= max_bundles <= 2:
        raise ValueError("max_bundles must be one or two")
    remaining = list(bundles)
    selected: list[EvidenceBundle] = []
    while remaining and len(selected) < max_bundles:
        ranked = sorted(
            ((score_bundle(bundle, selected), bundle) for bundle in remaining),
            key=lambda item: (-item[0].total, item[1].time_window.start, item[1].bundle_id),
        )
        winner = ranked[0][1]
        selected.append(winner)
        remaining.remove(winner)
    return selected
