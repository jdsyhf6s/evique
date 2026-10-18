"""Offline experimental measurements over frozen validators and external judgements.

``compute_metrics`` accepts a QueryPlan, EvidenceRequirement, or their dict form.
It returns six scalar values (None means unmeasured) plus ``metric_details``.
Each detail has status, numerator, denominator and unknown_count. Coverage is
covered / requested roles; consistency is the frozen checker's comparable checks.
These observation metrics do not establish answer truth or causal support.

External judgement schema (no model is called here)::

    {"preference": {"forward": "A", "reverse": "B"},
     "claims": [{"claim": "A bus enters.", "verdict": "supported",
                 "evidence_ids": ["entry"]}]}

The candidate is A in the forward presentation and B in the reversed one.
Votes are A, B, or tie. Both presentations must agree after reversing order;
otherwise preference is None with status order_sensitive. Ties score 0.5.
Claim verdicts are supported, unsupported, contradicted, or unknown. Supported
claims require nonempty IDs identifying unique supplied evidence records; this
checks citation integrity only, never substitutes for the external judgement.
Unsupported rate is (unsupported + contradicted) / assessed claims, and is None
if any claim is unknown or malformed. Empty claims are unmeasured, not zero.
Judgers are responsible for exhausting factual claims in the answer; this module
does not infer claim completeness from text or fabricate semantic assessments.

``answer`` is {"text": str, "abstained": bool}. Abstention quality is decision
accuracy against the independent bool ``answerable``. Missing labels remain
unavailable, including when coverage is complete or incomplete.

``aggregate_results`` accepts rows {"method": str, "metrics": result} and
returns one sorted row per method, with the same six scalar names. It pools
numerators/denominators only for measured rows, separately reporting observed
and unknown row counts and unknown checks. Denominators are roles, comparable
checks, fully assessed claims, or evaluated decisions; unavailable rows never
become zeros. A partial aggregate describes measured rows only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any

from ..evidence_consistency import EvidenceConsistencyChecker
from ..evidence_validator import EvidenceValidator


METRIC_NAMES = (
    'answer_preference', 'evidence_coverage', 'identity_consistency',
    'temporal_consistency', 'unsupported_claim_rate', 'abstention_quality',
)


def _mapping(value: Any) -> dict:
    return asdict(value) if is_dataclass(value) else value if isinstance(value, dict) else {}


def _measurement(value: float | None, denominator: int = 0, *,
                 status: str | None = None, unknown_count: int = 0, **extra: Any) -> dict:
    return {'status': status or ('available' if value is not None else 'unavailable'),
            'numerator': value * denominator if value is not None else None,
            'denominator': denominator, 'unknown_count': unknown_count, **extra}


def _preference(judgement: dict) -> tuple[float | None, dict]:
    votes = _mapping(judgement.get('preference'))
    forward = {'A': 1.0, 'B': 0.0, 'tie': .5}.get(str(votes.get('forward')))
    reverse = {'A': 0.0, 'B': 1.0, 'tie': .5}.get(str(votes.get('reverse')))
    if forward is None or reverse is None:
        return None, _measurement(None)
    if forward != reverse:
        return None, _measurement(None, status='order_sensitive', unknown_count=1)
    return forward, _measurement(forward, 1)


def _claims(judgement: dict, evidence: list[dict]) -> tuple[float | None, dict]:
    claims = judgement.get('claims')
    if not isinstance(claims, list) or not claims:
        return None, _measurement(None, total_claims=0, assessed_claims=0)
    ids = Counter(str(row.get('evidence_id') or row.get('id') or row.get('node_id') or '')
                  for row in evidence if isinstance(row, dict))
    valid_ids = {eid for eid, count in ids.items() if eid and count == 1}
    counts = Counter()
    for claim in claims:
        item = _mapping(claim)
        verdict = item.get('verdict')
        citations = item.get('evidence_ids')
        valid = (isinstance(item.get('claim'), str) and bool(item['claim'].strip())
                 and isinstance(citations, list)
                 and all(isinstance(eid, str) and eid in valid_ids for eid in citations))
        if (not valid or not isinstance(verdict, str)
                or verdict not in {'supported', 'unsupported', 'contradicted', 'unknown'}):
            verdict = 'unknown'
        if verdict == 'supported' and not citations:
            verdict = 'unknown'
        counts[verdict] += 1
    unknown = counts['unknown']
    assessed = len(claims) - unknown
    unsupported = counts['unsupported'] + counts['contradicted']
    value = unsupported / assessed if assessed and not unknown else None
    return value, _measurement(value, assessed, status='unknown' if unknown else None,
                               unknown_count=unknown, total_claims=len(claims),
                               assessed_claims=assessed, unsupported_claims=unsupported,
                               supported_claims=counts['supported'])


def compute_metrics(requirements: Any, evidence: list[dict], *, answer: dict | None = None,
                    judgement: dict | None = None, answerable: bool | None = None) -> dict:
    """Measure observations and supplied judgements without changing either input."""
    result = {name: None for name in METRIC_NAMES}
    details = {name: _measurement(None) for name in METRIC_NAMES}
    plan = _mapping(requirements)
    req = _mapping(plan.get('requirements', plan))
    has_requirements = bool(plan.get('required_evidence')) or any(req.get(key) for key in (
        'target_requirement', 'anchor_requirement', 'temporal_requirement',
        'comparison_requirement', 'event_requirement'))
    if has_requirements:
        validation = EvidenceValidator().validate(requirements, evidence)
        roles = validation['satisfied']
        if roles:
            value = sum(bool(ids) for ids in roles.values()) / len(roles)
            result['evidence_coverage'] = value
            details['evidence_coverage'] = _measurement(
                value, len(roles), missing_roles=validation['missing'],
                conflict_count=len(validation['conflicts']))

    consistency = EvidenceConsistencyChecker().check(evidence, requirements)
    for dimension in ('identity', 'temporal'):
        name = dimension + '_consistency'
        value = consistency[dimension + '_score']
        counts = consistency['checks'][dimension]
        result[name] = value
        details[name] = _measurement(value, counts['compared'],
                                     status='partial' if counts['unknown'] and value is not None
                                     else 'unknown' if counts['unknown'] else None,
                                     unknown_count=counts['unknown'])

    external = _mapping(judgement)
    result['answer_preference'], details['answer_preference'] = _preference(external)
    result['unsupported_claim_rate'], details['unsupported_claim_rate'] = _claims(external, evidence)
    supplied_answer = _mapping(answer)
    if (isinstance(answerable, bool) and isinstance(supplied_answer.get('abstained'), bool)
            and isinstance(supplied_answer.get('text'), str)):
        value = float(supplied_answer['abstained'] != answerable)
        result['abstention_quality'] = value
        details['abstention_quality'] = _measurement(value, 1)
    result['metric_details'] = details
    return result


def aggregate_results(rows: list[dict]) -> list[dict]:
    """Pool metric denominators by method, explicitly counting unmeasured rows."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row['method']), []).append(row['metrics'])
    result = []
    for method, measurements in sorted(grouped.items()):
        output: dict[str, Any] = {'method': method, 'row_count': len(measurements), 'metric_details': {}}
        for name in METRIC_NAMES:
            numerator = 0.0
            denominator = observed = unknown_checks = 0
            for measurement in measurements:
                detail = measurement.get('metric_details', {}).get(name, {})
                unknown_checks += detail.get('unknown_count', 0)
                if measurement.get(name) is not None and detail.get('denominator', 0) > 0:
                    numerator += detail['numerator']
                    denominator += detail['denominator']
                    observed += 1
            value = numerator / denominator if denominator else None
            unknown_rows = len(measurements) - observed
            output[name] = value
            output['metric_details'][name] = _measurement(
                value, denominator, status='partial' if observed and unknown_rows else None,
                unknown_count=unknown_rows, observed_count=observed, unknown_checks=unknown_checks)
        result.append(output)
    return result
