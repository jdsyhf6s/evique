"""Conservative consistency diagnostics over observations, never answer claims.

Scores are fractions of comparable checks. Unknown checks remain unknown; a
partially observed dimension cannot receive a perfect score. Requirement roles
select what to compare but never establish identity, chronology, or causality.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import combinations
import math
from typing import Any

from .evidence_graph_executor import evidence_time_range
from .evidence_validator import EvidenceValidator
from .hybrid_identity import strict_video_id


def _mapping(value: Any) -> dict:
    return asdict(value) if is_dataclass(value) else value if isinstance(value, dict) else {}


def _original(row: dict) -> dict:
    if 'record' in row and row['record'] is not None:
        return row['record'] if isinstance(row['record'], dict) else {}
    return row


def _get(row: dict, key: str, default: Any = None) -> Any:
    source = _original(row)
    attrs = source.get('attributes')
    return source.get(key, attrs.get(key, default) if isinstance(attrs, dict) else default)


def _ids(value: Any) -> set[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {str(item).strip() for item in values
            if isinstance(item, (str, int)) and not isinstance(item, bool) and str(item).strip()}


def _eid(row: dict) -> str:
    return str(row.get('evidence_id') or row.get('id') or _get(row, 'id') or '')


def _kind(row: dict) -> str:
    return str(row.get('evidence_type') or row.get('view') or _get(row, 'evidence_type') or _get(row, 'view') or '')


def _roles(row: dict) -> set[str]:
    return _ids(row.get('requirement_roles'))


_TRACK_KEYS = ('track_id', 'target_track_id', 'related_track_id', 'related_track_ids', 'related_tracks')
_OBJECT_KEYS = ('object_id', 'target_object_id', 'related_object_id', 'object_ids', 'related_object_ids')
_TARGET_KINDS = {'target', 'visual_object', 'object'}
_TRACK_KINDS = {'track', 'visual_track'}
_EVENT_KINDS = {'event', 'visual_event', 'adaptive_event'}


def _refs(row: dict, keys: tuple[str, ...]) -> set[str]:
    return set().union(*(_ids(_get(row, key)) for key in keys))


def _explicit_identity_conflict(left: dict, right: dict) -> bool:
    """Contradictions within a known namespace cannot be bridged away."""
    for a, b in ((_refs(left, _TRACK_KEYS), _refs(right, _TRACK_KEYS)),
                 (_ids(_get(left, 'subject_id')), _ids(_get(right, 'subject_id')))):
        if a and b and not a & b:
            return True
    return False


def identity_relation(left: Any, right: Any) -> bool | None:
    """Whether two rows explicitly refer to the same subject in one video.

    False is an explicit mismatch, None is insufficient identity information.
    Track/object namespaces stay distinct; subject_id is a generic identity.
    Shared context/support IDs alone do not establish a shared subject.
    """
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    lv, rv = strict_video_id(left), strict_video_id(right)
    if not lv or not rv:
        return None
    if lv != rv:
        return False
    lt, rt = _refs(left, _TRACK_KEYS), _refs(right, _TRACK_KEYS)
    lo, ro = _refs(left, _OBJECT_KEYS), _refs(right, _OBJECT_KEYS)
    ls, rs = _ids(_get(left, 'subject_id')), _ids(_get(right, 'subject_id'))
    # Native node IDs count only in their own namespace, not as subject IDs.
    if _kind(left) in _TRACK_KINDS:
        lt |= _ids(_eid(left))
    if _kind(right) in _TRACK_KINDS:
        rt |= _ids(_eid(right))
    if _kind(left) in _TARGET_KINDS:
        lo |= _ids(_eid(left))
    if _kind(right) in _TARGET_KINDS:
        ro |= _ids(_eid(right))
    # A contradictory explicit track cannot be repaired by a shared category
    # or an incidental shared support link.
    if _explicit_identity_conflict(left, right):
        return False
    if lt & rt or lo & ro or ls & (rt | ro | rs) or rs & (lt | lo):
        return True
    for a, b in ((lt, rt), (ls, rt), (rs, lt)):
        if a and b and not a & b:
            return False
    support_keys = ('supporting_ids', 'support_refs', 'supporting_refs')
    if (_refs(left, support_keys) & (rt | ro | rs) or
            _refs(right, support_keys) & (lt | lo | ls)):
        return True
    # Distinct object references conflict only if both are explicit fields;
    # object observations can have different IDs along the same track.
    explicit_lo, explicit_ro = _refs(left, _OBJECT_KEYS), _refs(right, _OBJECT_KEYS)
    if explicit_lo and explicit_ro and not explicit_lo & explicit_ro:
        return False
    return None


def _span(row: dict) -> tuple[float, float] | None:
    return evidence_time_range(_original(row))


def _score(checks: list[bool | None]) -> float | None:
    known = [value for value in checks if value is not None]
    if not known or (None in checks and all(known)):
        return None
    return sum(known) / len(known)


def _bbox(row: dict) -> tuple[float, ...] | None:
    box = _get(row, 'bbox')
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(v, bool) for v in box):
        return None
    try:
        box = tuple(float(v) for v in box)
    except (TypeError, ValueError, OverflowError):
        return None
    return box if all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3] else None


def _spatial_relation(left: dict, right: dict, *, compare_regions: bool = False) -> bool | None:
    lv, rv = strict_video_id(left), strict_video_id(right)
    if not lv or not rv or lv != rv:
        return None
    frame_left, frame_right = _get(left, 'frame_id'), _get(right, 'frame_id')
    if frame_left is not None and frame_right is not None:
        same_frame = frame_left == frame_right
    else:
        a, b = _span(left), _span(right)
        same_frame = a is not None and a == b and a[0] == a[1]
    if compare_regions or same_frame:
        lr = _refs(left, ('region_id', 'area_id', 'referenced_area_id', 'referenced_region_id'))
        rr = _refs(right, ('region_id', 'area_id', 'referenced_area_id', 'referenced_region_id'))
        if lr and rr:
            return bool(lr & rr)
    if identity_relation(left, right) is not True:
        return None
    space = _get(left, 'coordinate_space')
    if not same_frame or not space or space != _get(right, 'coordinate_space'):
        return None
    a, b = _bbox(left), _bbox(right)
    if a is None or b is None:
        return None
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _causal_support(row: dict, rows: list[dict]) -> bool:
    source = _original(row)
    relation = str(_get(row, 'relation_type', '')).lower()
    provenance = source.get('provenance')
    backing = _get(row, 'source_ref') or _get(row, 'source_path') or _get(row, 'annotation_id')
    if isinstance(provenance, dict):
        backing = backing or provenance.get('source_ref') or provenance.get('source_path')
    video = strict_video_id(row)
    endpoints = [_ids(_get(row, key)) for key in ('source_id', 'target_id')]
    if (relation not in {'causes', 'caused_by', 'causal'} or not video or
            not isinstance(backing, str) or not backing.strip() or
            any(len(endpoint) != 1 for endpoint in endpoints) or endpoints[0] == endpoints[1]):
        return False
    # A citation marker alone cannot ground a relation with missing endpoints.
    for endpoint in endpoints:
        matches = [item for item in rows if _eid(item) in endpoint]
        if len(matches) != 1 or strict_video_id(matches[0]) != video:
            return False
    return True


class EvidenceConsistencyChecker:
    """Check only comparisons grounded by explicit observation fields."""

    @staticmethod
    def check_pair(left: Any, right: Any) -> bool | None:
        return identity_relation(left, right)

    def check(self, evidence: list[dict], requirements: Any = None, *,
              query: str = '', anchor: Any = None) -> dict:
        plan = _mapping(requirements)
        req = _mapping(plan.get('requirements', plan))
        warnings: list[dict] = []
        checks: dict[str, list] = {'identity': [], 'temporal': [], 'spatial': []}

        def warn(code: str, rows: list[dict], **details: Any) -> None:
            warning = {'code': code, 'evidence_ids': list(dict.fromkeys(_eid(row) for row in rows if _eid(row))),
                       'details': details}
            if warning not in warnings:
                warnings.append(warning)

        rows = []
        for item in evidence or []:
            if not isinstance(item, dict) or not item:
                warn('malformed_evidence', [], reason='Expected an observation mapping')
                continue
            rows.append(item)
            if strict_video_id(item) is None:
                warn('identity_unknown', [item], reason='Missing or conflicting canonical video ID')
        videos = {strict_video_id(item) for item in rows} - {None}
        if len(videos) > 1:
            warn('cross_video', rows, video_ids=sorted(videos))

        validation = EvidenceValidator().validate(requirements, rows) if requirements is not None and rows else {}
        chosen = _mapping(anchor if anchor is not None else validation.get('anchor'))
        if isinstance(anchor, str):
            chosen = next((item for item in rows if _eid(item) == anchor), {})
        if chosen and _eid(chosen):
            chosen = next((item for item in rows if _eid(item) == _eid(chosen)
                           and (not strict_video_id(chosen) or strict_video_id(item) == strict_video_id(chosen))), chosen)
        targets = [item for item in rows if _kind(item) in _TARGET_KINDS or 'target_identity' in _roles(item)]
        links = [item for item in rows if _kind(item) in _TRACK_KINDS | _EVENT_KINDS and item not in targets]
        multi = _mapping(req.get('comparison_requirement')).get('mode') == 'targets'
        target_req = _mapping(req.get('target_requirement'))
        categories = {category.lower() for category in _ids(target_req.get('categories'))}
        single_chain = bool(chosen and target_req and len(categories) <= 1 and not multi)
        if single_chain and categories:
            targets = [item for item in targets if
                       not (_get(item, 'label') or _get(item, 'category') or _get(item, 'object_category')) or
                       str(_get(item, 'label') or _get(item, 'category') or _get(item, 'object_category')).lower() in categories]
        # A target may name a subject while its event names the track node.
        # Follow only proven identity edges to preserve that explicit bridge.
        connected = {id(item): {id(item)} for item in rows}
        for left, right in combinations(rows, 2):
            if identity_relation(left, right) is True:
                merged = connected[id(left)] | connected[id(right)]
                for member in merged:
                    connected[member] = merged
        for link in links:
            comparisons = [True if id(link) in connected[id(target)] and not _explicit_identity_conflict(target, link)
                           else identity_relation(target, link)
                           for target in targets]
            if not comparisons:
                continue
            if single_chain and _eid(link) == _eid(chosen):
                # Each candidate offered for one anchored subject must agree.
                # One valid target cannot hide another target's wrong track.
                for target, relation in zip(targets, comparisons):
                    checks['identity'].append(relation)
                    if relation is not True:
                        warn('identity_unknown' if relation is None else 'identity_conflict', [target, link],
                             reason='Candidate target does not share the anchored answer subject')
                continue
            relation = True if True in comparisons else None if None in comparisons else False
            if relation is False and multi:
                # An unrelated event does not assert both compared targets are one.
                continue
            checks['identity'].append(relation)
            if relation is not True:
                warn('identity_unknown' if relation is None else 'identity_conflict', targets + [link],
                     reason='Answer chain requires an explicit shared subject')
        # Explicitly grouped rows make an additional same-subject assertion.
        for left, right in combinations(rows, 2):
            chain = left.get('chain_id')
            if chain and chain == right.get('chain_id'):
                relation = identity_relation(left, right)
                checks['identity'].append(relation)
                if relation is not True:
                    warn('identity_unknown' if relation is None else 'identity_conflict', [left, right], chain_id=chain)

        temporal_req = _mapping(req.get('temporal_requirement'))
        temporal_roles = {role for item in rows for role in _roles(item) if role in {'before_state', 'after_state'}}
        temporal_roles |= {role for role in ('before_state', 'after_state')
                           if role.split('_')[0] in str(temporal_req.get('relation', '')) or role.split('_')[0] in temporal_req}
        for role in sorted(temporal_roles):
            candidates = [item for item in rows if role in _roles(item)]
            if not candidates:
                satisfied = validation.get('satisfied', {}).get(role, [])
                candidates = [item for item in rows if _eid(item) in satisfied]
            if not candidates or not chosen or _span(chosen) is None:
                checks['temporal'].append(None)
                warn('temporal_unknown', candidates, role=role, reason='Missing timed anchor or state')
                continue
            for item in candidates:
                span, anchor_span = _span(item), _span(chosen)
                relation = None
                if span is None:
                    supplied = any(_get(item, key) is not None for key in ('timestamp', 'time_range', 'start_time', 'end_time'))
                    warn('invalid_time' if supplied else 'temporal_unknown', [item], role=role)
                elif not strict_video_id(item) or strict_video_id(item) != strict_video_id(chosen):
                    warn('temporal_unknown', [item, chosen], role=role, reason='Different or unknown video clocks')
                else:
                    relation = span[1] < anchor_span[0] if role == 'before_state' else span[0] > anchor_span[1]
                    if not relation:
                        warn('temporal_conflict', [item, chosen], role=role,
                             observed_time=list(span), anchor_time=list(anchor_span))
                checks['temporal'].append(relation)

        state_ids = {eid for role in ('before_state', 'after_state')
                     for eid in validation.get('satisfied', {}).get(role, [])}
        for left, right in combinations(rows, 2):
            # Region checks compare scene states or explicitly linked subjects.
            states = bool((_roles(left) & {'before_state', 'after_state'} or _eid(left) in state_ids) and
                          (_roles(right) & {'before_state', 'after_state'} or _eid(right) in state_ids))
            if not states and identity_relation(left, right) is not True:
                continue
            relation = _spatial_relation(left, right, compare_regions=states)
            checks['spatial'].append(relation)
            if relation is False:
                warn('spatial_conflict', [left, right], reason='Incompatible region or simultaneous subject location')
        causal = [_eid(item) for item in rows if _causal_support(item, rows)]
        if not causal:
            warn('causality_unverified', rows,
                 reason='Temporal sequence and query wording do not establish a source-backed causal relation')
        return {**{name + '_score': _score(values) for name, values in checks.items()},
                'warnings': warnings,
                'checks': {name: {'compared': sum(value is not None for value in values),
                                 'unknown': sum(value is None for value in values)} for name, values in checks.items()},
                'causal_evidence_ids': causal}
