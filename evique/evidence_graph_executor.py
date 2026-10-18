"""Bounded traversal of stored evidence links without synthesizing observations."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import math
from typing import Any

from .hybrid_identity import strict_video_id
from .schema import normalize_evidence


def _sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    record = row.get('record')
    return [source for source in (row, record, row.get('provenance'),
            record.get('provenance') if isinstance(record, dict) else None)
            if isinstance(source, dict)]


def evidence_time_range(row: dict[str, Any]) -> tuple[float, float] | None:
    """Return an original point/range, rejecting malformed or conflicting times.

    Null placeholders are absent values. Every supplied numeric time must be
    finite and nonnegative, repeated fields must agree, and a point must fall
    within a supplied range. No time is inferred from neighboring evidence.
    """
    points: set[float] = set()
    starts: set[float] = set()
    ends: set[float] = set()

    def add(values: set[float], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            raise ValueError('boolean time')
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError('invalid time')
        values.add(number)

    try:
        for source in _sources(row):
            add(points, source.get('timestamp'))
            add(starts, source.get('start_time'))
            add(ends, source.get('end_time'))
            interval = source.get('time_range')
            if interval is not None and interval != []:
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    return None
                add(starts, interval[0])
                add(ends, interval[1])
        if any(len(values) > 1 for values in (points, starts, ends)):
            return None
        if bool(starts) != bool(ends):
            return None
        if starts:
            start, end = next(iter(starts)), next(iter(ends))
            if start > end or any(point < start or point > end for point in points):
                return None
            return start, end
        if points:
            point = next(iter(points))
            return point, point
    except (ValueError, TypeError, OverflowError):
        return None
    return None


# Only explicit structural references participate; label and temporal neighbors
# intentionally do not create connectivity.
_REFERENCE_FIELDS = {
    'track_id': {'track', 'visual_track'},
    'object_ids': {'target', 'visual_object'},
    'related_object_ids': {'target', 'visual_object'},
    'related_track_ids': {'track', 'visual_track'},
    'related_tracks': {'track', 'visual_track'},
    'related_event_ids': {'event', 'visual_event', 'adaptive_event'},
    'related_segment_ids': {'scope', 'segment', 'caption_context'},
    'segment_ids': {'scope', 'segment', 'caption_context'},
    'segment_id': {'scope', 'segment', 'caption_context'},
    'target_object_id': {'target', 'visual_object'},
    'related_object_id': {'target', 'visual_object'},
    'target_track_id': {'track', 'visual_track'},
    'related_track_id': {'track', 'visual_track'},
    'supporting_ids': None,
    'support_refs': None,
    'supporting_refs': None,
}
_TEMPORAL_RELATIONS = {'followed_by', 'precedes', 'before', 'after', 'next',
                       'previous', 'temporal_neighbor', 'temporal_adjacent',
                       'next_event', 'previous_event'}
NodeKey = tuple[str, str, str]


def _identity(row: dict[str, Any]) -> NodeKey | None:
    video = strict_video_id(row)
    sources = _sources(row)
    view = next((str(s.get('view') or s.get('evidence_type')) for s in sources
                 if s.get('view') or s.get('evidence_type')), '')
    identifier = next((str(s.get('id') or s.get('evidence_id') or s.get('node_id'))
                       for s in sources if s.get('id') or s.get('evidence_id') or s.get('node_id')), '')
    return (video, view, identifier) if video and view and identifier else None


def _normalized(row: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(row)
    key = _identity(item)
    if key:
        item.setdefault('id', key[2])
        item.setdefault('view', key[1])
    if not isinstance(item.get('record'), dict):
        item['record'] = deepcopy(row)
    return normalize_evidence(item)


def _observation(row: dict[str, Any]) -> dict[str, Any]:
    original = deepcopy(row.get('record') if isinstance(row.get('record'), dict) else row)
    original.pop('view', None)
    return original


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1 and math.isfinite(value) else None


def _subject_refs(row: dict[str, Any]) -> set[tuple[str, str]]:
    """Observed subject identifiers; labels and similarity never prove identity."""
    refs: set[tuple[str, str]] = set()
    key = _identity(row)
    original = row.get('record') if isinstance(row.get('record'), dict) else row
    for source in _sources(original):
        fields = [source]
        if isinstance(source.get('attributes'), dict):
            fields.append(source['attributes'])
        for values in fields:
            for namespace, names in (
                ('object', ('object_id', 'object_ids', 'related_object_ids')),
                ('track', ('track_id', 'related_track_ids', 'related_tracks')),
                ('subject', ('subject_id',)),
            ):
                for name in names:
                    value = values.get(name)
                    entries = value if isinstance(value, (list, tuple)) else [value]
                    for entry in entries:
                        if not isinstance(entry, bool) and isinstance(entry, (str, int)) and str(entry).strip():
                            # The canonical contract permits subject_id to name
                            # either an object or a track.
                            namespaces = ('object', 'track') if namespace == 'subject' else (namespace,)
                            refs.update((kind, str(entry)) for kind in namespaces)
        if key and key[1] in {'target', 'visual_object', 'track', 'visual_track'}:
            namespace = 'object' if key[1] in {'target', 'visual_object'} else 'track'
            for name in ('id', 'evidence_id', 'node_id'):
                if source.get(name):
                    refs.add((namespace, str(source[name])))
    return refs


class EvidenceGraphExecutor:
    """Follow at most three explicit graph hops, within one canonical video."""

    def __init__(self, records: list[dict], relations: list[dict] = (), *,
                 max_hops: int = 3, max_nodes: int = 100,
                 min_edge_confidence: float = 0.5):
        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or not 0 <= max_hops <= 3:
            raise ValueError('max_hops must be an integer between 0 and 3')
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError('max_nodes must be a positive finite integer')
        threshold = _confidence(min_edge_confidence)
        if threshold is None:
            raise ValueError('min_edge_confidence must be a finite number between 0 and 1')
        self.min_edge_confidence = threshold
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.records = deepcopy(records)
        self.relations = deepcopy(relations)

    def expand(self, candidates: list[dict], *, video_id: str | None = None) -> dict:
        videos = {v for row in candidates if (v := strict_video_id(row))}
        if video_id is None and len(videos) > 1:
            raise ValueError('Mixed seed videos require an explicit video_id')
        selected = str(video_id).strip() if video_id is not None else next(iter(videos), None)
        rejected: list[dict] = []
        rejection_keys = set()

        def reject(identifier: Any, reason: str, **details: Any) -> None:
            entry = {'id': str(identifier or ''), 'reason': reason, **details}
            signature = repr(entry)
            if signature not in rejection_keys:
                rejected.append(entry)
                rejection_keys.add(signature)

        nodes: dict[NodeKey, dict] = {}
        originals: dict[NodeKey, dict] = {}
        inputs: dict[NodeKey, dict] = {}
        conflicts: set[NodeKey] = set()
        for row in self.records:
            key = _identity(row)
            if key is None:
                reject(row.get('id'), 'invalid_video_identity' if not strict_video_id(row) else 'invalid_node_identity')
                continue
            original = _observation(row)
            if key in originals and originals[key] != original:
                conflicts.add(key)
                reject(key[2], 'conflicting_duplicate', video_id=key[0], view=key[1])
            else:
                originals[key] = original
                inputs[key] = deepcopy(row)
                nodes[key] = _normalized(row)

        seed_keys: list[NodeKey] = []
        for candidate in candidates:
            key = _identity(candidate)
            if key is None:
                reject(candidate.get('id'), 'invalid_video_identity' if not strict_video_id(candidate) else 'invalid_node_identity')
                continue
            if key[0] != selected:
                reject(key[2], 'wrong_video', video_id=key[0])
                continue
            if key in conflicts:
                continue
            if evidence_time_range(candidate) is None:
                reject(key[2], 'invalid_time', video_id=key[0], hop=0)
                continue
            if key in originals:
                if isinstance(candidate.get('record'), dict):
                    conflicting = _observation(candidate) != originals[key]
                else:
                    conflicting = any(candidate[field] != value for field, value in originals[key].items()
                                      if field in candidate and field not in {'score', 'rank', 'source'})
                if conflicting:
                    conflicts.add(key)
                    reject(key[2], 'conflicting_duplicate', video_id=key[0], view=key[1])
                    continue
            if key not in nodes:
                nodes[key] = _normalized(candidate)
                originals[key] = _observation(candidate)
                inputs[key] = deepcopy(candidate)
            else:
                # A hit's scores and attribution survive alongside the stored raw record.
                merged = {**nodes[key], **deepcopy(candidate), 'record': deepcopy(nodes[key]['record'])}
                if evidence_time_range(merged) is None:
                    reject(key[2], 'invalid_time', video_id=key[0], hop=0)
                    continue
                nodes[key] = normalize_evidence(merged)
            if key not in seed_keys:
                seed_keys.append(key)

        aliases: dict[tuple[str, str], set[NodeKey]] = defaultdict(set)
        for key, row in nodes.items():
            for source in _sources(row):
                fields = ('id', 'evidence_id', 'node_id')
                if key[1] in {'scope', 'segment', 'caption_context'}:
                    fields += ('segment_id',)
                for field in fields:
                    if source.get(field) is not None:
                        aliases[(key[0], str(source[field]))].add(key)

        def resolve(reference: Any, video: str, views: set[str] | None = None, *,
                    require_global: bool = False) -> NodeKey | None:
            if isinstance(reference, dict):
                ref_video = strict_video_id(reference)
                if any('video_id' in s for s in _sources(reference)) and ref_video != video:
                    reject(reference.get('id'), 'wrong_video', video_id=ref_video)
                    return None
                if ref_video is not None:
                    require_global = False
                if reference.get('view') or reference.get('evidence_type'):
                    views = {str(reference.get('view') or reference['evidence_type'])}
                reference = reference.get('id') or reference.get('evidence_id') or reference.get('node_id')
            if reference is None:
                return None
            if require_global and len({vid for vid, alias in aliases if alias == str(reference)}) > 1:
                reject(reference, 'ambiguous_reference', video_id=video, scope='unscoped_relation')
                return None
            matches = aliases.get((video, str(reference)), set())
            if views is not None:
                matches = {key for key in matches if key[1] in views}
            if len(matches) != 1:
                reason = 'ambiguous_reference' if matches else ('wrong_video' if any(
                    alias == str(reference) and vid != video for vid, alias in aliases) else 'missing_reference')
                reject(reference, reason, video_id=video)
                return None
            key = next(iter(matches))
            if key in conflicts:
                reject(key[2], 'conflicting_duplicate', video_id=video, view=key[1])
                return None
            return key

        adjacency: dict[NodeKey, list[tuple[NodeKey, str, bool, str, dict]]] = defaultdict(list)
        explicit_pairs: set[frozenset[NodeKey]] = set()
        node_checks: dict[NodeKey, tuple[tuple[float, float] | None, set[tuple[str, str]]]] = {}

        def checks(key: NodeKey) -> tuple[tuple[float, float] | None, set[tuple[str, str]]]:
            # Nodes are immutable after seed merging. Parse each original ID
            # list and time range once, even for tracks with thousands of links.
            if key not in node_checks:
                node_checks[key] = (evidence_time_range(nodes[key]), _subject_refs(inputs[key]))
            return node_checks[key]

        def connect(source: NodeKey, target: NodeKey, relation: str, origin: str,
                    edge: dict[str, Any]) -> None:
            if source == target or source in conflicts or target in conflicts:
                return
            # Absence keeps legacy structural links usable; supplied values must
            # be valid. This is link confidence, never observation confidence.
            raw_confidence = edge.get('confidence', 1.0)
            confidence = _confidence(raw_confidence)
            declared = edge.get('constraints', {})
            source_time, source_refs = checks(source)
            target_time, target_refs = checks(target)
            evaluation = {
                'same_video': source[0] == target[0] == selected,
                'same_object': bool(source_refs & target_refs) if source_refs and target_refs else None,
                'time_overlap': max(source_time[0], target_time[0]) <= min(source_time[1], target_time[1])
                                if source_time is not None and target_time is not None else None,
            }
            details = {'source': source[2], 'relation': relation, 'origin': origin,
                       'confidence': confidence, 'constraints': deepcopy(declared),
                       'constraint_evaluation': evaluation}
            if confidence is None:
                reject(target[2], 'invalid_edge_confidence', **details)
                return
            if confidence < self.min_edge_confidence:
                reject(target[2], 'low_edge_confidence', **details)
                return
            if not isinstance(declared, dict) or any(
                name not in evaluation or not isinstance(required, bool)
                for name, required in declared.items()
            ):
                reject(target[2], 'invalid_edge_constraints', **details)
                return
            constraints = {'same_video': True, **declared}
            if any(required is False or evaluation[name] is not True
                   for name, required in constraints.items()):
                reject(target[2], 'edge_constraint_failed', **details)
                return
            metadata = {'confidence': confidence, 'constraints': constraints,
                        'constraint_evaluation': evaluation}
            forward = (target, relation, False, origin, metadata)
            reverse = (source, relation, True, origin, metadata)
            if forward not in adjacency[source]:
                adjacency[source].append(forward)
            if reverse not in adjacency[target]:
                adjacency[target].append(reverse)

        for edge in self.relations:
            relation = str(edge.get('relation') or '')
            if not relation or relation.lower() in _TEMPORAL_RELATIONS:
                continue
            edge_video = strict_video_id(edge)
            if any(s.get('video_id') is not None for s in _sources(edge)) and edge_video != selected:
                reject(edge.get('target'), 'wrong_video', video_id=edge_video)
                continue
            source = resolve(edge.get('source'), selected, require_global=edge_video is None)
            target = resolve(edge.get('target'), selected, require_global=edge_video is None)
            if source is not None and target is not None:
                explicit_pairs.add(frozenset((source, target)))
                connect(source, target, relation, 'evidence_relations', edge)

        for key, row in nodes.items():
            if key[0] != selected or key in conflicts:
                continue
            # Read explicit source fields, not supporting_ids added by normalization.
            for source in _sources(inputs[key]):
                attributes = source.get('attributes')
                sources = [source, attributes] if isinstance(attributes, dict) else [source]
                for fields in sources:
                    for field, views in _REFERENCE_FIELDS.items():
                        value = fields.get(field)
                        if value is None:
                            continue
                        refs = value if isinstance(value, (list, tuple)) else [value]
                        for ref in refs:
                            target = resolve(ref, key[0], views)
                            if target is not None and frozenset((key, target)) not in explicit_pairs:
                                connect(key, target, field, 'structured_reference',
                                        ref if isinstance(ref, dict) else {})

        expanded: list[dict] = []
        trace: list[dict] = []
        visited: set[NodeKey] = set()
        queue = deque()
        truncated = False
        for key in seed_keys:
            if key in conflicts:
                continue
            if len(visited) >= self.max_nodes:
                truncated = True
                break
            visited.add(key)
            expanded.append(deepcopy(nodes[key]))
            queue.append((key, 0, [key[2]]))
        while queue:
            source, hop, path = queue.popleft()
            for target, relation, reverse, origin, metadata in adjacency[source]:
                if target in visited:
                    continue
                if checks(target)[0] is None:
                    reject(target[2], 'invalid_time', video_id=target[0], source=source[2], hop=hop + 1)
                    continue
                if hop >= self.max_hops or len(visited) >= self.max_nodes:
                    truncated = True
                    continue
                visited.add(target)
                target_path = path + [target[2]]
                expanded.append(deepcopy(nodes[target]))
                trace.append({'source': source[2], 'target': target[2], 'relation': relation,
                              'hop': hop + 1, 'video_id': target[0], 'path': target_path,
                              'source_view': source[1], 'target_view': target[1],
                              'reverse': reverse, 'origin': origin, **deepcopy(metadata)})
                queue.append((target, hop + 1, target_path))
        return {'expanded_evidence': expanded, 'expansion_trace': trace,
                'rejected': rejected, 'max_hops': self.max_hops, 'truncated': truncated}
