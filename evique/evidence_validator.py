"""Conservative observation coverage checks, independent of retrieval ranking.

Completeness means that observations cover the requested roles. It does not mean
that a question's proposed event or causal explanation has been proved true.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any

from .evidence_graph_executor import evidence_time_range
from .evidence_status import COMPLETE, INSUFFICIENT, PARTIAL, UNAVAILABLE
from .hybrid_identity import strict_video_id


def _mapping(value: Any) -> dict:
    return asdict(value) if is_dataclass(value) else value if isinstance(value, dict) else {}


def _values(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else [value] if value else []


def _sources(row: dict) -> list[dict]:
    record = row.get('record')
    return [record, row] if isinstance(record, dict) else [row]


def _get(row: dict, key: str, default: Any = None) -> Any:
    for source in _sources(row):
        attrs = source.get('attributes')
        for candidate in (source, attrs if isinstance(attrs, dict) else {}):
            if key in candidate and candidate[key] is not None:
                return candidate[key]
    return default


def _text(row: dict) -> str:
    for source in _sources(row):
        if 'raw_text' in source:
            return str(source['raw_text'] or '')
        values = [source.get(k) for k in ('caption', 'transcript', 'summary', 'evidence_text', 'motion_summary')]
        if any(values):
            return ' '.join(str(v) for v in values if v)
        if source.get('text'):
            return str(source['text'])
    return ''


def _kind(row: dict) -> str:
    return str(row.get('evidence_type') or row.get('view') or row.get('type') or '')


def _eid(row: dict) -> str:
    return str(row.get('evidence_id') or row.get('id') or row.get('node_id') or '')


def _tokens(value: Any) -> set[str]:
    words = set(re.findall(r'[a-z]+|[\u4e00-\u9fff]+', str(value).lower()))
    aliases = {'buses': 'bus', 'cars': 'car', 'vehicles': 'vehicle', 'trucks': 'truck',
               'people': 'person', 'persons': 'person', 'pedestrians': 'person', 'pedestrian': 'person'}
    return {aliases.get(word, word) for word in words}


def _category(row: dict, categories: list) -> bool:
    labels = [_get(row, k, '') for k in ('label', 'category', 'object_category')]
    observed = set().union(*(_tokens(label) for label in labels))
    return not categories or any(_tokens(category) <= observed for category in categories)


def _attributes(row: dict, required: Any) -> bool:
    if isinstance(required, dict):
        return all(str(_get(row, key, '')).lower() == str(value).lower() for key, value in required.items())
    observed = []
    for source in _sources(row):
        attrs = source.get('attributes', {})
        observed.extend(attrs.values() if isinstance(attrs, dict) else _values(attrs))
        observed.extend(source.get(k, '') for k in ('color', 'label', 'attribute_tags'))
    words = _tokens(observed)
    return all(_tokens(value) <= words for value in _values(required))


def _usable(row: dict) -> bool:
    for source in _sources(row):
        for key in ('quality_status', 'caption_status', 'status'):
            status = str(source.get(key, '')).lower()
            if any(word in status for word in ('degraded', 'fallback', 'missing', 'failed', 'error', 'invalid')):
                return False
    return not re.search(r'\b(?:caption unavailable|caption generation failed|no caption available|no native caption model output is available)\b', _text(row), re.I)


def _refs(row: dict) -> set[str]:
    keys = ('subject_id', 'track_id', 'object_id', 'supporting_ids', 'object_ids',
            'related_object_ids', 'related_track_ids', 'related_tracks', 'target_track_id', 'target_object_id')
    return {str(value) for key in keys for value in _values(_get(row, key)) if value}


def _transcript(row: dict) -> bool:
    for source in _sources(row):
        if any(any(word in str(source.get(key, '')).lower() for word in ('failed', 'missing', 'error', 'degraded', 'fallback'))
               for key in ('asr_status', 'transcript_status')):
            return False
    if str(_get(row, 'transcript', '')).strip():
        return True
    return _get(row, 'source_type') == 'transcript' and _usable(row) and bool(_text(row).strip())


_EVENTS = {'enter': r'enter(?:s|ed|ing)?|arriv(?:e|es|ed|ing)',
           'exit': r'exit(?:s|ed|ing)?|leav(?:e|es|ing)|left',
           'stop': r'stop(?:s|ped|ping)?', 'move': r'mov(?:e|es|ed|ing)',
           'congestion': r'congest(?:ed|ion)?|traffic jam'}


def _event(row: dict, requested: list) -> bool:
    observed = str(_get(row, 'event_type', '')).lower()
    if not observed:
        return False
    if re.search(r"\b(?:not|never|no|cannot|can't|doesn't|didn't|won't|if|might|may|could|would|hypothetical)\b", _text(row), re.I):
        return False
    return not requested or any(
        str(event).lower() == observed or
        re.search(r'(?:^|[_\s])(?:' + _EVENTS.get(str(event), re.escape(str(event))) + r')(?:$|[_\s])', observed)
        for event in requested)


def _explicit_event_target(row: dict, categories: list, events: list) -> bool:
    # Structured event labels are observation attributes; text needs a direct
    # subject/verb phrase, never a category anywhere in the same caption.
    if _category(row, categories) and categories:
        return True
    text = _text(row).lower()
    return any(re.search(r'\b' + re.escape(str(category)) + r'(?:es|s)?\s+(?:(?:is|was|has|had)\s+)?(?:' +
                         _EVENTS.get(str(event), re.escape(str(event))) + r')\b', text)
               for category in categories for event in events)


def _state(row: dict, context: Any) -> bool:
    state = _get(row, 'state_signature', {})
    caption = _get(row, 'caption', '')
    if (_get(row, 'source_type') == 'transcript' or _get(row, 'transcript')) and not caption and not state:
        return False
    text = str(caption) if caption else _text(row)
    if _kind(row) not in {'scope', 'segment', 'state', 'caption_context', 'caption', 'scene'} or not (text.strip() or state):
        return False
    requested = _tokens(context)
    observed = _tokens(text) | _tokens(state)
    traffic = {'traffic', 'vehicle', 'road', 'car', 'bus', 'truck', 'congestion', 'street', 'lane'}
    if requested & traffic:
        return bool(observed & traffic)
    requested -= {'state', 'before', 'after', 'context', 'scene', 'condition'}
    return not requested or bool(requested & observed)


class EvidenceValidator:
    """Validate raw evidence coverage using one coherent temporal anchor."""

    def validate(self, requirements: Any, evidence: list[dict], *,
                 required_evidence: list[str] | None = None, video_id: str | None = None) -> dict:
        req = _mapping(requirements)
        if 'requirements' in req:
            if required_evidence is None:
                required_evidence = req.get('required_evidence')
            req = _mapping(req['requirements'])
        target = _mapping(req.get('target_requirement'))
        anchor_req = _mapping(req.get('anchor_requirement'))
        temporal = _mapping(req.get('temporal_requirement'))
        comparison = _mapping(req.get('comparison_requirement'))
        event_req = _mapping(req.get('event_requirement'))
        if required_evidence is None:
            required_evidence = []
            if target:
                required_evidence.append('target_identity')
            if anchor_req:
                required_evidence.append('anchor_event')
            if temporal:
                relation = temporal.get('relation', 'before_after')
                required_evidence.extend(role for role in ('before_state', 'after_state') if role.split('_')[0] in relation)
            if event_req and not anchor_req:
                required_evidence.append('event_evidence')
            if comparison.get('mode') == 'targets':
                required_evidence.extend(['comparison_left', 'comparison_right'])
            if not required_evidence:
                required_evidence = ['scene_context']
        roles = list(dict.fromkeys(required_evidence))
        satisfied = {role: [] for role in roles}
        conflicts: list[dict] = []
        accepted = []
        transcript_rows = []
        seen: dict[str, dict] = {}
        subjects: dict[tuple, tuple[dict, str]] = {}
        videos = {strict_video_id(row) for row in evidence if isinstance(row, dict)} - {None}
        selected_video = video_id or (next(iter(videos)) if len(videos) == 1 else None)
        for row in evidence:
            if not isinstance(row, dict):
                continue
            eid = _eid(row)
            video = strict_video_id(row)
            if not video or not selected_video or video != selected_video:
                conflicts.append({'type': 'cross_video', 'evidence_id': eid, 'video_id': video, 'expected_video_id': selected_video})
                continue
            span = evidence_time_range(row)
            if span is None:
                explicit_time = any(source.get(key) is not None and source.get(key) != [None, None]
                                    for source in _sources(row) + [row.get('provenance') or {}]
                                    for key in ('timestamp', 'time_range', 'start_time', 'end_time'))
                if explicit_time:
                    conflicts.append({'type': 'invalid_time', 'evidence_id': eid})
                continue
            signature = {'time_range': span, 'text': _text(row),
                         **{key: _get(row, key) for key in ('label', 'category', 'color', 'event_type', 'attributes', 'subject_id')}}
            for key in ('label', 'category', 'color', 'event_type'):
                values = set()
                for source in _sources(row):
                    attrs = source.get('attributes') or {}
                    for mapping in (source, attrs if isinstance(attrs, dict) else {}):
                        value = mapping.get(key)
                        if value is not None and str(value).strip().lower() not in {'', 'unknown'}:
                            values.add(str(value).strip().lower())
                if len(values) > 1:
                    conflicts.append({'type': 'incompatible_observation', 'evidence_id': eid, 'attribute': key})
            if eid in seen and seen[eid] != signature:
                conflicts.append({'type': 'incompatible_observation', 'evidence_id': eid})
                continue
            seen[eid] = signature
            subject = _get(row, 'subject_id') or _get(row, 'object_id') or _get(row, 'track_id')
            if subject:
                identity = (video, str(subject), span, _kind(row))
                previous, previous_eid = subjects.get(identity, ({}, ''))
                for key in ('label', 'category', 'color', 'event_type'):
                    if previous.get(key) and signature.get(key) and previous[key] != signature[key]:
                        conflicts.append({'type': 'incompatible_observation', 'evidence_id': eid, 'attribute': key})
                        if previous_eid:
                            conflicts.append({'type': 'incompatible_observation',
                                              'evidence_id': previous_eid, 'attribute': key})
                subjects[identity] = (signature, eid)
            if eid and _usable(row) and not any(_eid(other) == eid for other in accepted):
                accepted.append(row)
            if eid and _transcript(row) and not any(_eid(other) == eid for other in transcript_rows):
                transcript_rows.append(row)
        conflict_ids = {str(item.get('evidence_id') or '') for item in conflicts
                        if item.get('evidence_id')}
        accepted = [row for row in accepted if _eid(row) not in conflict_ids]
        transcript_rows = [row for row in transcript_rows if _eid(row) not in conflict_ids]
        accepted.sort(key=lambda row: (evidence_time_range(row), _eid(row)))
        transcript_rows.sort(key=lambda row: (evidence_time_range(row), _eid(row)))
        identities = [row for row in accepted if _kind(row) in {'visual_object', 'target', 'visual_track', 'track', 'object'}
                      and _category(row, _values(target.get('categories')))
                      and _attributes(row, target.get('attributes'))
                      and any(_get(row, key) for key in ('label', 'category', 'object_category'))]
        all_targets = all(any(_category(row, [category]) for row in identities) for category in _values(target.get('categories')))
        if 'target_identity' in satisfied and all_targets:
            satisfied['target_identity'] = [_eid(row) for row in identities]
        categories = _values(anchor_req.get('target_categories') or target.get('categories'))
        events = _values(anchor_req.get('event_types') or event_req.get('event_types'))
        anchors = []
        for row in accepted:
            if not _event(row, events) or not all(_event(row, [event]) for event in events):
                continue
            linked = [identity for identity in identities if _refs(row) & (_refs(identity) | {_eid(identity)})]
            explicit_subjects = {str(value) for key in ('subject_id', 'track_id', 'object_id', 'target_track_id', 'target_object_id',
                                                        'related_track_ids', 'related_object_ids')
                                 for value in _values(_get(row, key)) if value}
            if explicit_subjects and identities and not linked:
                continue
            if not all(any(_category(identity, [category]) for identity in linked) or
                       (_explicit_event_target(row, [category], events) and _attributes(row, target.get('attributes')))
                       for category in categories):
                continue
            before = [item for item in accepted if _state(item, temporal.get('before') or comparison.get('aspect'))
                      and evidence_time_range(item)[1] < evidence_time_range(row)[0]]
            after = [item for item in accepted if _state(item, temporal.get('after') or comparison.get('aspect'))
                     and evidence_time_range(item)[0] > evidence_time_range(row)[1]]
            support = int(bool(before) and 'before_state' in roles) + int(bool(after) and 'after_state' in roles)
            anchors.append((support, row, before, after, linked))
        chosen = None
        if anchors:
            _, chosen, before, after, linked = sorted(anchors, key=lambda item: (-item[0], evidence_time_range(item[1]), _eid(item[1])))[0]
            for role, rows in (('anchor_event', [chosen]), ('before_state', before), ('after_state', after)):
                if role in satisfied:
                    satisfied[role] = [_eid(row) for row in rows]
            if linked and 'target_identity' in satisfied and all_targets and len(_values(target.get('categories'))) <= 1:
                satisfied['target_identity'] = [_eid(row) for row in linked]
        for role in roles:
            if role == 'scene_context':
                satisfied[role] = [_eid(row) for row in accepted if _kind(row) in {'scope', 'segment', 'scene', 'caption_context', 'caption'} and _text(row).strip()]
            elif role == 'transcript':
                satisfied[role] = [_eid(row) for row in transcript_rows]
            elif role == 'event_evidence':
                requested = _values(event_req.get('event_types'))
                if all(any(_event(row, [event]) for row in accepted) for event in requested):
                    satisfied[role] = [_eid(row) for row in accepted if _event(row, requested)]
            elif role in {'comparison_left', 'comparison_right'}:
                side = _mapping(comparison.get(role.split('_')[1]))
                satisfied[role] = [_eid(row) for row in identities if side and _category(row, _values(side.get('categories'))) and _attributes(row, side.get('attributes'))]
        if 'comparison_left' in satisfied and 'comparison_right' in satisfied:
            satisfied['comparison_right'] = [eid for eid in satisfied['comparison_right'] if eid not in satisfied['comparison_left']]
        missing = [role for role in roles if not satisfied[role]]
        return {'complete': bool(roles) and not missing and not conflicts, 'missing': missing,
                'conflicts': conflicts, 'satisfied': satisfied,
                'anchor': {'evidence_id': _eid(chosen), 'video_id': selected_video,
                           'time_range': list(evidence_time_range(chosen))} if chosen is not None else None}

    def validate_status(self, requirements: Any, evidence: list[dict], *,
                        required_evidence: list[str] | None = None,
                        video_id: str | None = None) -> dict:
        """Classify sufficiency without relaxing the existing coverage result.

        Strictly supported roles come only from :meth:`validate`. Partial IDs are
        separately labelled candidates that match a requested category/event but
        do not establish every attribute, link, or requested target. They may be
        shown with an explicit limitation, never counted as complete coverage.
        """
        validation = self.validate(requirements, evidence,
                                   required_evidence=required_evidence, video_id=video_id)
        req = _mapping(requirements)
        if 'requirements' in req:
            if required_evidence is None:
                required_evidence = req.get('required_evidence')
            req = _mapping(req['requirements'])
        roles = list(validation.get('satisfied', {}))
        if required_evidence is not None:
            roles = list(dict.fromkeys(required_evidence))
        supported = [role for role in roles if validation.get('satisfied', {}).get(role)]
        selected_video = video_id
        if selected_video is None:
            videos = {strict_video_id(row) for row in evidence if isinstance(row, dict)} - {None}
            selected_video = next(iter(videos)) if len(videos) == 1 else None
        conflict_ids = {str(item.get('evidence_id') or '')
                        for item in validation.get('conflicts', []) if item.get('evidence_id')}
        valid_rows = []
        for row in evidence:
            if (not isinstance(row, dict) or not _eid(row) or not selected_video
                    or strict_video_id(row) != selected_video or evidence_time_range(row) is None
                    or not _usable(row) or _eid(row) in conflict_ids):
                continue
            if not any(_eid(existing) == _eid(row) for existing in valid_rows):
                valid_rows.append(row)

        partial: dict[str, list[str]] = {}
        missing = list(validation.get('missing', []))
        target = _mapping(req.get('target_requirement'))
        anchor = _mapping(req.get('anchor_requirement'))
        event_req = _mapping(req.get('event_requirement'))
        comparison = _mapping(req.get('comparison_requirement'))
        target_categories = _values(target.get('categories'))

        for role in missing:
            matches: list[dict] = []
            if role == 'target_identity':
                matches = [row for row in valid_rows
                           if _kind(row) in {'visual_object', 'target', 'visual_track', 'track', 'object'}
                           and _category(row, target_categories)
                           and any(_get(row, key) for key in ('label', 'category', 'object_category'))]
            elif role in {'anchor_event', 'event_evidence'}:
                requirement = anchor if role == 'anchor_event' else event_req
                events = _values(requirement.get('event_types'))
                categories = _values(requirement.get('target_categories') or target.get('categories'))
                matches = [row for row in valid_rows if _event(row, events)
                           and (not categories or _category(row, categories))]
            elif role in {'comparison_left', 'comparison_right'}:
                side = _mapping(comparison.get(role.split('_')[1]))
                matches = [row for row in valid_rows
                           if side and _category(row, _values(side.get('categories')))]
            elif role == 'scene_context':
                matches = [row for row in valid_rows if _kind(row) in {
                    'scope', 'segment', 'scene', 'caption_context', 'caption'} and _text(row).strip()]
            elif role == 'transcript':
                matches = [row for row in valid_rows if _transcript(row)]
            ids = list(dict.fromkeys(_eid(row) for row in matches if _eid(row)))
            if ids:
                partial[role] = ids

        if validation.get('complete'):
            status = COMPLETE
        elif not valid_rows:
            status = UNAVAILABLE
        else:
            core = [role for role in roles if role not in {'before_state', 'after_state'}]
            if not core:
                core = list(roles)
            core_supported = bool(core) and all(
                validation.get('satisfied', {}).get(role) or partial.get(role) for role in core)
            relevant = bool(supported or partial)
            status = PARTIAL if core_supported and relevant else INSUFFICIENT

        role_count = len(roles)
        coverage = len(supported) / role_count if role_count else 0.0
        partial_only = sum(1 for role in partial if role not in supported)
        if status == COMPLETE:
            confidence = 1.0
        elif status == UNAVAILABLE:
            confidence = 0.0
        else:
            confidence = min(.99, (len(supported) + .5 * partial_only) / role_count) if role_count else 0.0
        warnings = []
        if missing:
            warnings.append({'code': 'missing_requirements', 'requirements': missing})
        if partial:
            warnings.append({'code': 'partial_requirement_support',
                             'requirements': list(partial)})
        if validation.get('conflicts'):
            warnings.append({'code': 'evidence_conflicts',
                             'count': len(validation['conflicts'])})
        if status == INSUFFICIENT:
            warnings.append({'code': 'insufficient_answer_support',
                             'reason': 'Valid observations do not support every core requirement'})
        if status == UNAVAILABLE:
            warnings.append({'code': 'no_valid_evidence'})
        requirement_details = {}
        for role in missing:
            if role == 'target_identity':
                detail = target
            elif role == 'anchor_event':
                detail = anchor
            elif role in {'before_state', 'after_state'}:
                detail = {'temporal_role': role.split('_')[0], **_mapping(req.get('temporal_requirement'))}
            elif role == 'event_evidence':
                detail = event_req
            elif role in {'comparison_left', 'comparison_right'}:
                detail = _mapping(comparison.get(role.split('_')[1]))
            elif role == 'scene_context':
                detail = {'requirement': 'scene_context'}
            elif role == 'transcript':
                detail = {'requirement': 'transcript'}
            else:
                detail = {'requirement': role}
            requirement_details[role] = detail
        return {
            'status': status,
            'coverage_score': round(coverage, 6),
            'missing_requirements': missing,
            'supported_requirements': supported,
            'confidence': round(confidence, 6),
            'warnings': warnings,
            'missing_requirement_details': requirement_details,
            'supporting_evidence_ids': {
                role: list(validation.get('satisfied', {}).get(role, []))
                for role in supported
            },
            'partial_supporting_evidence_ids': partial,
            'valid_evidence_ids': [_eid(row) for row in valid_rows],
            'validation': validation,
        }
