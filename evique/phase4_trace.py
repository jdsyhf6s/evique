"""Final package assembly and audit output, outside the frozen retrieval layer."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

from .evidence_consistency import EvidenceConsistencyChecker
from .evidence_validator import EvidenceValidator
from .hybrid_identity import strict_video_id


CAUSALITY_POLICY = ('Do not infer causality from temporal order, co-occurrence, or the question. '
                    'Only report an explicitly source-backed causal relation, with its evidence ID; '
                    'otherwise describe observations and state that causality is unverified.')


def _key(row):
    return (strict_video_id(row), row.get('view') or row.get('evidence_type'),
            row.get('evidence_id') or row.get('id'))


def _io_path(path: Path) -> Path:
    absolute = str(path.resolve())
    if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
        return Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute)
    return path


def finalize_phase4_package(package: dict) -> dict:
    """Reconcile display supplements with the packer's selected evidence budget.

    Does not retrieve, expand, or alter original observations. Library callers
    should call this before format_package; the CLI does so automatically.
    """
    result = deepcopy(package)
    phase4 = result.get('phase4') or result.get('evidence_packing_metadata', {}).get('phase4')
    if phase4 is None:
        return result
    evidence = result.get('evidence', [])
    selected = {_key(row) for row in evidence}
    supplements = result.get('temporal_sequence_evidence', [])
    removed = [row for row in supplements if _key(row) not in selected]
    # Selected observations already appear in the evidence section. Render them
    # once: a display alias must not introduce different content or spend twice.
    result['temporal_sequence_evidence'] = []
    result['temporal_context_count'] = 0
    result['temporal_context_used'] = False
    phase4['unpacked_supplements_removed'] = removed
    phase4['duplicate_display_supplements_omitted'] = [row for row in supplements if _key(row) in selected]
    phase4['consistency_final'] = EvidenceConsistencyChecker().check(
        evidence, result.get('plan'), query=result.get('query', ''))
    validator = EvidenceValidator()
    status = validator.validate_status(result.get('plan') or {}, evidence)
    phase4['validation_final'] = status['validation']
    phase4['evidence_status_final'] = status
    result['phase4'] = phase4
    result['evidence_status'] = status
    result['missing_requirement_report'] = {
        key: status[key] for key in (
            'status', 'missing_requirements', 'supported_requirements',
            'missing_requirement_details', 'confidence', 'warnings',
        )
    }
    supported_ids = {
        evidence_id
        for mapping in (status.get('supporting_evidence_ids', {}),
                        status.get('partial_supporting_evidence_ids', {}))
        for ids in mapping.values()
        for evidence_id in ids
    }
    selected_ids = {str(row.get('evidence_id') or row.get('id') or row.get('node_id') or '')
                    for row in evidence}
    result['unsupported_evidence_ids'] = sorted(
        evidence_id for evidence_id in selected_ids if evidence_id and evidence_id not in supported_ids)
    result['unsupported_evidence_count'] = len(result['unsupported_evidence_ids'])
    if isinstance(result.get('construction'), dict):
        result['construction']['final_validation'] = phase4['validation_final']
        result['construction']['final_evidence_status'] = status
    constraints = result.setdefault('answer_constraints', [])
    if CAUSALITY_POLICY not in constraints:
        constraints.append(CAUSALITY_POLICY)
    if not phase4['validation_final']['complete']:
        constraints.append('Required evidence is incomplete: ' + ', '.join(phase4['validation_final']['missing']) +
                           '. Do not fill missing observations with assumptions.')
    if str(status['status']) == 'PARTIAL':
        constraints.append('The evidence package does not fully satisfy all requested aspects. '
                           'Answer only supported parts and explicitly state missing evidence.')
    elif str(status['status']) == 'INSUFFICIENT':
        constraints.append('The evidence package is insufficient to answer the requested aspects. '
                           'Describe only the supplied observations and explicitly abstain from unsupported conclusions.')
    return result


def synchronize_debug_trace(package: dict, context: str) -> None:
    """Finalize this invocation's debug artifact after package assembly."""
    if not package.get('phase4') or not package.get('trace_path'):
        return
    path = _io_path(Path(package['trace_path']))
    trace = json.loads(path.read_text(encoding='utf-8'))
    trace['pre_phase4_prompt_summary'] = trace['final_prompt_summary']
    trace['final_prompt_summary'] = {'context': context, 'context_chars': len(context),
                                     'llm_called': False, 'scope': 'formatted_evidence_context'}
    trace['selected_evidence'] = package.get('evidence', [])
    trace['temporal_sequence_evidence'] = package.get('temporal_sequence_evidence', [])
    trace['validation_result'] = package['phase4']['validation_final']
    trace['final_package'] = {k: v for k, v in package.items() if k != 'trace_path'}
    trace['phase4_applied'] = True
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_phase4_trace(package: dict, path: Path, *, query_metadata: dict | None = None) -> Path:
    phase4 = package.get('phase4') or package.get('evidence_packing_metadata', {}).get('phase4', {})
    construction = package.get('construction', {})
    trace = {'query': package.get('query'), 'query_metadata': query_metadata or {},
             'requirements': (package.get('plan') or {}).get('requirements'),
             'initial_candidates': construction.get('initial_candidates', []),
             'candidates': phase4.get('candidates', []),
             'expanded_graph': construction.get('graph_expansion', []),
             'consistency': {'before': phase4.get('consistency_before'),
                             'after_packing': phase4.get('consistency_after'),
                             'final': phase4.get('consistency_final')},
             'utility_ranking': phase4.get('utility_ranking', []),
             'package_before': phase4.get('package_before', []),
             'final_package': {k: v for k, v in package.items() if k != 'phase4_trace_path'},
             'comparison_scope': 'same packing candidates and budget; legacy versus utility selection',
             'generation': {'llm_called': False, 'answer_accuracy_measured': False}}
    path = Path(path)
    io_path = _io_path(path)
    io_path.parent.mkdir(parents=True, exist_ok=True)
    with io_path.open('x', encoding='utf-8') as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return path
