"""Reproducible evaluation orchestration around frozen algorithms."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .manifest import build_manifest, file_hash, hash_index, sanitize, verify_release, write_manifest
from .metrics import METRIC_NAMES, aggregate_results, compute_metrics
from .protocol import ANSWER_SYSTEM, PROTOCOL_VERSION, CachedVideoRAG, JsonChatClient, digest, parse_answer, shared_messages
from ..embedding_index import EmbeddingIndex, FastEmbedEncoder
from ..phase4_trace import finalize_phase4_package, _io_path
from ..planner import QueryPlanner
from ..retriever import EvidenceRetriever
from ..utils.query_io import load_query_file


METHODS = ['VideoRAG', 'Phase2', 'Phase3', 'Phase4', 'Full EVIQUE']
METHOD_DEFINITIONS = {
    'VideoRAG': 'Authentic cached retrieval contexts; historical answers discarded',
    'Phase2': 'Frozen hybrid retrieval; construction disabled; legacy packing',
    'Phase3': 'Matched construction candidates with legacy packing; frozen graph validity rules',
    'Phase4': 'Frozen construction plus utility packing and final assembly',
    'Full EVIQUE': 'Exact Phase4 algorithm alias; no additional algorithm or independent replicate',
}


@contextmanager
def _environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if not key.startswith('EVIQUE_'):
                raise ValueError('Only EVIQUE retrieval environment overrides are accepted')
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write(path, value):
    path = _io_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')


def _read(path):
    return json.loads(_io_path(Path(path)).read_text(encoding='utf-8-sig'))


def _trace_name(query, method):
    return digest([query['dataset'], query['query_id'], query['video_id'], method])[:20] + '.json'


def _save_tables(directory, rows):
    aggregates = aggregate_results(rows)
    fields = ['method', 'row_count', *METRIC_NAMES]
    for metric in METRIC_NAMES:
        fields.extend([metric + '_status', metric + '_n', metric + '_unknown', metric + '_denominator'])
    flattened = []
    for aggregate in aggregates:
        row = {key: aggregate.get(key) for key in ['method', 'row_count', *METRIC_NAMES]}
        for metric in METRIC_NAMES:
            details = aggregate['metric_details'][metric]
            for suffix, key in (('status', 'status'), ('n', 'observed_count'), ('unknown', 'unknown_count'), ('denominator', 'denominator')):
                row[metric + '_' + suffix] = details.get(key)
        flattened.append(row)
    for name, methods in [('main_results.csv', {'VideoRAG', 'Full EVIQUE'}),
                           ('ablation_results.csv', {'Phase2', 'Phase3', 'Phase4'})]:
        with (directory / name).open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in flattened if row['method'] in methods)
    _write(directory / 'metric_details.json', aggregates)
    return aggregates


class _Arms:
    def __init__(self, config):
        self.config = config
        self.video = CachedVideoRAG(Path(config['videorag']['cache_dir'])) if config.get('videorag') else None
        self.engines = {}
        self.dense = None
        if config.get('embedding_index_dir'):
            manifest = _read(Path(config['embedding_index_dir']) / 'manifest.json')
            encoder = FastEmbedEncoder(manifest['model_id'], cache_dir=config.get('embedding_cache_dir') or manifest.get('cache_dir'),
                                       local_files_only=True, threads=config.get('embedding_threads', 2))
            self.dense = EmbeddingIndex.load(Path(config['embedding_index_dir']), encoder, source_dir=Path(config['index_dir']))
        elif not config.get('allow_missing_embedding', False):
            raise ValueError('A dense embedding sidecar is required; diagnostic runs must explicitly allow_missing_embedding')

    def _engine(self, construction):
        if construction not in self.engines:
            self.engines[construction] = EvidenceRetriever(Path(self.config['index_dir']),
                max_evidence=self.config.get('max_evidence', 18), token_budget=self.config.get('token_budget', 12000),
                embedding_index=self.dense, hybrid_enabled=True, construction_enabled=construction)
        return self.engines[construction]

    def retrieve(self, query, method, cache):
        if method == 'VideoRAG':
            if self.video is None:
                raise ValueError('Authentic VideoRAG cache is not configured')
            evidence, provenance = self.video.retrieve(query['query'], query['video_id'])
            return {'evidence': evidence}, provenance
        key = 'Phase2' if method == 'Phase2' else 'construction'
        if key not in cache:
            engine = self._engine(key == 'construction')
            cache[key] = engine.retrieve(query['query'], {**query.get('metadata', {}),
                'query_id': query['query_id'], 'video_id': query['video_id'], 'dataset': query['dataset']})
        raw = cache[key]
        if method == 'Phase3':
            phase4 = raw.get('phase4', {})
            if not phase4:
                raise ValueError('Construction did not produce a legacy packing comparison')
            package = {'evidence': deepcopy(phase4['package_before']),
                       'temporal_sequence_evidence': deepcopy(raw.get('temporal_sequence_evidence', [])),
                       'construction': raw.get('construction'), 'plan': raw.get('plan'),
                       'hybrid_retrieval': raw.get('hybrid_retrieval')}
            return package, {'backend': 'matched_candidate_legacy_packing', 'shared_retrieval': 'Phase4',
                             'graph_rules': 'frozen Phase4 edge validity retained'}
        if method in {'Phase4', 'Full EVIQUE'}:
            return finalize_phase4_package(raw), {'backend': 'frozen_evique', 'algorithm_alias': 'Phase4'}
        return raw, {'backend': 'frozen_evique', 'construction': False}


def run_benchmark(config: dict, output_dir: Path, *, stage='pilot', pilot_dir: Path | None = None) -> dict:
    """Run ten queries first; full runs require a matching completed pilot.

    Serial by design: the frozen implementation reads process environment flags.
    Evidence-only mode never generates placeholder answers or subjective metrics.
    """
    config = deepcopy(config)
    root = Path(__file__).resolve().parents[2]
    methods = config.get('methods', METHODS)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError('Unknown or duplicate experiment method')
    methods = sorted(methods, key=METHODS.index)
    mode = config.get('mode', 'generate')
    if mode not in {'generate', 'evidence_only'} or stage not in {'pilot', 'full'}:
        raise ValueError('Unknown experiment mode/stage')
    client = JsonChatClient(config.get('model', {})) if mode == 'generate' else None
    judge = JsonChatClient(config['judge']) if mode == 'generate' and config.get('judge') else None
    for name, service in (('model', client), ('judge', judge)):
        if service is not None:
            settings = config.get(name, {})
            config['effective_' + name] = sanitize({**settings,
                'model': getattr(service, 'model', settings.get('model')),
                'base_url': getattr(service, 'base_url', None) or settings.get('base_url')
                    or os.getenv(settings.get('base_url_env', 'EVIQUE_LLM_BASE_URL')),
                'temperature': settings.get('temperature', 0), 'max_tokens': settings.get('max_tokens', 1024)})
    if config.get('release_lock'):
        release = verify_release(config['release_lock'], root)
        if not release['valid']:
            raise ValueError('Frozen release source changed')
    elif not config.get('allow_unfrozen', False):
        raise ValueError('A frozen release lock is required')
    else:
        release = {'valid': False, 'status': 'explicit_unfrozen_diagnostic'}
    queries = [q.to_dict() for q in load_query_file(config['query_file']).queries]
    if len(queries) < 10:
        raise ValueError('A pilot requires at least ten queries')
    seen = set()
    for query in queries:
        query['metadata']['source_query_video_id'] = query['video_id']
        query['video_id'] = config.get('video_id_map', {}).get(query['video_id'], query['video_id'])
        key = (query['dataset'], query['query_id'], query['video_id'])
        if not query['video_id'] or key in seen:
            raise ValueError('Query video identity missing or duplicate query identity')
        seen.add(key)
    ambient = {key: value for key, value in os.environ.items() if key.startswith('EVIQUE_')
               and not any(part in key for part in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'LLM', 'JUDGE'))}
    effective_env = {**ambient, **config.get('retrieval_environment', {}), 'EVIQUE_UTILITY_PACKING': '1'}
    config['effective_retrieval_environment'] = effective_env
    embedding = hash_index(config['embedding_index_dir']) if config.get('embedding_index_dir') else {'status': 'not_configured'}
    if config.get('embedding_index_dir'):
        embedding['manifest'] = _read(Path(config['embedding_index_dir']) / 'manifest.json')
    manifest = build_manifest(root, config['index_dir'], model=config.get('model', {}), embedding=embedding,
                              config=config, query_file=config['query_file'])
    baseline = hash_index(config['videorag']['cache_dir']) if config.get('videorag') else None
    manifest.update(method_definitions=METHOD_DEFINITIONS, release=release, baseline=baseline,
                    stage=stage, execution_mode=mode, prompt_version=PROTOCOL_VERSION,
                    prompt_template_hash=digest(ANSWER_SYSTEM))
    protocol_hash = digest({'config': manifest['config_sha256'], 'index': manifest['index']['sha256'],
                            'embedding': embedding, 'query': manifest['query'], 'baseline': baseline,
                            'code': manifest['code']['sha256'], 'prompt': digest(ANSWER_SYSTEM),
                            'runtime': manifest['runtime']})
    manifest['protocol_hash'] = protocol_hash
    prior_summary = None
    if stage == 'full':
        if pilot_dir is None:
            raise ValueError('Full benchmark requires a successful ten-query pilot')
        prior_summary = _read(Path(pilot_dir) / 'summary.json')
        if prior_summary.get('protocol_hash') != protocol_hash:
            raise ValueError('Full benchmark protocol differs from pilot')
        if prior_summary.get('query_count') != 10 or prior_summary.get('error_count') or prior_summary.get('status') not in {'complete', 'evidence_only_complete'}:
            raise ValueError('Full benchmark requires a successful ten-query pilot')
        expected_names = {_trace_name(query, method) for query in queries[:10] for method in methods}
        hashes = prior_summary.get('trace_hashes', {})
        if set(hashes) != expected_names or prior_summary.get('result_count') != len(expected_names):
            raise ValueError('Pilot trace integrity/count mismatch')
        for query in queries[:10]:
            for method in methods:
                name = _trace_name(query, method)
                path = Path(pilot_dir) / 'per_query_trace' / name
                if not path.is_file() or file_hash(path) != hashes[name]:
                    raise ValueError('Pilot trace integrity check failed')
                trace = _read(path)
                expected_status = 'evidence_only' if mode == 'evidence_only' else 'complete'
                if (trace.get('query') != query or trace.get('method') != method
                        or trace.get('status') != expected_status or trace.get('protocol_hash') != protocol_hash):
                    raise ValueError('Pilot trace integrity/identity/status mismatch')
    selected_queries = queries[:10] if stage == 'pilot' else queries
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Experiment output already exists; choose a fresh run directory')
    output.mkdir(parents=True)
    write_manifest(manifest, output / 'experiment_manifest.json')
    _write(output / 'queries.json', selected_queries)
    summaries, errors, reused = [], 0, 0
    started = time.monotonic()
    with _environment(effective_env):
        arms = _Arms(config)
        for index, query in enumerate(selected_queries):
            traces, retrieval_cache = {}, {}
            reuse = prior_summary is not None and index < 10
            if reuse:
                reused += 1
            for method in methods:
                if reuse:
                    trace = _read(Path(pilot_dir) / 'per_query_trace' / _trace_name(query, method))
                    if trace.get('query') != query or trace.get('protocol_hash') != protocol_hash:
                        raise ValueError('Pilot trace identity/protocol mismatch')
                    trace['reused_from_pilot'] = str(Path(pilot_dir).resolve())
                    traces[method] = trace
                    continue
                began = time.monotonic()
                requirements = QueryPlanner().build_requirement(query['query'])
                trace = {'query': query, 'method': method, 'protocol_hash': protocol_hash,
                         'requirements': requirements.to_dict(), 'answer': None, 'judgement': None,
                         'status': 'evidence_only' if mode == 'evidence_only' else 'complete'}
                try:
                    package, provenance = arms.retrieve(query, method, retrieval_cache)
                    items = package.get('evidence', []) + package.get('temporal_sequence_evidence', [])
                    # Display each record once without reconstructing hidden facts.
                    unique = {}; [unique.setdefault((item.get('video_id'), item.get('view'), item.get('evidence_id') or item.get('id')), item) for item in items]
                    messages, visible, prompt = shared_messages(query['query'], list(unique.values()),
                                                               max_context_chars=config.get('context_char_budget', 12000))
                    trace.update(package=package, provenance=provenance, messages=messages,
                                 visible_evidence=visible, prompt_metadata=prompt)
                    if client:
                        alias = traces.get('Phase4') if method == 'Full EVIQUE' else None
                        if alias and alias.get('prompt_metadata', {}).get('messages_hash') == prompt['messages_hash'] and alias.get('answer'):
                            trace['answer'], trace['generation'] = deepcopy(alias['answer']), deepcopy(alias['generation'])
                            trace['generation_reused_from'] = 'Phase4: exact algorithm alias'
                        else:
                            generation = client.complete(messages)
                            trace['generation'] = generation
                            trace['answer'] = parse_answer(generation['content'], {item['evidence_id'] for item in visible})
                        if judge and alias and alias.get('judgement'):
                            trace['judgement'] = deepcopy(alias['judgement'])
                        elif judge:
                            from .judging import judge_answer
                            trace['judgement'] = judge_answer(judge, query['query'], visible, trace['answer'])
                except Exception as exc:
                    errors += 1
                    trace.update(status='error', error={'type': type(exc).__name__, 'message': sanitize(str(exc))})
                    trace.setdefault('visible_evidence', [])
                trace['runtime_seconds'] = time.monotonic() - began
                traces[method] = trace
            if judge and not reuse and traces.get('VideoRAG', {}).get('answer'):
                from .judging import judge_preference
                reference = traces['VideoRAG']
                for method, trace in traces.items():
                    if method == 'VideoRAG' or not trace.get('answer') or (method == 'Full EVIQUE' and 'Phase4' in traces):
                        continue
                    try:
                        preference = judge_preference(judge, query['query'], reference['answer'], trace['answer'],
                                                      reference['visible_evidence'], trace['visible_evidence'])
                        claim = trace.get('judgement') or {}
                        trace['judgement'] = {**claim, **preference,
                            'judge_trace': claim.get('judge_trace', []) + preference.get('judge_trace', [])}
                    except Exception as exc:
                        errors += 1
                        trace.update(status='error', judge_error={'type': type(exc).__name__, 'message': sanitize(str(exc))})
                if 'Full EVIQUE' in traces and 'Phase4' in traces:
                    traces['Full EVIQUE']['judgement'] = deepcopy(traces['Phase4'].get('judgement'))
                    traces['Full EVIQUE']['status'] = traces['Phase4']['status']
                    traces['Full EVIQUE']['evaluation_reused_from'] = 'Phase4: exact algorithm alias'
            for method, trace in traces.items():
                if not reuse:
                    trace['metrics'] = compute_metrics(trace['requirements'], trace.get('visible_evidence', []),
                        answer=trace.get('answer'), judgement=trace.get('judgement'),
                        answerable=query.get('metadata', {}).get('answerable'))
                    if trace['status'] == 'error':
                        for metric in METRIC_NAMES:
                            trace['metrics'][metric] = None
                            trace['metrics']['metric_details'][metric].update(status='execution_error', numerator=None, denominator=0, unknown_count=1)
                _write(output / 'per_query_trace' / _trace_name(query, method), trace)
                summaries.append({'query_id': query['query_id'], 'video_id': query['video_id'], 'method': method,
                                  'status': trace['status'], 'metrics': trace['metrics']})
            print(f'[{stage}] query {index + 1}/{len(selected_queries)}: {query["query_id"]}', flush=True)
    _save_tables(output, summaries)
    errors = sum(row['status'] == 'error' for row in summaries)
    trace_hashes = {path.name: file_hash(path) for path in (output / 'per_query_trace').glob('*.json')}
    summary = {'version': 'EVIQUE-v2', 'stage': stage, 'protocol_hash': protocol_hash,
               'status': 'failed' if errors else 'evidence_only_complete' if mode == 'evidence_only' else 'complete',
               'query_count': len(selected_queries), 'result_count': len(summaries), 'error_count': errors,
               'reused_pilot_queries': reused, 'runtime_seconds': time.monotonic() - started,
               'trace_hashes': trace_hashes,
               'completed_at': datetime.now(timezone.utc).isoformat(),
               'answer_metrics_evaluated': bool(client and judge),
               'scientific_limitations': ['Structural diagnostics are not answer accuracy.',
                  'Phase4 and Full EVIQUE are aliases, not independent algorithms.',
                  'Cached VideoRAG retrieval is historical; answers require common-prompt regeneration.']}
    _write(output / 'summary.json', summary)
    return summary
