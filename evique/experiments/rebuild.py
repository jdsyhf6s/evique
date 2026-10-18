"""Fresh-output orchestration of the existing video, view, embedding and graph builders.

Graph construction is coupled to views in the frozen builder. The final stage
checks and publishes those exact graph bytes after embedding persistence.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from .manifest import VERSION, build_manifest, file_hash, hash_index, sanitize, write_manifest


@contextmanager
def _environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _save_state(state, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(sanitize(state), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _quality_gate(report, state, allow_degraded):
    if report.get('overall_quality') != 'ok':
        state['degraded'] = True
        if not allow_degraded:
            raise ValueError('degraded evidence rejected; explicitly set allow_degraded for a non-scientific diagnostic build')


def _publish_graph(index, destination):
    names = ('evidence_nodes.jsonl', 'evidence_relations.jsonl', 'graph_stats.json')
    nodes = [json.loads(line) for line in (index / names[0]).read_text(encoding='utf-8').splitlines() if line.strip()]
    edges = [json.loads(line) for line in (index / names[1]).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = {row['id'] for row in nodes}
    if not ids or len(ids) != len(nodes):
        raise ValueError('Graph must contain unique nodes')
    if any(edge['source'] not in ids or edge['target'] not in ids for edge in edges):
        raise ValueError('Graph has dangling relation endpoints')
    destination.mkdir(exist_ok=False)
    for name in names:
        shutil.copyfile(index / name, destination / name)
        if file_hash(index / name) != file_hash(destination / name):
            raise ValueError('Published graph hash mismatch')
    return {'nodes': len(nodes), 'relations': len(edges), 'artifacts': hash_index(destination)}


def run_rebuild(config, output_dir, *, repo_root=None, encoder=None):
    """Rebuild in a new directory; failures retain stage status and then raise.

    ``mode='raw_video'`` needs ``video_paths``. Imported ``segments_file`` is
    accepted only with ``mode='segments_replay'`` and can never be scientific-ready.
    Injected encoders are for explicit replay/testing, never semantic validation.
    Environment overrides are process-global, so builds must run serially.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = output / 'index'
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    mode = config.get('mode', 'raw_video')
    state = {'schema_version': 'evique_rebuild_v1', 'version': VERSION,
             'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
             'input_kind': mode, 'scientific_ready': False, 'degraded': False,
             'config': sanitize(config), 'stages': [], 'encoder_injected': encoder is not None}
    state_path = output / 'rebuild_manifest.json'
    write_manifest(state, state_path)
    stage = None
    clock = time.perf_counter()

    def begin(name, note):
        nonlocal stage, clock
        stage = {'name': name, 'status': 'running', 'note': note,
                 'started_at': datetime.now(timezone.utc).isoformat()}
        state['stages'].append(stage)
        clock = time.perf_counter()
        _save_state(state, state_path)

    def finish(**details):
        stage.update(status='completed', elapsed_seconds=time.perf_counter() - clock, **details)
        _save_state(state, state_path)

    try:
        begin('base', 'Raw video processing' if mode == 'raw_video' else 'Imported segments replay; video decoding, ASR and caption inference not run')
        if mode not in {'raw_video', 'segments_replay'}:
            raise ValueError('mode must be raw_video or segments_replay')
        if mode == 'raw_video' and 'segments_file' in config:
            raise ValueError('Imported segments require explicit mode=segments_replay')
        allow_degraded = config.get('allow_degraded', False)
        if not isinstance(allow_degraded, bool):
            raise ValueError('allow_degraded must be a boolean')
        environment = config.get('environment', {})
        if not isinstance(environment, dict) or any(not str(key).startswith(('EVIQUE_', 'ASR_', 'CAPTION_', 'WHISPER_', 'OPENAI_')) for key in environment):
            raise ValueError('environment overrides must be EVIQUE, ASR, CAPTION, WHISPER or OPENAI settings')
        with _environment(environment):
            state['effective_environment'] = sanitize({key: value for key, value in os.environ.items()
                if key.startswith(('EVIQUE_', 'ASR_', 'CAPTION_', 'WHISPER_', 'OPENAI_'))})
            from evique.index_quality import IndexQualityChecker
            if mode == 'raw_video':
                from evique.standalone_base_builder import build_evique_standalone_base
                paths = [Path(path).expanduser().resolve() for path in config.get('video_paths', [])]
                if not paths or any(not path.is_file() for path in paths):
                    raise ValueError('raw_video requires existing video_paths')
                state['inputs'] = [{'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size} for path in paths]
                base = build_evique_standalone_base(video_paths=paths, output_base_dir=index / 'base',
                    **config.get('base', {}))
                segments = base['video_segments']
                video_map = base['video_path_map']
                base_manifest = base['manifest']
                # The legacy base builder captures selected environment values.
                # Sanitize its metadata in this fresh run before publishing it.
                base_path = index / 'base' / 'evique_base_manifest.json'
                base_path.write_text(json.dumps(sanitize(_read(base_path)), indent=2), encoding='utf-8')
            else:
                path = Path(config['segments_file']).resolve()
                state['inputs'] = [{'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}]
                segments = _read(path)
                video_map, base_manifest = {}, {}
                write_manifest(segments, index / 'base' / 'replayed_segments.json')
            quality = IndexQualityChecker().check(segments, base_manifest=base_manifest)
            _quality_gate(quality, state, allow_degraded)
            finish(quality=quality, artifacts=hash_index(index / 'base'))

            begin('views', 'Frozen builder: view and graph construction are coupled; graph is published after embeddings')
            from evique.builder import build_evique_from_segments
            built = build_evique_from_segments(segments, video_path_map=video_map, output_dir=index,
                                                **config.get('views', {}))
            quality = IndexQualityChecker().check(segments, base_manifest=base_manifest, index_manifest=built,
                                                   visual_field=config.get('views', {}).get('visual_field'))
            _quality_gate(quality, state, allow_degraded)
            finish(quality=quality, builder_version=built.get('evique_version'), artifacts=hash_index(index))

            begin('embeddings', 'Existing persisted EmbeddingIndex; injected vectors only demonstrate algebra and persistence')
            from evique.embedding_index import DEFAULT_MODEL, EmbeddingIndex, FastEmbedEncoder
            embedding_config = dict(config.get('embedding', {}))
            if encoder is None:
                encoder = FastEmbedEncoder(model_id=embedding_config.get('model_id', DEFAULT_MODEL),
                    cache_dir=embedding_config.get('cache_dir'), local_files_only=embedding_config.get('local_files_only', True),
                    threads=embedding_config.get('threads'))
            elif embedding_config.get('model_id', encoder.model_id) != encoder.model_id:
                raise ValueError('Configured embedding model_id differs from injected encoder')
            dense = EmbeddingIndex.build(index, index / 'embeddings', encoder)
            EmbeddingIndex.load(index / 'embeddings', encoder=encoder, source_dir=index)
            finish(model=dense.manifest, artifacts=hash_index(index / 'embeddings'))

            begin('graph', 'Validate and publish the graph already constructed by the frozen view builder; no reconstruction')
            graph = _publish_graph(index, output / 'graph')
            finish(**graph)

        manifest = build_manifest(root, index, model=config.get('model', {}), embedding=dense.manifest,
                                  config={**config, 'effective_environment': state['effective_environment']})
        write_manifest(manifest, output / 'experiment_manifest.json')
        state['status'] = 'completed'
        state['scientific_ready'] = mode == 'raw_video' and not state['degraded'] and not state['encoder_injected'] and not allow_degraded
        state['readiness_scope'] = 'Pipeline completeness and availability only; no semantic accuracy or benchmark claim'
        _save_state(state, state_path)
        return state
    except BaseException as exc:
        state['status'] = 'failed'
        state['scientific_ready'] = False
        if stage is not None:
            stage.update(status='failed', elapsed_seconds=time.perf_counter() - clock, error_type=type(exc).__name__)
        _save_state(state, state_path)
        raise
