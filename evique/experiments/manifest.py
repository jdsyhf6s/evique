"""Content-addressed experiment provenance without changing the frozen algorithm."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit

VERSION = 'EVIQUE-v2'
_SECRET = re.compile(r'(api[_-]?key|secret|password|passwd|credential|authorization|cookie|private[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|token$)', re.I)
_URL = re.compile(r'https?://[^\s\"<>]+', re.I)
_SELF = {'experiment_manifest.json', 'rebuild_manifest.json', 'run_manifest.json'}


def sanitize(value):
    """Redact credential fields recursively; discard URL auth, queries and fragments."""
    if isinstance(value, dict):
        return {str(key): '[REDACTED]' if _SECRET.search(str(key)) else sanitize(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        def clean(match):
            try:
                parts = urlsplit(match.group())
                host = parts.netloc.rsplit('@', 1)[-1]
                return urlunsplit((parts.scheme, host, parts.path, '', ''))
            except ValueError:
                return '[REDACTED_URL]'
        return _URL.sub(clean, value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError('Manifest values must be JSON serializable')


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root, *, core=False):
    files = {}
    for directory in ('evique', 'baselines') if core else ('evique', 'baselines', 'scripts', 'tests'):
        for path in sorted((root / directory).rglob('*.py')):
            relative = path.relative_to(root)
            if path.is_symlink() or '__pycache__' in relative.parts:
                continue
            if core and 'experiments' in relative.parts:
                continue
            files[relative.as_posix()] = file_hash(path)
    if not core:
        for pattern in ('requirements*.txt', 'pyproject.toml', 'environment.yml', 'pytest.ini'):
            for path in sorted(root.glob(pattern)):
                files[path.relative_to(root).as_posix()] = file_hash(path)
    return dict(sorted(files.items()))


def hash_index(index_dir):
    root = Path(index_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError('Index directory does not exist')
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError('Symlink artifacts cannot be included in an index snapshot')
        if not path.is_file():
            continue
        if any(part.startswith('.') or part.lower() in {'logs', 'log', '__pycache__'} for part in relative.parts):
            continue
        if path.name in _SELF or path.suffix.lower() in {'.log', '.tmp', '.pyc'}:
            continue
        files[relative.as_posix()] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    return {'sha256': _hash_json(files), 'files': files}


def _git(root):
    git_root = next((path for path in (root, *root.parents) if (path / '.git').exists()), root)
    command = ['git', '-c', 'safe.directory=' + str(git_root), '-C', str(root)]
    try:
        head = subprocess.run(command + ['rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        status = subprocess.run(command + ['status', '--porcelain', '--untracked-files=normal'], capture_output=True, text=True, timeout=10)
        return {'head': head.stdout.strip() if head.returncode == 0 else None,
                'dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
                'available': head.returncode == 0}
    except (OSError, subprocess.TimeoutExpired):
        return {'head': None, 'dirty': None, 'available': False}


def build_manifest(repo_root, index_dir, *, model: dict, embedding: dict, config: dict, query_file=None) -> dict:
    root = Path(repo_root).resolve()
    code = _source_files(root)
    dependencies = sorted({(dist.metadata.get('Name', 'unknown'), dist.version)
                           for dist in importlib.metadata.distributions()})
    clean_config = sanitize(config)
    return {'schema_version': 'evique_experiment_manifest_v1', 'version': VERSION,
            'created_at': datetime.now(timezone.utc).isoformat(), 'git': _git(root),
            'code': {'sha256': _hash_json(code), 'files': code}, 'index': hash_index(index_dir),
            'model': sanitize(model), 'embedding': sanitize(embedding),
            'config': clean_config, 'config_sha256': _hash_json(clean_config),
            'config_hash_scope': 'redacted_config_excludes_credentials',
            'query': {'sha256': file_hash(query_file), 'bytes': Path(query_file).stat().st_size} if query_file else None,
            'runtime': {'python': sys.version, 'platform': platform.platform(),
                        'dependencies': [dict(name=name, version=version) for name, version in dependencies],
                        'dependencies_sha256': _hash_json(dependencies)}}


def write_manifest(manifest, path):
    path = Path(path)
    payload = json.dumps(sanitize(manifest), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload)


def freeze_release(repo_root, output_path):
    root = Path(repo_root).resolve()
    files = _source_files(root, core=True)
    if not files:
        raise ValueError('No core sources found to freeze')
    lock = {'schema_version': 'evique_release_lock_v1', 'version': VERSION,
            'created_at': datetime.now(timezone.utc).isoformat(), 'git': _git(root),
            'source_files': files, 'source_sha256': _hash_json(files),
            'scope': 'Existing evique and baselines Python sources, excluding experiment orchestration; no retagging of internal legacy versions'}
    write_manifest(lock, output_path)
    return lock


def verify_release(lock, repo_root):
    if not isinstance(lock, dict):
        lock = json.loads(Path(lock).read_text(encoding='utf-8'))
    root = Path(repo_root).resolve()
    expected = lock.get('source_files', {})
    if lock.get('version') != VERSION or not expected or lock.get('source_sha256') != _hash_json(expected):
        raise ValueError('Invalid release lock')
    changed, missing = [], []
    added = sorted(set(_source_files(root, core=True)) - set(expected))
    for name, expected_hash in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Release lock source escapes repository')
        if not path.is_file():
            missing.append(name)
        elif file_hash(path) != expected_hash:
            changed.append(name)
    return {'valid': not changed and not missing and not added, 'version': VERSION,
            'changed': changed, 'missing': missing, 'added': added}
