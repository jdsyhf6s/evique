from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any

from evique.experiments.manifest import file_hash, hash_index

from .config import Phase7Config
from .trace import canonical_hash, exclusive_write_json


@dataclass(frozen=True)
class Phase7GateReport:
    valid: bool
    checked_hashes: dict[str, str]
    failures: list[dict[str, Any]] = field(default_factory=list)


def _resolve(root: Path, value: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _load_release_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8-sig"))
    sources = lock.get("source_files")
    if (lock.get("schema_version") != "evique_release_lock_v1"
            or lock.get("version") != "EVIQUE-v2" or not isinstance(sources, dict)
            or lock.get("source_sha256") != canonical_hash(sources)):
        raise ValueError("Invalid reviewed release lock")
    return lock


def _snapshot(config: Phase7Config, root: Path) -> dict[str, Any]:
    root = root.resolve()
    release_path = _resolve(root, config.release_lock)
    release = _load_release_lock(release_path)
    protected = {}
    for name, expected in sorted(release["source_files"].items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Protected source missing or outside repository: {name}")
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Protected source changed: {name}")
        protected[name] = actual

    files: dict[str, str] = {}
    values = [config.query_file, config.classification_file, config.spec_file,
              config.plan_file, *config.immutable_files]
    for value in values:
        path = _resolve(root, value)
        if not path.is_file():
            raise FileNotFoundError(path)
        files[_relative(root, path)] = file_hash(path)
    directories = {}
    for value in config.immutable_dirs:
        path = _resolve(root, value)
        directories[_relative(root, path)] = hash_index(path)
    return {
        "schema_version": "phase7_input_lock_v1",
        "protocol_version": config.protocol_version,
        "phase7_start_commit": _git_head(root),
        "release_lock": {"path": _relative(root, release_path), "sha256": file_hash(release_path),
                         "source_sha256": release["source_sha256"]},
        "protected_source_files": protected,
        "immutable_files": dict(sorted(files.items())),
        "immutable_dirs": dict(sorted(directories.items())),
        "config": config.to_dict(),
    }


def freeze_phase7_inputs(config: Phase7Config, repo_root: Path, output: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    ledger = _snapshot(config, root)
    ledger["ledger_sha256"] = canonical_hash(ledger)
    exclusive_write_json(Path(output), ledger)
    return ledger


def verify_phase7_inputs(config: Phase7Config, repo_root: Path,
                         ledger_path: Path) -> Phase7GateReport:
    root = Path(repo_root).resolve()
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8-sig"))
    failures: list[dict[str, Any]] = []
    checked: dict[str, str] = {}
    stored_hash = ledger.get("ledger_sha256")
    body = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if stored_hash != canonical_hash(body):
        failures.append({"kind": "invalid_ledger_hash", "path": str(ledger_path)})
    if ledger.get("protocol_version") != config.protocol_version:
        failures.append({"kind": "protocol_mismatch", "path": str(ledger_path)})
    if ledger.get("config") != config.to_dict():
        failures.append({"kind": "config_mismatch", "path": str(ledger_path)})

    release_path = _resolve(root, config.release_lock)
    try:
        release = _load_release_lock(release_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append({"kind": "invalid_release_lock", "path": _relative(root, release_path),
                         "error": type(exc).__name__})
        release = {"source_files": {}}
    for name, expected in release.get("source_files", {}).items():
        path = (root / name).resolve()
        if not path.is_file():
            failures.append({"kind": "missing", "path": name})
            continue
        actual = file_hash(path)
        checked[name] = actual
        if actual != expected:
            failures.append({"kind": "hash_mismatch", "path": name,
                             "expected": expected, "actual": actual})

    for name, expected in ledger.get("immutable_files", {}).items():
        path = (root / name).resolve() if not Path(name).is_absolute() else Path(name)
        if not path.is_file():
            failures.append({"kind": "missing", "path": name})
            continue
        actual = file_hash(path)
        checked[name] = actual
        if actual != expected:
            failures.append({"kind": "hash_mismatch", "path": name,
                             "expected": expected, "actual": actual})
    for name, expected in ledger.get("immutable_dirs", {}).items():
        path = (root / name).resolve() if not Path(name).is_absolute() else Path(name)
        if not path.is_dir():
            failures.append({"kind": "missing", "path": name})
            continue
        actual = hash_index(path)
        checked[name] = actual["sha256"]
        if actual != expected:
            failures.append({"kind": "hash_mismatch", "path": name,
                             "expected": expected.get("sha256"), "actual": actual["sha256"]})
    return Phase7GateReport(valid=not failures, checked_hashes=checked, failures=failures)
