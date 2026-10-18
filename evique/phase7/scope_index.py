from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from evique.embedding_index import EmbeddingIndex, Encoder

from .schema import (
    CaptionStatus,
    ObservationSource,
    ScopeRecord,
    TimeRange,
    TranscriptStatus,
    from_dict,
)
from .scope_builder import scope_is_retrievable
from .trace import exclusive_write_json


ASSEMBLY_SCHEMA = "phase7_scope_index_v1"
SHARED_SCHEMA = "phase7_shared_index_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object in {path.name}:{number}")
                yield value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _legacy_scope(record: ScopeRecord) -> dict[str, Any]:
    caption = record.caption_text if record.caption_status is CaptionStatus.VALID else ""
    transcript = record.transcript_text if record.transcript_status is TranscriptStatus.VALID else ""
    return {
        "id": record.scope_id,
        "evidence_id": record.scope_id,
        "scope_id": record.scope_id,
        "video_id": record.video_id,
        "segment_id": record.segment_id,
        "view": "scope",
        "time": [record.time_range.start, record.time_range.end],
        "start_time": record.time_range.start,
        "end_time": record.time_range.end,
        "caption": caption,
        "caption_status": "ok" if caption else record.caption_status.value,
        "transcript": transcript,
        "transcript_status": record.transcript_status.value,
        "source_frame_ids": list(record.source_frame_ids),
        "source_frame_timestamps": list(record.source_frame_timestamps),
        "model_id": record.model_id,
        "prompt_version": record.prompt_version,
        "provenance": dict(record.provenance),
        "quality_warnings": list(record.quality_warnings),
    }


def assemble_phase7_index(scope_dir: Path, frozen_index_dir: Path, output_dir: Path) -> dict:
    scope_dir = Path(scope_dir).resolve()
    frozen = Path(frozen_index_dir).resolve()
    output = Path(output_dir).resolve()
    source_path = scope_dir / "semantic_scope.jsonl"
    if not source_path.is_file() or not frozen.is_dir():
        raise FileNotFoundError("semantic Scope corpus and frozen index are required")
    if output.exists():
        raise FileExistsError(f"Phase 7 index output already exists: {output}")

    source_hash = _sha256(source_path)
    frozen_hashes = {
        path.relative_to(frozen).as_posix(): _sha256(path)
        for path in sorted(frozen.rglob("*")) if path.is_file()
    }
    records = []
    for payload in _read_jsonl(source_path):
        record = from_dict(ScopeRecord, payload)
        if (
            scope_is_retrievable(record)
            and record.caption_source is not ObservationSource.QUERY_CONDITIONED_OBSERVATION
        ):
            records.append(_legacy_scope(record))

    output.mkdir(parents=True, exist_ok=False)
    copied_hashes: dict[str, str] = {}
    for relative, expected_hash in frozen_hashes.items():
        if relative == "scope_view.jsonl" or relative == "phase7_index_manifest.json":
            continue
        source = frozen / Path(relative)
        destination = output / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        actual = _sha256(destination)
        if actual != expected_hash:
            raise ValueError(f"copied frozen artifact hash mismatch: {relative}")
        copied_hashes[relative] = actual

    scope_view = output / "scope_view.jsonl"
    _write_jsonl(scope_view, records)
    if _sha256(source_path) != source_hash:
        raise ValueError("semantic Scope source changed during assembly")
    if any(_sha256(frozen / Path(name)) != digest for name, digest in frozen_hashes.items()):
        raise ValueError("frozen index changed during assembly")
    manifest = {
        "schema_version": ASSEMBLY_SCHEMA,
        "scope_source_sha256": source_hash,
        "scope_sha256": _sha256(scope_view),
        "scope_record_count": len(records),
        "copied_hashes": copied_hashes,
    }
    exclusive_write_json(output / "phase7_index_manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class RankedScope:
    evidence_id: str
    scope_id: str
    video_id: str
    source_kind: str
    text: str
    score: float
    rank: int
    time_range: TimeRange
    source_frame_ids: tuple[str, ...]
    provenance: dict[str, Any]


def _ranked(row: dict[str, Any], score: float, rank: int) -> RankedScope:
    record = row["record"]
    scope_id = str(record.get("scope_id") or row["evidence_id"])
    return RankedScope(
        evidence_id=f"{scope_id}:{row['source_type']}",
        scope_id=scope_id,
        video_id=row["video_id"],
        source_kind=row["source_type"],
        text=row["text"],
        score=float(score),
        rank=rank,
        time_range=TimeRange(float(record["start_time"]), float(record["end_time"])),
        source_frame_ids=tuple(record.get("source_frame_ids") or ()),
        provenance=dict(record.get("provenance") or {}),
    )


class SharedPhase7Index:
    def __init__(self, index_dir: Path, dense_index: EmbeddingIndex, manifest: dict[str, Any]):
        self.index_dir = Path(index_dir)
        self.dense_index = dense_index
        self.manifest = manifest
        self._scope_rows = [
            row for row in dense_index.rows
            if row["view"] == "scope" and row["source_type"] in {"caption", "transcript"}
        ]

    @classmethod
    def build(cls, index_dir: Path, output_dir: Path, encoder: Encoder) -> "SharedPhase7Index":
        index_dir = Path(index_dir).resolve()
        scope_path = index_dir / "scope_view.jsonl"
        assembly_path = index_dir / "phase7_index_manifest.json"
        if not scope_path.is_file() or not assembly_path.is_file():
            raise FileNotFoundError("assembled Phase 7 index is incomplete")
        assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
        scope_hash = _sha256(scope_path)
        if assembly.get("scope_sha256") != scope_hash:
            raise ValueError("Scope bytes differ from assembly manifest")
        dense = EmbeddingIndex.build(index_dir, output_dir, encoder)
        if _sha256(scope_path) != scope_hash:
            raise ValueError("Scope bytes changed during encoding")
        manifest = {
            "schema_version": SHARED_SCHEMA,
            "scope_sha256": scope_hash,
            "assembly_manifest_sha256": _sha256(assembly_path),
            "model_id": encoder.model_id,
            "embedding_manifest_sha256": _sha256(Path(output_dir) / "manifest.json"),
        }
        exclusive_write_json(Path(output_dir) / "phase7_shared_manifest.json", manifest)
        return cls(index_dir, dense, manifest)

    @classmethod
    def load(cls, output_dir: Path, encoder: Encoder, index_dir: Path) -> "SharedPhase7Index":
        output_dir, index_dir = Path(output_dir), Path(index_dir)
        manifest = json.loads((output_dir / "phase7_shared_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SHARED_SCHEMA:
            raise ValueError("unsupported shared index schema")
        if encoder.model_id != manifest.get("model_id"):
            raise ValueError("encoder model differs from shared index model")
        if _sha256(index_dir / "scope_view.jsonl") != manifest.get("scope_sha256"):
            raise ValueError("Scope bytes differ from shared index manifest")
        if _sha256(index_dir / "phase7_index_manifest.json") != manifest.get("assembly_manifest_sha256"):
            raise ValueError("assembly manifest differs from shared index manifest")
        if _sha256(output_dir / "manifest.json") != manifest.get("embedding_manifest_sha256"):
            raise ValueError("embedding manifest differs from shared index manifest")
        dense = EmbeddingIndex.load(output_dir, encoder=encoder, source_dir=index_dir)
        return cls(index_dir, dense, manifest)

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))

    def search_scope_keyword(self, query: str, video_id: str, limit: int) -> list[RankedScope]:
        if limit <= 0:
            return []
        query_tokens = set(self._tokens(query))
        scored = []
        for position, row in enumerate(self._scope_rows):
            if row["video_id"] != video_id:
                continue
            tokens = self._tokens(row["text"])
            score = sum(1 for token in tokens if token in query_tokens) / max(len(tokens), 1)
            if score > 0:
                scored.append((score, position, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_ranked(row, score, rank) for rank, (score, _, row) in enumerate(scored[:limit], 1)]

    def search_scope_dense(self, query: str, video_id: str, limit: int) -> list[RankedScope]:
        if limit <= 0:
            return []
        query_vector = np.asarray(self.dense_index.encoder.encode_query(query), dtype=np.float64)
        query_norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_vector).all() or not np.isfinite(query_norm) or query_norm <= 0:
            raise ValueError("query embedding must be finite and nonzero")
        scored = []
        for position, row in enumerate(self.dense_index.rows):
            if row["view"] != "scope" or row["video_id"] != video_id:
                continue
            vector = np.asarray(row["vector"], dtype=np.float64)
            score = float(vector @ query_vector / (np.linalg.norm(vector) * query_norm))
            scored.append((score, position, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_ranked(row, score, rank) for rank, (score, _, row) in enumerate(scored[:limit], 1)]


def search_scope_keyword(index: SharedPhase7Index, query: str, video_id: str, limit: int) -> list[RankedScope]:
    return index.search_scope_keyword(query, video_id, limit)


def search_scope_dense(index: SharedPhase7Index, query: str, video_id: str, limit: int) -> list[RankedScope]:
    return index.search_scope_dense(query, video_id, limit)
