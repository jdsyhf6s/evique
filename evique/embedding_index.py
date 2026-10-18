"""Persisted dense text retrieval over original evidence views.

FastEmbed is optional and imported only when an encoder is first used. Original
views are never modified; build destinations must be new directories.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .index_quality import FALLBACK_CAPTION_MARKER, IndexQualityChecker
from .utils import extract_caption_and_transcript


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
SOURCE_FILES = {view: view + "_view.jsonl" for view in (
    "scope", "target", "event", "adaptive_event", "visual_event", "visual_object",
)}
SCHEMA_VERSION = "evique_dense_index_v1"


class Encoder(Protocol):
    model_id: str

    def encode_documents(self, texts: list[str]) -> Sequence[Sequence[float]]: ...

    def encode_query(self, text: str) -> Sequence[float]: ...


class FastEmbedEncoder:
    """CPU dense embeddings with model-specific passage/query preprocessing."""

    def __init__(self, model_id: str = DEFAULT_MODEL, cache_dir: str | Path | None = None,
                 local_files_only: bool = False, threads: int | None = None):
        self.model_id = model_id
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.threads = threads
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError("Dense retrieval requires pip install -r requirements-retrieval.txt") from exc
            self._model = TextEmbedding(model_name=self.model_id, cache_dir=self.cache_dir,
                                        local_files_only=self.local_files_only, threads=self.threads)
        return self._model

    def encode_documents(self, texts: list[str]) -> Sequence[Sequence[float]]:
        return list(self._get_model().passage_embed(texts))

    def encode_query(self, text: str) -> Sequence[float]:
        return next(iter(self._get_model().query_embed(text)))


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(source_dir: Path) -> dict[str, str | None]:
    # Absent files are part of the snapshot: adding one makes the index stale.
    return {filename: _file_hash(source_dir / filename) for filename in SOURCE_FILES.values()}


def _read_rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a record object in {path.name}:{number}")
            yield record


def _vector(value: Any, dimension: int | None = None) -> np.ndarray:
    if isinstance(value, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError("Embedding vector cannot contain booleans")
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid embedding vector") from exc
    if array.ndim != 1 or not array.size or array.dtype.kind not in "iuf":
        raise ValueError("Embedding vector must be a nonempty numeric vector")
    array = array.astype(np.float64)
    norm = np.linalg.norm(array)
    if not np.isfinite(array).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding vector must be finite and nonzero")
    if dimension is not None and len(array) != dimension:
        raise ValueError(f"Embedding vector dimension {len(array)} differs from manifest dimension {dimension}")
    return array


def _timestamp(record: dict[str, Any]) -> float | None:
    for source in (record, record.get("provenance") or {}):
        for field in ("timestamp", "start_time"):
            value = source.get(field)
            if value is not None:
                try:
                    result = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(result):
                    return result
    return None


def _text_parts(view: str, record: dict[str, Any], counts: Counter):
    if view == "scope":
        payload = record
        if not record.get("caption") and not record.get("transcript") and record.get("text"):
            payload = {**record, "content": record.get("content") or record["text"]}
        caption, transcript = extract_caption_and_transcript(payload)
        if IndexQualityChecker.caption_status(payload) == "ok":
            yield "caption", caption
        elif caption:
            counts["skipped_caption"] += 1
        yield "transcript", transcript
    else:
        source_type = "detector" if view == "visual_object" or (
            view == "target" and record.get("source_type") == "detector") else "rule"
        text = record.get("description") or record.get("summary") or record.get("evidence_text") or record.get("text")
        if not text and view in {"event", "adaptive_event", "visual_event"}:
            text = str(record.get("event_type") or "").replace("_", " ")
        if not text and view in {"target", "visual_object"}:
            attributes = record.get("attributes") or []
            if isinstance(attributes, dict):
                attributes = [f"{key}: {value}" for key, value in attributes.items()]
            if isinstance(attributes, str):
                attributes = [attributes]
            text = " ".join(str(value) for value in (record.get("label"), record.get("color"), *attributes) if value)
        text = str(text or "").strip()
        if FALLBACK_CAPTION_MARKER in text.lower():
            counts["skipped_fallback_description"] += 1
            return
        yield source_type, text


def _collect_rows(source_dir: Path, counts: Counter) -> list[dict[str, Any]]:
    rows = []
    for view, filename in SOURCE_FILES.items():
        path = source_dir / filename
        if not path.exists():
            continue
        for record in _read_rows(path):
            counts["source_records"] += 1
            video_id = record.get("video_id")
            provenance = record.get("provenance") or {}
            if not isinstance(provenance, dict):
                raise ValueError(f"Invalid provenance in {filename}")
            if not isinstance(video_id, str) or not video_id.strip():
                counts["skipped_missing_video_id"] += 1
                continue
            if provenance.get("video_id") is not None and provenance["video_id"] != video_id:
                counts["skipped_conflicting_video_id"] += 1
                continue
            evidence_id = record.get("evidence_id") or record.get("id") or record.get("node_id")
            if not evidence_id:
                counts["skipped_missing_evidence_id"] += 1
                continue
            for source_type, text in _text_parts(view, record, counts):
                if not text:
                    continue
                key = json.dumps([video_id, view, evidence_id, source_type, len(rows)], ensure_ascii=False)
                rows.append({"embedding_id": hashlib.sha256(key.encode()).hexdigest(),
                             "evidence_id": str(evidence_id), "source_type": source_type,
                             "timestamp": _timestamp(record), "video_id": video_id,
                             "view": view, "record": record, "text": text})
    counts["embedding_rows"] = len(rows)
    return rows


class EmbeddingIndex:
    """Cosine retrieval; max-score deduplication happens before result truncation."""

    def __init__(self, rows: list[dict[str, Any]], manifest: dict[str, Any], encoder: Encoder):
        self.rows = rows
        self.manifest = manifest
        self.encoder = encoder
        self.dimension = manifest["dimension"]
        matrix = np.asarray([row["vector"] for row in rows], dtype=np.float64).reshape(-1, self.dimension)
        self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True) if len(rows) else matrix

    @classmethod
    def build(cls, index_dir: str | Path, output_dir: str | Path, encoder: Encoder) -> EmbeddingIndex:
        source, output = Path(index_dir).resolve(), Path(output_dir).resolve()
        if output.exists():
            raise FileExistsError(f"Embedding output already exists: {output}; choose a fresh directory")
        if not source.is_dir():
            raise FileNotFoundError(f"Evidence index directory does not exist: {source}")
        if not isinstance(encoder.model_id, str) or not encoder.model_id:
            raise ValueError("Encoder model_id must be a nonempty string")
        hashes = _source_hashes(source)
        counts = Counter()
        rows = _collect_rows(source, counts)
        dimension = len(_vector(encoder.encode_query("embedding dimension check")))
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            vectors = list(encoder.encode_documents([row["text"] for row in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Encoder returned a different vector count than document count")
            for row, vector in zip(batch, vectors):
                row["vector"] = _vector(vector, dimension).tolist()
        if _source_hashes(source) != hashes:
            raise ValueError("Source files changed during embedding build; rebuild from a stable source")
        # Exclusive directory creation also prevents races with another builder.
        output.mkdir(parents=True, exist_ok=False)
        with (output / "embeddings.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        manifest = {"schema_version": SCHEMA_VERSION, "model_id": encoder.model_id,
                    "encoder": type(encoder).__name__, "dimension": dimension,
                    "metric": "cosine", "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_dir": str(source), "source_hashes": hashes,
                    "embeddings_sha256": _file_hash(output / "embeddings.jsonl"),
                    "cache_dir": getattr(encoder, "cache_dir", None),
                    "counts": dict(counts), "row_count": len(rows)}
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return cls(rows, manifest, encoder)

    @classmethod
    def load(cls, output_dir: str | Path, encoder: Encoder | None = None,
             source_dir: str | Path | None = None) -> EmbeddingIndex:
        output = Path(output_dir)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported embedding index schema")
        dimension = manifest.get("dimension")
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Invalid embedding manifest dimension")
        if source_dir is not None and not Path(source_dir).is_dir():
            raise FileNotFoundError(f"Embedding source directory does not exist: {source_dir}")
        if source_dir is not None and _source_hashes(Path(source_dir)) != manifest.get("source_hashes"):
            raise ValueError("Embedding index is stale: source hashes differ")
        if _file_hash(output / "embeddings.jsonl") != manifest.get("embeddings_sha256"):
            raise ValueError("Embedding file hash differs from manifest")
        rows = list(_read_rows(output / "embeddings.jsonl"))
        if len(rows) != manifest.get("row_count"):
            raise ValueError("Embedding row count differs from manifest")
        for row in rows:
            _vector(row.get("vector"), dimension)
            for key in ("embedding_id", "evidence_id", "video_id", "source_type", "view", "text"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"Missing or invalid embedding field: {key}")
            if not isinstance(row.get("record"), dict) or "timestamp" not in row:
                raise ValueError("Missing embedding record or timestamp")
        if encoder is None:
            encoder = FastEmbedEncoder(model_id=manifest["model_id"], cache_dir=manifest.get("cache_dir"),
                                      local_files_only=True)
        if encoder.model_id != manifest.get("model_id"):
            raise ValueError("Encoder model_id differs from embedding manifest model_id")
        _vector(encoder.encode_query("embedding dimension check"), dimension)
        return cls(rows, manifest, encoder)

    def search(self, query: str, top_k: int = 20,
               video_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.rows:
            return []
        allowed = {video_ids} if isinstance(video_ids, str) else set(video_ids) if video_ids is not None else None
        positions = [i for i, row in enumerate(self.rows) if allowed is None or row["video_id"] in allowed]
        if not positions:
            return []
        query_vector = _vector(self.encoder.encode_query(query), self.dimension)
        scores = self._matrix[positions] @ (query_vector / np.linalg.norm(query_vector))
        best = {}
        for position, score in zip(positions, scores):
            row = self.rows[position]
            key = row["video_id"], row["view"], row["evidence_id"]
            if key not in best or score > best[key][0]:
                best[key] = float(score), position
        ranked = sorted(best.values(), key=lambda item: (-item[0], item[1]))[:top_k]
        return [{**{key: value for key, value in self.rows[position].items() if key != "vector"},
                 "score": score} for score, position in ranked]
