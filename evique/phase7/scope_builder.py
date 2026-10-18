from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from evique.index_quality import FALLBACK_CAPTION_MARKER
from evique.utils import parse_time_range

from .schema import (
    CaptionStatus, ObservationSource, ScopeRecord, TimeRange, TranscriptStatus, to_dict,
)
from .trace import canonical_hash, exclusive_write_json


def _identity(record: dict[str, Any], *, expected: str | None = None) -> str:
    provenance = record.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    values = {str(value) for value in (record.get("video_id"), provenance.get("video_id")) if value}
    if expected:
        values.add(str(expected))
    if len(values) != 1:
        raise ValueError("conflicting or missing video_id")
    return next(iter(values))


def _caption_metadata(manifest: dict[str, Any], video_name: str) -> dict[str, Any]:
    nested = manifest.get("caption_by_video") or manifest.get("videos") or {}
    if isinstance(nested, dict) and isinstance(nested.get(video_name), dict):
        return {**manifest, **nested[video_name]}
    return manifest


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _caption_status(segment: dict[str, Any], caption: str, *, has_frames: bool,
                    model_id: str | None, prompt_version: str | None) -> tuple[CaptionStatus, list[str]]:
    warnings: list[str] = []
    declared = _status(segment.get("caption_status"))
    if FALLBACK_CAPTION_MARKER in caption.lower():
        return CaptionStatus.PLACEHOLDER, ["placeholder_caption"]
    if segment.get("caption_error") or declared in {"failed", "error"}:
        return CaptionStatus.FAILED, ["caption_generation_failed"]
    if not caption.strip():
        return CaptionStatus.EMPTY, []
    missing = []
    if not has_frames:
        missing.append("source_frames")
    if not model_id:
        missing.append("model_id")
    if not prompt_version:
        missing.append("prompt_version")
    if missing:
        warnings.extend(f"missing_{name}" for name in missing)
        return CaptionStatus.FAILED, warnings
    return CaptionStatus.VALID, warnings


def _transcript_status(segment: dict[str, Any], transcript: str) -> TranscriptStatus:
    declared = _status(segment.get("transcript_status"))
    if declared in {"failed", "error"}:
        return TranscriptStatus.FAILED
    return TranscriptStatus.VALID if transcript.strip() else TranscriptStatus.EMPTY


def build_scope_records(segments: dict, caption_manifest: dict,
                        keyframes: list[dict]) -> list[ScopeRecord]:
    if not isinstance(segments, dict) or not isinstance(caption_manifest, dict):
        raise TypeError("segments and caption_manifest must be objects")
    frames_by_segment: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for frame in keyframes:
        if not isinstance(frame, dict) or not frame.get("segment_id"):
            continue
        frame_video_id = _identity(frame)
        frames_by_segment[(frame_video_id, str(frame["segment_id"]))].append(frame)

    records: list[ScopeRecord] = []
    for video_name, video_segments in segments.items():
        if not isinstance(video_segments, dict):
            raise TypeError("video segments must be objects")
        metadata = _caption_metadata(caption_manifest, str(video_name))
        model_id = metadata.get("model_id") or metadata.get("caption_model_path")
        prompt_version = metadata.get("prompt_version")
        for index, segment in sorted(video_segments.items(), key=lambda item: str(item[0])):
            if not isinstance(segment, dict):
                raise TypeError("segment records must be objects")
            segment_id = str(segment.get("segment_id") or f"{video_name}_{index}")
            video_id = _identity(segment)
            start, end = parse_time_range(segment.get("time") or segment.get("time_range"))
            if start is None or end is None:
                start, end = segment.get("start_time"), segment.get("end_time")
            time_range = TimeRange(start, end)
            bound_frames = sorted(
                frames_by_segment.get((video_id, segment_id), []),
                key=lambda row: (float(row.get("timestamp", 0)), str(row.get("frame_id") or row.get("id") or "")),
            )
            frame_ids: list[str] = []
            frame_times: list[float] = []
            for frame in bound_frames:
                _identity(frame, expected=video_id)
                frame_id = frame.get("frame_id") or frame.get("id")
                if not frame_id or frame.get("timestamp") is None:
                    continue
                frame_ids.append(str(frame_id))
                frame_times.append(float(frame["timestamp"]))
            caption = str(segment.get("caption") or "").strip()
            transcript = str(segment.get("transcript") or "").strip()
            caption_status, warnings = _caption_status(
                segment, caption, has_frames=bool(frame_ids),
                model_id=str(model_id) if model_id else None,
                prompt_version=str(prompt_version) if prompt_version else None,
            )
            provenance = dict(segment.get("provenance") or {})
            provenance.update({
                "video_id": video_id,
                "video_name": str(video_name),
                "segment_id": segment_id,
                "source": "phase7_semantic_scope_adapter",
            })
            records.append(ScopeRecord(
                scope_id=str(segment.get("scope_id") or f"scope:{segment_id}"),
                video_id=video_id,
                segment_id=segment_id,
                time_range=time_range,
                caption_text=caption,
                caption_status=caption_status,
                caption_source=ObservationSource.OFFLINE_CAPTION,
                transcript_text=transcript,
                transcript_status=_transcript_status(segment, transcript),
                transcript_source=ObservationSource.TRANSCRIPT,
                source_frame_ids=tuple(frame_ids),
                source_frame_timestamps=tuple(frame_times),
                model_id=str(model_id) if model_id else None,
                prompt_version=str(prompt_version) if prompt_version else None,
                provenance=provenance,
                quality_warnings=tuple(warnings),
            ))
    return records


def scope_is_retrievable(record: ScopeRecord) -> bool:
    return (
        record.query_hash is None
        and (
            record.caption_status is CaptionStatus.VALID
            or record.transcript_status is TranscriptStatus.VALID
        )
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_scope_corpus(records: list[ScopeRecord], output_dir: Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        payload = to_dict(record)
        counts[f"caption_{record.caption_status.value}"] += 1
        counts[f"transcript_{record.transcript_status.value}"] += 1
        if record.query_hash is not None:
            rejected.append({"scope_id": record.scope_id, "reason": "query_conditioned", "record": payload})
        elif not scope_is_retrievable(record):
            rejected.append({"scope_id": record.scope_id, "reason": "not_retrievable", "record": payload})
        else:
            accepted.append(payload)
    scope_path = output / "semantic_scope.jsonl"
    rejection_path = output / "scope_rejections.jsonl"
    _write_jsonl(scope_path, accepted)
    _write_jsonl(rejection_path, rejected)
    manifest = {
        "schema_version": "phase7_semantic_scope_v1",
        "record_count": len(records),
        "retrievable_count": len(accepted),
        "rejected_count": len(rejected),
        "status_counts": dict(sorted(counts.items())),
        "source_records_sha256": canonical_hash([to_dict(record) for record in records]),
        "semantic_scope_sha256": __import__("hashlib").sha256(scope_path.read_bytes()).hexdigest(),
        "rejections_sha256": __import__("hashlib").sha256(rejection_path.read_bytes()).hexdigest(),
        "model_ids": sorted({record.model_id for record in records if record.model_id}),
        "prompt_versions": sorted({record.prompt_version for record in records if record.prompt_version}),
    }
    exclusive_write_json(output / "scope_manifest.json", manifest)
    return manifest
