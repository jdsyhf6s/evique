"""Experiment-only materialization helpers for Phase 7.5 semantic Scope.

The helpers are deliberately query blind.  They prepare repaired Scope inputs
without changing any Phase 1--7 retrieval or evidence-construction algorithm.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable

from evique.index_quality import FALLBACK_CAPTION_MARKER


FRAME_SELECTION_VERSION = "first-middle-last-v1"


@dataclass(frozen=True)
class CaptionInputs:
    segments: dict[str, Any]
    selected_keyframes: list[dict[str, Any]]
    manifest: dict[str, Any]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["timestamp"]), str(row["frame_id"])))
    if len(ordered) <= 3:
        return ordered
    positions = (0, len(ordered) // 2, len(ordered) - 1)
    return [ordered[position] for position in dict.fromkeys(positions)]


def _matches_segment(row: dict[str, Any], video_id: str, segment_id: str) -> bool:
    if str(row.get("video_id") or "") != video_id:
        return False
    value = str(row.get("segment_id") or "")
    return value in {segment_id, f"{video_id}_{segment_id}"}


def _valid_caption(text: str) -> bool:
    normalized = text.strip()
    return bool(normalized) and FALLBACK_CAPTION_MARKER not in normalized.casefold()


def build_caption_inputs(
    segments: dict[str, Any],
    keyframes: list[dict[str, Any]],
    *,
    caption_frame: Callable[[dict[str, Any]], str],
    local_image_resolver: Callable[[dict[str, Any]], str],
    source_video_sha256: str,
    model_id: str,
    model_sha256: str,
    prompt_version: str,
) -> CaptionInputs:
    """Create immutable inputs for the existing repaired Scope writer.

    Frame selection is fixed before caption inference and never consumes a
    query, requirement, answer, retrieval result, or human annotation.
    """
    materialized = deepcopy(segments)
    selected_keyframes: list[dict[str, Any]] = []
    valid_count = 0
    failed_count = 0
    for video_name, video_segments in materialized.items():
        video_id = str(video_name)
        for segment_key, segment in sorted(video_segments.items()):
            segment_id = str(segment.get("segment_id") or segment_key)
            candidates = [
                row for row in keyframes if isinstance(row, dict)
                and _matches_segment(row, video_id, segment_id)
                and row.get("timestamp") is not None
                and row.get("frame_id")
            ]
            chosen = _selected(candidates)
            captions: list[tuple[str, float, str]] = []
            errors: list[str] = []
            labels = ("Start", "Middle", "End") if len(chosen) == 3 else tuple(
                f"Frame {index + 1}" for index in range(len(chosen))
            )
            for label, source in zip(labels, chosen):
                frame = deepcopy(source)
                frame["local_image_path"] = str(local_image_resolver(source))
                frame["segment_id"] = segment_id
                frame["provenance"] = {
                    **(source.get("provenance") or {}),
                    "video_id": video_id,
                    "source_frame_record_id": source.get("id"),
                    "source_segment_id": source.get("segment_id"),
                    "frame_selection_version": FRAME_SELECTION_VERSION,
                }
                selected_keyframes.append(frame)
                try:
                    text = str(caption_frame(frame)).strip()
                except Exception as exc:  # noqa: BLE001 - failures are explicit data
                    errors.append(f"{frame['frame_id']}: {type(exc).__name__}: {exc}")
                    continue
                if _valid_caption(text):
                    captions.append((label, float(frame["timestamp"]), text))
                else:
                    errors.append(f"{frame['frame_id']}: invalid_or_placeholder_caption")
            if captions:
                segment["caption"] = " ".join(
                    f"{label} frame ({timestamp:.2f}s): {text}" for label, timestamp, text in captions
                )
                segment["caption_status"] = "valid"
                segment.pop("caption_error", None)
                valid_count += 1
            else:
                segment["caption"] = ""
                segment["caption_status"] = "failed"
                segment["caption_error"] = "; ".join(errors) or "no_bound_frames"
                failed_count += 1
            segment["provenance"] = {
                **(segment.get("provenance") or {}),
                "video_id": video_id,
                "segment_id": segment_id,
                "caption_source": "offline_caption",
                "caption_model": model_id,
                "caption_prompt_version": prompt_version,
                "source_video_sha256": source_video_sha256,
                "frame_selection_version": FRAME_SELECTION_VERSION,
            }
            if errors:
                segment["caption_warnings"] = errors

    manifest = {
        "schema_version": "phase7_5_caption_inputs_v1",
        "query_independent": True,
        "query_hash": None,
        "source_video_sha256": source_video_sha256,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "prompt_version": prompt_version,
        "frame_selection_version": FRAME_SELECTION_VERSION,
        "selected_frame_count": len(selected_keyframes),
        "valid_segment_count": valid_count,
        "failed_segment_count": failed_count,
        "external_request_count": 0,
        "segments_sha256": _canonical_hash(materialized),
        "selected_keyframes_sha256": _canonical_hash(selected_keyframes),
    }
    return CaptionInputs(materialized, selected_keyframes, manifest)
