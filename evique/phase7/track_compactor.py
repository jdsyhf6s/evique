from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .schema import TimeRange, TrackEpisode


@dataclass(frozen=True)
class TrackPoint:
    timestamp: float
    frame_id: str | None
    bbox: tuple[float, float, float, float]
    source_evidence_id: str | None = None


@dataclass(frozen=True)
class CompactTrack:
    track_id: str
    video_id: str
    label: str | None
    points: tuple[TrackPoint, ...]
    raw_sequence_reference: str
    anchor_time: float | None
    anchor_supported: bool
    motion_supported: bool
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value: dict[str, Any]) -> TrackPoint:
    timestamp = _finite(value.get("timestamp"), "track timestamp")
    if timestamp < 0:
        raise ValueError("track timestamp must be nonnegative")
    bbox = value.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("track bbox must contain four coordinates")
    box = tuple(_finite(coordinate, "track bbox coordinate") for coordinate in bbox)
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("track bbox coordinates must be ordered")
    return TrackPoint(
        timestamp=timestamp,
        frame_id=str(value["frame_id"]) if value.get("frame_id") is not None else None,
        bbox=box,
        source_evidence_id=str(value["evidence_id"]) if value.get("evidence_id") is not None else None,
    )


def _center(point: TrackPoint) -> tuple[float, float]:
    return (point.bbox[0] + point.bbox[2]) / 2, (point.bbox[1] + point.bbox[3]) / 2


def compact_track(track: dict | TrackEpisode, anchor_time: float | None,
                  required_windows: list[TimeRange]) -> CompactTrack:
    if isinstance(track, TrackEpisode):
        episode = track
        snapshots = episode.source_provenance.get("observations", [])
        if not isinstance(snapshots, list):
            raise TypeError("episode observations must be a list")
        eligible_ids = set(episode.ordered_observation_ids)
        by_id = {}
        for value in snapshots:
            observation_id = value.get("observation_id")
            if observation_id not in eligible_ids:
                continue
            if observation_id in by_id:
                raise ValueError("duplicate episode observation")
            by_id[observation_id] = value
        raw = []
        for observation_id in episode.ordered_observation_ids:
            if observation_id not in by_id:
                raise ValueError("missing episode observation")
            value = by_id[observation_id]
            timestamp = _finite(value.get("timestamp"), "track timestamp")
            # Category-only observations have no geometric point to compact.
            if value.get("bbox") is not None:
                if not episode.start_time <= timestamp <= episode.end_time:
                    raise ValueError("source point outside episode time range")
                raw.append(value)
        track = {
            "track_id": episode.episode_id, "video_id": episode.video_id,
            "raw_sequence_id": episode.raw_track_id, "observations": raw,
            "label": episode.category_evidence[0].observed_category if episode.category_evidence else None,
            "provenance": deepcopy(episode.source_provenance),
        }
    if not isinstance(track, dict):
        raise TypeError("track must be an object")
    track_id, video_id = track.get("track_id"), track.get("video_id")
    provenance = track.get("provenance")
    if not isinstance(track_id, str) or not track_id or not isinstance(video_id, str) or not video_id:
        raise ValueError("track identity is required")
    if not isinstance(provenance, dict) or provenance.get("video_id") != video_id:
        raise ValueError("track provenance must preserve video identity")
    raw = track.get("bbox_sequence") or track.get("points") or track.get("observations") or []
    if not isinstance(raw, list):
        raise TypeError("track sequence must be a list")
    unique: dict[tuple[Any, ...], TrackPoint] = {}
    for value in raw:
        point = _point(value)
        key = (point.timestamp, point.frame_id, point.bbox)
        unique.setdefault(key, point)
    points = sorted(unique.values(), key=lambda item: (item.timestamp, item.frame_id or ""))
    selected: dict[tuple[Any, ...], TrackPoint] = {}

    def keep(point: TrackPoint) -> None:
        selected[(point.timestamp, point.frame_id, point.bbox, point.source_evidence_id)] = point

    if points:
        keep(points[0])
        keep(points[-1])
    anchor_supported = False
    normalized_anchor = None
    if anchor_time is not None:
        normalized_anchor = _finite(anchor_time, "anchor_time")
        if points and points[0].timestamp <= normalized_anchor <= points[-1].timestamp:
            keep(min(points, key=lambda item: (abs(item.timestamp - normalized_anchor), item.timestamp)))
            anchor_supported = True
    for window in required_windows:
        within = [point for point in points if window.start <= point.timestamp <= window.end]
        if within:
            midpoint = (window.start + window.end) / 2
            keep(min(within, key=lambda item: (abs(item.timestamp - midpoint), item.timestamp)))
    compact_points = tuple(sorted(selected.values(), key=lambda item: (item.timestamp, item.frame_id or "")))
    centers = {_center(point) for point in compact_points}
    motion_supported = len(compact_points) >= 2 and len(centers) >= 2
    warnings = []
    if anchor_time is not None and not anchor_supported:
        warnings.append("anchor_not_observed")
    if not motion_supported:
        warnings.append("motion_not_supported")
    raw_reference = track.get("raw_sequence_id") or track.get("evidence_id") or f"{track_id}:raw"
    return CompactTrack(
        track_id=track_id, video_id=video_id, label=str(track["label"]) if track.get("label") else None,
        points=compact_points, raw_sequence_reference=str(raw_reference),
        anchor_time=normalized_anchor, anchor_supported=anchor_supported,
        motion_supported=motion_supported, provenance=dict(provenance), warnings=tuple(warnings),
    )
