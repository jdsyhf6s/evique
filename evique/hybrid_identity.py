"""Exact video boundaries for candidate retrieval, separate from legacy aliases."""
from __future__ import annotations

from typing import Any


class ExactVideoFilter(set):
    """Canonical IDs, never filename/stem aliases."""


def strict_video_id(row: dict[str, Any]) -> str | None:
    values = set()
    for source in (row, row.get('provenance') or {}, row.get('record') or {}):
        if not isinstance(source, dict):
            continue
        value = source.get('video_id')
        if value is not None and str(value).strip():
            values.add(str(value).strip())
        nested = source.get('provenance') or {}
        if isinstance(nested, dict) and nested.get('video_id'):
            values.add(str(nested['video_id']).strip())
    return next(iter(values)) if len(values) == 1 else None


def resolve_video_filter(rows: list[dict[str, Any]], metadata: dict[str, Any] | None) -> ExactVideoFilter | None:
    metadata = metadata or {}
    available = {video for row in rows if (video := strict_video_id(row))}
    explicit = metadata.get('video_id')
    if explicit is not None and str(explicit).strip():
        video = str(explicit).strip()
        if video not in available:
            raise ValueError(f'Unknown canonical video_id: {video}')
        return ExactVideoFilter([video])
    keys = ('source_vid', 'video_name', 'video_path', 'original_video_path', 'uploaded_filename')
    requested = {key: str(metadata[key]).replace('\\', '/') for key in keys if metadata.get(key)}
    if not requested:
        return None
    matches = set()
    for row in rows:
        video = strict_video_id(row)
        if not video:
            continue
        if all(str(row.get('video_path' if key == 'original_video_path' else key) or '').replace('\\', '/') == value
               for key, value in requested.items()):
            matches.add(video)
    if len(matches) != 1:
        raise ValueError('Video metadata is absent or ambiguous; provide an exact video_id')
    return ExactVideoFilter(matches)
