"""Report index availability and known input degradation without claiming accuracy."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .utils import extract_caption_and_transcript, write_json


QUALITY_REPORT_FILE = "index_quality_report.json"
FALLBACK_CAPTION_MARKER = "no native caption model output is available"


class IndexQualityChecker:
    """Check persisted evidence, including bases created before status fields existed.

    ``ok`` means usable data is present, not that visual facts were validated.
    Empty detections and optional ASR are reported but are not automatic failures.
    """

    @staticmethod
    def caption_status(record: dict[str, Any], visual_field: str | None = None) -> str:
        caption, _ = extract_caption_and_transcript(record, visual_field=visual_field)
        if FALLBACK_CAPTION_MARKER in caption.lower():
            return "fallback"
        explicit = str(record.get("caption_status") or "").lower()
        if explicit in {"fallback", "failed", "unavailable", "disabled"}:
            return explicit
        return "ok" if caption.strip() else "empty"

    def annotate_segments(self, video_segments: dict[str, dict[str, dict[str, Any]]]) -> None:
        for segments in video_segments.values():
            for record in segments.values():
                status = self.caption_status(record)
                record["caption_status"] = status
                record["quality_status"] = "ok" if status == "ok" else "degraded"

    @staticmethod
    def _asr_status(records: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
        if any(extract_caption_and_transcript(record)[1].strip() for record in records):
            return "ok"
        if metadata.get("audio_present") is False:
            return "no_audio"
        provider = str(metadata.get("asr_provider") or "").lower()
        if provider in {"off", "disabled", "none"}:
            return "disabled"
        if any(
            token in str(warning).lower()
            for warning in metadata.get("warnings") or []
            for token in ("asr failed", "could not extract audio", "could not load faster-whisper asr model", "unsupported asr_provider")
        ):
            return "failed"
        return "empty" if provider else "unavailable"

    @staticmethod
    def _stage_status(stats: dict[str, Any], key: str, *, attempted: bool) -> str:
        if not attempted:
            return "not_run"
        count = stats.get(key)
        if not isinstance(count, (int, float)) or count < 0:
            return "unavailable"
        return "ok" if count > 0 else "empty"

    def check(
        self,
        video_segments: dict[str, dict[str, dict[str, Any]]],
        *,
        base_manifest: dict[str, Any] | None = None,
        index_manifest: dict[str, Any] | None = None,
        visual_field: str | None = None,
    ) -> dict[str, Any]:
        records = [record for segments in video_segments.values() for record in segments.values()]
        counts = Counter(self.caption_status(record, visual_field) for record in records)
        failures = len(records) - counts.get("ok", 0)
        if not records:
            caption_status = "empty"
        elif not failures:
            caption_status = "ok"
        elif counts.get("ok"):
            caption_status = "partial_failure"
        elif counts.get("fallback") == len(records):
            caption_status = "fallback"
        else:
            caption_status = "unavailable" if counts.get("failed") or counts.get("unavailable") else "empty"

        base = base_manifest or {}
        asr_by_video = base.get("asr_by_video") or {}
        asr_statuses = {
            name: self._asr_status(list(segments.values()), asr_by_video.get(name, base))
            for name, segments in video_segments.items()
        }
        statuses = set(asr_statuses.values())
        asr_status = next(iter(statuses)) if len(statuses) == 1 else "mixed" if statuses else "unavailable"
        manifest = index_manifest or {}
        stats = manifest.get("graph_stats") or {}
        visual_attempted = manifest.get("visual_mode") in {"visual", "hybrid"}
        object_status = self._stage_status(stats, "visual_object", attempted=visual_attempted)
        track_status = self._stage_status(stats, "visual_track" if visual_attempted else "track", attempted=index_manifest is not None)
        event_status = self._stage_status(stats, "visual_event" if visual_attempted else "event", attempted=index_manifest is not None)
        missing_stage = "unavailable" in {object_status, track_status, event_status}
        quality = "unavailable" if not records else "degraded" if failures or missing_stage or "failed" in statuses else "ok"
        return {
            "schema_version": "evique_index_quality_v1",
            "caption_status": caption_status,
            "caption_failure_ratio": failures / len(records) if records else 0.0,
            "caption_status_counts": dict(counts),
            "segment_count": len(records),
            "asr_status": asr_status,
            "asr_status_by_video": asr_statuses,
            "object_detection_status": object_status,
            "track_status": track_status,
            "event_status": event_status,
            "overall_quality": quality,
            "quality_status": quality,
            "quality_scope": "availability_and_known_fallbacks_not_semantic_accuracy",
        }

    def save(self, output_dir: str | Path, video_segments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        report = self.check(video_segments, **kwargs)
        write_json(report, Path(output_dir) / QUALITY_REPORT_FILE)
        return report
