from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .fact_builder import normalize_visual_record
from .schema import EvidenceFact, ObservationSource, TimeRange


@dataclass(frozen=True)
class NearRule:
    version: str
    max_center_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("NearRule version is required")
        if isinstance(self.max_center_distance, bool) or not isinstance(self.max_center_distance, (int, float)):
            raise TypeError("near distance must be numeric")
        if not math.isfinite(self.max_center_distance) or self.max_center_distance < 0:
            raise ValueError("near distance must be finite and nonnegative")


def _center(box: tuple[float, ...]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _overlap(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _relation_fact(kind: str, subject: dict[str, Any], reference: dict[str, Any], rule: str) -> EvidenceFact:
    source_ids = (subject["evidence_id"], reference["evidence_id"])
    digest = hashlib.sha256(json.dumps([kind, subject["video_id"], *source_ids]).encode()).hexdigest()[:16]
    return EvidenceFact(
        fact_id=f"relation:{digest}", fact_type=kind, video_id=subject["video_id"],
        time_span=TimeRange(min(subject["timestamp"], reference["timestamp"]),
                            max(subject["timestamp"], reference["timestamp"])),
        frame_ids=(str(subject["frame_id"]),),
        role_bindings={
            "subject": str(subject.get("object_id") or subject["evidence_id"]),
            "reference": str(reference.get("object_id") or reference["evidence_id"]),
        },
        value={"coordinate_space": "image_plane", "camera_id": subject["camera_id"]},
        source_evidence_ids=source_ids,
        source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
        derivation_rule=rule,
        provenance={
            "video_id": subject["video_id"], "frame_id": subject["frame_id"],
            "camera_id": subject["camera_id"], "source_evidence_ids": list(source_ids),
        },
    )


def build_same_frame_relations(objects: list[dict], *, near_rule: NearRule) -> list[EvidenceFact]:
    normalized = [normalize_visual_record(item) for item in objects]
    facts: list[EvidenceFact] = []
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1:]:
            identity = ("video_id", "camera_id", "frame_id")
            if any(not left.get(field) or left.get(field) != right.get(field) for field in identity):
                continue
            if "bbox" not in left or "bbox" not in right:
                continue
            left_center, right_center = _center(left["bbox"]), _center(right["bbox"])
            directional = []
            if left_center[0] < right_center[0]:
                directional.extend((("left_of", left, right), ("right_of", right, left)))
            elif left_center[0] > right_center[0]:
                directional.extend((("left_of", right, left), ("right_of", left, right)))
            if left_center[1] < right_center[1]:
                directional.extend((("above", left, right), ("below", right, left)))
            elif left_center[1] > right_center[1]:
                directional.extend((("above", right, left), ("below", left, right)))
            for kind, subject, reference in directional:
                facts.append(_relation_fact(kind, subject, reference, f"same_frame_geometry_v1:{kind}"))
            if _overlap(left["bbox"], right["bbox"]):
                facts.append(_relation_fact("overlap", left, right, "same_frame_geometry_v1:overlap"))
            distance = math.dist(left_center, right_center)
            if distance <= near_rule.max_center_distance:
                facts.append(_relation_fact("image_near", left, right, f"{near_rule.version}:image_near"))
    return facts
