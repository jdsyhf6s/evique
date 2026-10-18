from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .fact_builder import normalize_visual_record
from .schema import EvidenceFact, ObservationSource, TimeRange


@dataclass(frozen=True)
class SceneRegionDefinition:
    region_id: str
    video_id: str
    camera_id: str
    polygon: tuple[tuple[float, float], ...]
    semantic_label: str
    definition_method: str
    definition_version: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("region_id", "video_id", "camera_id", "semantic_label", "definition_method", "definition_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if len(self.polygon) < 3:
            raise ValueError("region polygon requires at least three points")
        if any(len(point) != 2 or any(not math.isfinite(value) for value in point) for point in self.polygon):
            raise ValueError("region polygon must contain finite points")
        if not self.provenance or self.provenance.get("video_id") != self.video_id:
            raise ValueError("region provenance must preserve video identity")
        if self.definition_method == "image_thirds" and any(
            term in self.semantic_label.casefold() for term in ("road", "lane", "entrance", "exit", "yield")
        ):
            raise ValueError("image thirds cannot define road-semantic labels")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_region_definitions(path: Path, expected_hash: str) -> list[SceneRegionDefinition]:
    path = Path(path)
    if _sha256(path) != expected_hash:
        raise ValueError("region definition hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("region definition file must contain a list")
    regions = []
    for value in payload:
        if not isinstance(value, dict):
            raise TypeError("region definition must be an object")
        allowed = {
            "region_id", "video_id", "camera_id", "polygon", "semantic_label",
            "definition_method", "definition_version", "provenance",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown region fields: {sorted(unknown)}")
        polygon = tuple(tuple(float(coordinate) for coordinate in point) for point in value["polygon"])
        regions.append(SceneRegionDefinition(polygon=polygon, **{key: item for key, item in value.items() if key != "polygon"}))
    return regions


def _inside(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def assign_regions(objects: list[dict], regions: list[SceneRegionDefinition]) -> list[EvidenceFact]:
    normalized = [normalize_visual_record(item) for item in objects]
    facts = []
    for item in normalized:
        if "bbox" not in item or not item.get("camera_id"):
            continue
        center = ((item["bbox"][0] + item["bbox"][2]) / 2, (item["bbox"][1] + item["bbox"][3]) / 2)
        for region in regions:
            if region.video_id != item["video_id"] or region.camera_id != item["camera_id"]:
                continue
            if not _inside(center, region.polygon):
                continue
            digest = hashlib.sha256(f"{item['evidence_id']}:{region.region_id}".encode()).hexdigest()[:16]
            facts.append(EvidenceFact(
                fact_id=f"region:{digest}", fact_type="located_in_region", video_id=item["video_id"],
                time_span=TimeRange(item["timestamp"], item["timestamp"]),
                frame_ids=(str(item["frame_id"]),) if item.get("frame_id") else (),
                role_bindings={"subject": str(item.get("object_id") or item["evidence_id"])},
                value={"region_id": region.region_id, "semantic_label": region.semantic_label},
                source_evidence_ids=(item["evidence_id"],),
                source_kind=ObservationSource.RULE_DERIVED_OBSERVATION,
                derivation_rule=f"region_assignment_v1:{region.definition_version}",
                provenance={
                    "video_id": item["video_id"], "region_id": region.region_id,
                    "region_definition_provenance": region.provenance,
                },
            ))
    return facts
