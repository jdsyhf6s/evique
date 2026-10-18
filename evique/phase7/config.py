from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evique.experiments.manifest import sanitize


PROTOCOL_VERSION = "phase7-semantic-evidence-v1"
PHASE_D_PROTOCOL_VERSION = "phase-d-temporal-relation-v1"


def _paths(values: Any) -> tuple[Path, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TypeError("path collections must be lists")
    return tuple(Path(str(value)) for value in values)


@dataclass(frozen=True)
class TemporalRelationBridgeConfig:
    enabled: bool = False
    protocol_version: str = PHASE_D_PROTOCOL_VERSION
    max_refined_windows: int = 2
    preserve_base_retrieval_top_k: bool = True
    trace_enabled: bool = True

    @classmethod
    def from_dict(cls, value: Any) -> "TemporalRelationBridgeConfig":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("temporal_relation_bridge must be an object")
        flags = {}
        for name, default in (
            ("enabled", False),
            ("preserve_base_retrieval_top_k", True),
            ("trace_enabled", True),
        ):
            actual = value.get(name, default)
            if not isinstance(actual, bool):
                raise TypeError(f"temporal_relation_bridge.{name} must be a boolean")
            flags[name] = actual
        protocol = value.get("protocol_version", PHASE_D_PROTOCOL_VERSION)
        if protocol != PHASE_D_PROTOCOL_VERSION:
            raise ValueError("Unsupported temporal relation bridge protocol version")
        max_windows = value.get("max_refined_windows", 2)
        if isinstance(max_windows, bool) or max_windows != 2:
            raise ValueError("temporal_relation_bridge.max_refined_windows must equal 2")
        if flags["preserve_base_retrieval_top_k"] is not True:
            raise ValueError("temporal_relation_bridge.preserve_base_retrieval_top_k must be true")
        return cls(
            protocol_version=protocol,
            max_refined_windows=max_windows,
            **flags,
        )


@dataclass(frozen=True)
class Phase7Config:
    release_lock: Path
    query_file: Path
    classification_file: Path
    immutable_files: tuple[Path, ...] = ()
    immutable_dirs: tuple[Path, ...] = ()
    spec_file: Path = Path("docs/superpowers/specs/2026-10-03-phase7-semantic-evidence-reconstruction-design.md")
    plan_file: Path = Path("docs/superpowers/plans/2026-10-03-phase7-semantic-evidence-reconstruction.md")
    protocol_version: str = PROTOCOL_VERSION
    context_limit_chars: int = 12000
    max_candidate_bundles: int = 2
    completion_rounds: int = 1
    legacy_raw_packing: bool = False
    legacy_graph_expansion: bool = False
    legacy_cost_pruning: bool = False
    visual_completion: bool = False
    temporal_relation_bridge: TemporalRelationBridgeConfig = field(
        default_factory=TemporalRelationBridgeConfig
    )
    output_roots: tuple[Path, ...] = (
        Path("indexes/phase7_semantic_v1"),
        Path("results/phase7_warsaw12"),
        Path("experiments/phase7/annotations"),
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Phase7Config":
        if not isinstance(value, dict):
            raise TypeError("Phase7 config must be an object")
        required = ("release_lock", "query_file", "classification_file")
        missing = [name for name in required if not value.get(name)]
        if missing:
            raise ValueError(f"Missing Phase7 config fields: {', '.join(missing)}")
        protocol = value.get("protocol_version", PROTOCOL_VERSION)
        if protocol != PROTOCOL_VERSION:
            raise ValueError("Unsupported Phase7 protocol version")
        fixed = {
            "context_limit_chars": 12000,
            "max_candidate_bundles": 2,
            "completion_rounds": 1,
        }
        for name, expected in fixed.items():
            actual = value.get(name, expected)
            if isinstance(actual, bool) or actual != expected:
                raise ValueError(f"{name} must equal {expected}")
        flags = {}
        for name in (
            "legacy_raw_packing", "legacy_graph_expansion",
            "legacy_cost_pruning", "visual_completion",
        ):
            actual = value.get(name, False)
            if not isinstance(actual, bool):
                raise TypeError(f"{name} must be a boolean")
            flags[name] = actual
        return cls(
            release_lock=Path(value["release_lock"]),
            query_file=Path(value["query_file"]),
            classification_file=Path(value["classification_file"]),
            immutable_files=_paths(value.get("immutable_files")),
            immutable_dirs=_paths(value.get("immutable_dirs")),
            spec_file=Path(value.get("spec_file", cls.__dataclass_fields__["spec_file"].default)),
            plan_file=Path(value.get("plan_file", cls.__dataclass_fields__["plan_file"].default)),
            output_roots=_paths(value.get("output_roots")) or cls.__dataclass_fields__["output_roots"].default,
            metadata=sanitize(value.get("metadata", {})),
            temporal_relation_bridge=TemporalRelationBridgeConfig.from_dict(
                value.get("temporal_relation_bridge")
            ),
            **fixed,
            **flags,
        )

    def to_dict(self) -> dict[str, Any]:
        return sanitize(asdict(self))

    def output_allowed(self, path: Path) -> bool:
        candidate_parts = tuple(part.lower() for part in Path(path).resolve().parts)
        for configured in self.output_roots:
            root_parts = tuple(part.lower() for part in configured.parts)
            if not root_parts or len(root_parts) > len(candidate_parts):
                continue
            if any(candidate_parts[index:index + len(root_parts)] == root_parts
                   for index in range(len(candidate_parts) - len(root_parts) + 1)):
                return True
        return False
