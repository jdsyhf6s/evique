from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
import math
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints


class ObservationSource(str, Enum):
    OFFLINE_CAPTION = "offline_caption"
    TRANSCRIPT = "transcript"
    QUERY_CONDITIONED_OBSERVATION = "query_conditioned_observation"
    DETECTOR_OBSERVATION = "detector_observation"
    RULE_DERIVED_OBSERVATION = "rule_derived_observation"


class SourceFamily(str, Enum):
    DETECTOR = "DETECTOR"
    TRACK_DERIVED = "TRACK_DERIVED"
    EVENT_DERIVED = "EVENT_DERIVED"
    SCOPE_SEMANTIC = "SCOPE_SEMANTIC"
    VISUAL_SEMANTIC = "VISUAL_SEMANTIC"
    OTHER_INDEPENDENT = "OTHER_INDEPENDENT"


class IdentityStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICT = "CONFLICT"


class SupportLevel(str, Enum):
    FULL_SUPPORT = "FULL_SUPPORT"
    CONDITIONAL_SUPPORT = "CONDITIONAL_SUPPORT"
    NO_SUPPORT = "NO_SUPPORT"


class CaptionStatus(str, Enum):
    VALID = "valid"
    EMPTY = "empty"
    FAILED = "failed"
    PLACEHOLDER = "placeholder"


class TranscriptStatus(str, Enum):
    VALID = "valid"
    EMPTY = "empty"
    FAILED = "failed"


class RequirementType(str, Enum):
    TARGET_ATTRIBUTE = "TARGET_ATTRIBUTE"
    EXISTENCE = "EXISTENCE"
    COUNT = "COUNT"
    SPATIAL_RELATION = "SPATIAL_RELATION"
    SCENE_REGION = "SCENE_REGION"
    MOTION = "MOTION"
    ENTRY_EXIT = "ENTRY_EXIT"
    INTERACTION = "INTERACTION"
    BEFORE_STATE = "BEFORE_STATE"
    AFTER_STATE = "AFTER_STATE"
    STATE_CHANGE = "STATE_CHANGE"
    COMPARISON = "COMPARISON"
    EVENT = "EVENT"


class Optionality(str, Enum):
    MANDATORY = "mandatory"
    OPTIONAL = "optional"


class RequirementStatus(str, Enum):
    PARSED = "parsed"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"


class ParseStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"


class LogicOperator(str, Enum):
    AND = "AND"
    OR = "OR"


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {name}")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} values must be nonempty strings")


def _validate_json(value: Any, path: str = "provenance") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.endswith("_id") and item is not None and not isinstance(item, str):
                raise TypeError(f"{path} identifiers must be strings")
            if key.endswith("_ids") and item is not None and (
                not isinstance(item, (list, tuple)) or any(not isinstance(entry, str) for entry in item)
            ):
                raise TypeError(f"{path} identifiers must be strings")
            _validate_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value")


def _instance(value: Any, cls: type, name: str) -> None:
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be a {cls.__name__} instance")


def _tuple_of(values: tuple, cls: type, name: str) -> None:
    _instance(values, tuple, name)
    for value in values:
        _instance(value, cls, name)


def _id_tuple(values: tuple[str, ...], name: str) -> None:
    _tuple_of(values, str, name)
    for value in values:
        _text(value, name)
    _unique(values, name)


def _optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _text(value, name)


def _json_object(value: dict[str, Any], name: str) -> None:
    _instance(value, dict, name)
    _validate_json(value, name)


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        start = _finite(self.start, "start")
        end = _finite(self.end, "end")
        if start < 0 or end < 0 or start > end:
            raise ValueError("time range must be nonnegative and ordered")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class IdentityEvidence:
    evidence_id: str
    observed_category: str
    source_family: SourceFamily
    derived_from: tuple[str, ...]
    independence_root_ids: tuple[str, ...]
    episode_id: str | None
    local_target_id: str
    time_span: TimeRange
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("evidence_id", "observed_category", "local_target_id"):
            _text(getattr(self, name), name)
        _instance(self.source_family, SourceFamily, "source_family")
        _id_tuple(self.derived_from, "derived_from")
        _id_tuple(self.independence_root_ids, "independence_root_ids")
        _optional_text(self.episode_id, "episode_id")
        _instance(self.time_span, TimeRange, "time_span")
        _json_object(self.provenance, "provenance")


@dataclass(frozen=True)
class IdentityAssessment:
    status: IdentityStatus
    episode_id: str | None
    local_target_id: str
    observed_labels: tuple[IdentityEvidence, ...]
    independent_source_groups: tuple[dict[str, Any], ...]
    compatible_category: str | None
    rationale_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _instance(self.status, IdentityStatus, "status")
        _optional_text(self.episode_id, "episode_id")
        _text(self.local_target_id, "local_target_id")
        _tuple_of(self.observed_labels, IdentityEvidence, "observed_labels")
        _tuple_of(self.independent_source_groups, dict, "independent_source_groups")
        for group in self.independent_source_groups:
            _json_object(group, "independent_source_groups")
        _optional_text(self.compatible_category, "compatible_category")
        _id_tuple(self.rationale_codes, "rationale_codes")


@dataclass(frozen=True)
class TrackEpisode:
    episode_id: str
    raw_track_id: str
    video_id: str
    start_time: float
    end_time: float
    ordered_observation_ids: tuple[str, ...]
    category_evidence: tuple[IdentityEvidence, ...]
    source_provenance: dict[str, Any]
    split_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("episode_id", "raw_track_id", "video_id"):
            _text(getattr(self, name), name)
        time_range = TimeRange(self.start_time, self.end_time)
        object.__setattr__(self, "start_time", time_range.start)
        object.__setattr__(self, "end_time", time_range.end)
        _id_tuple(self.ordered_observation_ids, "ordered_observation_ids")
        _tuple_of(self.category_evidence, IdentityEvidence, "category_evidence")
        _json_object(self.source_provenance, "source_provenance")
        _id_tuple(self.split_reasons, "split_reasons")


@dataclass(frozen=True)
class SupportComponentDecision:
    component_id: str
    level: SupportLevel
    fact_ids: tuple[str, ...]
    observation_only_fact_ids: tuple[str, ...]
    identity_assessment_refs: tuple[str, ...]
    episode_ids: tuple[str, ...]
    provenance_root_ids: tuple[str, ...]
    rationale_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.component_id, "component_id")
        _instance(self.level, SupportLevel, "level")
        for name in ("fact_ids", "observation_only_fact_ids", "identity_assessment_refs",
                     "episode_ids", "provenance_root_ids", "rationale_codes"):
            _id_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class RequirementSupportDecision:
    requirement_id: str
    level: SupportLevel
    supporting_fact_ids: tuple[str, ...]
    observation_only_fact_ids: tuple[str, ...]
    component_decisions: tuple[SupportComponentDecision, ...]
    identity_assessment_refs: tuple[str, ...]
    provenance_root_ids: tuple[str, ...]
    rationale_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.requirement_id, "requirement_id")
        _instance(self.level, SupportLevel, "level")
        _tuple_of(self.component_decisions, SupportComponentDecision, "component_decisions")
        for name in ("supporting_fact_ids", "observation_only_fact_ids",
                     "identity_assessment_refs", "provenance_root_ids", "rationale_codes"):
            _id_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class ScopeRecord:
    scope_id: str
    video_id: str
    segment_id: str
    time_range: TimeRange
    caption_text: str = ""
    caption_status: CaptionStatus = CaptionStatus.EMPTY
    caption_source: ObservationSource = ObservationSource.OFFLINE_CAPTION
    transcript_text: str = ""
    transcript_status: TranscriptStatus = TranscriptStatus.EMPTY
    transcript_source: ObservationSource = ObservationSource.TRANSCRIPT
    source_frame_ids: tuple[str, ...] = ()
    source_frame_timestamps: tuple[float, ...] = ()
    model_id: str | None = None
    prompt_version: str | None = None
    query_hash: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    quality_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("scope_id", "video_id", "segment_id"):
            _text(getattr(self, name), name)
        if self.transcript_source is not ObservationSource.TRANSCRIPT:
            raise ValueError("transcript_source must be transcript")
        if self.caption_source not in {
            ObservationSource.OFFLINE_CAPTION,
            ObservationSource.QUERY_CONDITIONED_OBSERVATION,
        }:
            raise ValueError("invalid caption_source")
        if (self.caption_source is ObservationSource.QUERY_CONDITIONED_OBSERVATION
                and not self.query_hash):
            raise ValueError("query_conditioned_observation requires query_hash")
        if len(self.source_frame_ids) != len(self.source_frame_timestamps):
            raise ValueError("source frame IDs and timestamps must align")
        _unique(self.source_frame_ids, "source_frame_id")
        for timestamp in self.source_frame_timestamps:
            point = _finite(timestamp, "source frame timestamp")
            if point < self.time_range.start or point > self.time_range.end:
                raise ValueError("source frame timestamp outside time_range")
        if self.caption_status is CaptionStatus.VALID and (
            not self.caption_text.strip()
            or not self.source_frame_ids
            or not self.model_id
            or not self.prompt_version
        ):
            raise ValueError(
                "valid caption requires text, source frame, model_id, and prompt_version"
            )
        if not self.provenance or self.provenance.get("video_id") != self.video_id:
            raise ValueError("Scope provenance must preserve video identity")
        _validate_json(self.provenance)


@dataclass(frozen=True)
class SourceTextSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("span offsets must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must be nonempty and ordered")
        _text(self.text, "span text")


@dataclass(frozen=True)
class RoleReference:
    role_id: str
    category: str | None = None
    modifiers: tuple[str, ...] = ()
    coreference_target: str | None = None

    def __post_init__(self) -> None:
        _text(self.role_id, "role_id")
        _unique(self.modifiers, "modifier")


@dataclass(frozen=True)
class RegionReference:
    role_id: str
    label: str | None = None

    def __post_init__(self) -> None:
        _text(self.role_id, "region role_id")


@dataclass(frozen=True)
class TemporalReference:
    anchor_role_id: str
    direction: str
    aspect: str | None = None

    def __post_init__(self) -> None:
        _text(self.anchor_role_id, "anchor_role_id")
        if self.direction not in {"before", "after", "around", "before_after"}:
            raise ValueError("invalid temporal direction")


@dataclass(frozen=True)
class QuestionRequirement:
    requirement_id: str
    source_text_span: SourceTextSpan
    requirement_type: RequirementType
    subject: RoleReference | None = None
    attributes: tuple[str, ...] = ()
    predicate: str | None = None
    reference_object: RoleReference | None = None
    reference_region: RegionReference | None = None
    temporal_relation: TemporalReference | None = None
    comparison_target: RoleReference | None = None
    optionality: Optionality = Optionality.MANDATORY
    status: RequirementStatus = RequirementStatus.PARSED
    logic_group_id: str = "root"
    logic_operator: LogicOperator = LogicOperator.AND
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.requirement_id, "requirement_id")
        _text(self.logic_group_id, "logic_group_id")
        _unique(self.attributes, "attribute")
        _unique(self.depends_on, "dependency")


@dataclass(frozen=True)
class BooleanExpression:
    operator: LogicOperator
    children: tuple[Union[str, "BooleanExpression"], ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("BooleanExpression requires children")
        for child in self.children:
            if isinstance(child, str):
                _text(child, "BooleanExpression child")
            elif not isinstance(child, BooleanExpression):
                raise TypeError("BooleanExpression children must be IDs or expressions")


@dataclass(frozen=True)
class QuestionRequirementSet:
    query_hash: str
    parser_version: str
    parse_status: ParseStatus
    requirements: tuple[QuestionRequirement, ...]
    root_logic: BooleanExpression
    uncovered_spans: tuple[SourceTextSpan, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.query_hash, "query_hash")
        _text(self.parser_version, "parser_version")
        identifiers = tuple(item.requirement_id for item in self.requirements)
        _unique(identifiers, "requirement_id")
        def logic_ids(expression: BooleanExpression) -> set[str]:
            result = set()
            for child in expression.children:
                result.add(child) if isinstance(child, str) else result.update(logic_ids(child))
            return result
        if logic_ids(self.root_logic) != set(identifiers):
            raise ValueError("root logic must reference every and only requirement ID")
        if any(dependency not in set(identifiers)
               for requirement in self.requirements for dependency in requirement.depends_on):
            raise ValueError("requirement dependency is unknown")


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    fact_type: str
    video_id: str
    time_span: TimeRange
    frame_ids: tuple[str, ...] = ()
    role_bindings: dict[str, str] = field(default_factory=dict)
    value: Any = None
    source_evidence_ids: tuple[str, ...] = ()
    source_kind: ObservationSource = ObservationSource.RULE_DERIVED_OBSERVATION
    derivation_rule: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    # None is retained for pre-normalization legacy facts only.
    source_family: SourceFamily | None = None
    derived_from: tuple[str, ...] = ()
    independence_root_ids: tuple[str, ...] = ()
    episode_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("fact_id", "fact_type", "video_id"):
            _text(getattr(self, name), name)
        _unique(self.frame_ids, "frame_id")
        _unique(self.source_evidence_ids, "source_evidence_id")
        if self.source_family is not None:
            _instance(self.source_family, SourceFamily, "source_family")
        _id_tuple(self.derived_from, "derived_from")
        _id_tuple(self.independence_root_ids, "independence_root_ids")
        _optional_text(self.episode_id, "episode_id")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in self.role_bindings.items()):
            raise TypeError("role_bindings must map strings to strings")
        _validate_json(self.value, "fact value")
        _validate_json(self.provenance)


@dataclass(frozen=True)
class SubjectBinding:
    role_id: str
    subject_ids: tuple[str, ...]
    track_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.role_id, "role_id")
        _unique(self.subject_ids, "subject_id")


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    video_id: str
    time_window: TimeRange
    focal_subjects: dict[str, SubjectBinding] = field(default_factory=dict)
    scene_evidence: tuple[str, ...] = ()
    object_evidence: tuple[str, ...] = ()
    relation_evidence: tuple[str, ...] = ()
    track_evidence: tuple[str, ...] = ()
    temporal_evidence: tuple[str, ...] = ()
    event_evidence: tuple[str, ...] = ()
    supported_requirement_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    token_cost: int = 0
    consistency: dict[str, Any] = field(default_factory=dict)
    utility: dict[str, Any] = field(default_factory=dict)
    requirement_support_decisions: dict[str, RequirementSupportDecision] = field(default_factory=dict)
    conditional_requirement_ids: tuple[str, ...] = ()
    no_support_requirement_ids: tuple[str, ...] = ()
    # This keyword-only init marker preserves legacy ``missing=()`` behavior
    # while allowing deserialization to retain a logical empty set.
    _missing_requirement_ids_explicit: InitVar[bool] = field(default=False, kw_only=True)

    def __post_init__(self, _missing_requirement_ids_explicit: bool = False) -> None:
        _text(self.bundle_id, "bundle_id")
        _text(self.video_id, "video_id")
        evidence = (
            self.scene_evidence + self.object_evidence + self.relation_evidence
            + self.track_evidence + self.temporal_evidence + self.event_evidence
        )
        _unique(evidence, "fact_id")
        projection_levels = (
            ("supported_requirement_ids", SupportLevel.FULL_SUPPORT),
            ("conditional_requirement_ids", SupportLevel.CONDITIONAL_SUPPORT),
            ("no_support_requirement_ids", SupportLevel.NO_SUPPORT),
        )
        for name, _ in projection_levels:
            _id_tuple(getattr(self, name), name)
        _instance(self.requirement_support_decisions, dict, "requirement_support_decisions")
        for identifier, decision in self.requirement_support_decisions.items():
            _text(identifier, "requirement_support_decisions key")
            _instance(decision, RequirementSupportDecision, "requirement_support_decisions value")
            if identifier != decision.requirement_id:
                raise ValueError("requirement_support_decisions key must match requirement_id")
        if self.requirement_support_decisions:
            for name, level in projection_levels:
                identifiers = tuple(identifier for identifier, decision
                                    in self.requirement_support_decisions.items()
                                    if decision.level is level)
                supplied = getattr(self, name)
                if supplied and supplied != identifiers:
                    raise ValueError(f"{name} disagrees with requirement_support_decisions")
                object.__setattr__(self, name, identifiers)
        if not isinstance(_missing_requirement_ids_explicit, bool):
            raise TypeError("_missing_requirement_ids_explicit must be bool")
        _id_tuple(self.missing_requirement_ids, "missing_requirement_ids")
        if not self.missing_requirement_ids and not _missing_requirement_ids_explicit:
            object.__setattr__(self, "missing_requirement_ids",
                               self.conditional_requirement_ids + self.no_support_requirement_ids)
        projections = [set(getattr(self, name)) for name, _ in projection_levels]
        if any(projections[left] & projections[right]
               for left in range(len(projections)) for right in range(left + 1, len(projections))):
            raise ValueError("full, conditional, and no-support requirement IDs must be disjoint")
        if set(self.supported_requirement_ids) & set(self.missing_requirement_ids):
            raise ValueError("requirement cannot be supported and missing")
        if isinstance(self.token_cost, bool) or not isinstance(self.token_cost, int) or self.token_cost < 0:
            raise ValueError("token_cost must be a nonnegative integer")
        _validate_json(self.provenance)
        _validate_json(self.consistency, "consistency")
        _validate_json(self.utility, "utility")


@dataclass(frozen=True)
class SceneState:
    state_id: str
    video_id: str
    time_window: TimeRange
    object_categories: tuple[str, ...] = ()
    object_counts: dict[str, int] = field(default_factory=dict)
    motion_summary: str | None = None
    queue_like_observation: dict[str, Any] | None = None
    traffic_density_proxy: float | None = None
    scene_caption: str | None = None
    source_fact_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.state_id, "state_id")
        _text(self.video_id, "video_id")
        _unique(self.object_categories, "object category")
        _unique(self.source_fact_ids, "source fact_id")
        if any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0
               for key, value in self.object_counts.items()):
            raise ValueError("object_counts must be nonnegative integer counts")
        if self.traffic_density_proxy is not None:
            _finite(self.traffic_density_proxy, "traffic_density_proxy")
        _validate_json(self.queue_like_observation, "queue_like_observation")
        _validate_json(self.provenance)


T = TypeVar("T")


def to_dict(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError("to_dict requires a Phase 7 dataclass")

    def convert(item: Any) -> Any:
        if isinstance(item, Enum):
            return item.value
        if is_dataclass(item):
            return {field.name: convert(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(entry) for entry in item]
        return item

    return convert(value)


def _construct(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _construct(option, value)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        raise TypeError("value does not match union") from (errors[-1] if errors else None)
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("tuple fields must be JSON arrays")
        subtype = args[0] if args else Any
        return tuple(_construct(subtype, entry) for entry in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError("dict field must be a JSON object")
        key_type, val_type = args or (Any, Any)
        return {_construct(key_type, key): _construct(val_type, val) for key, val in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return from_dict(annotation, value)
    if annotation in (str, int, float, bool):
        if not isinstance(value, annotation):
            raise TypeError(f"expected {annotation.__name__}")
        return value
    return value


def from_dict(cls: type[T], value: dict[str, Any]) -> T:
    if not isinstance(value, dict) or not is_dataclass(cls):
        raise TypeError("from_dict requires a dataclass type and object")
    declared = {field.name: field for field in fields(cls)}
    unknown = sorted(set(value) - set(declared))
    if unknown:
        raise ValueError(f"unknown fields for {cls.__name__}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    converted = {
        name: _construct(hints[name], item)
        for name, item in value.items()
    }
    if cls is EvidenceBundle and value.get("missing_requirement_ids") == []:
        converted["_missing_requirement_ids_explicit"] = True
    return cls(**converted)
