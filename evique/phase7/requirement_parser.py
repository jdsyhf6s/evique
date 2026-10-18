from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Any, Protocol

from .schema import (
    BooleanExpression,
    LogicOperator,
    Optionality,
    ParseStatus,
    QuestionRequirement,
    QuestionRequirementSet,
    RegionReference,
    RequirementStatus,
    RequirementType,
    RoleReference,
    SourceTextSpan,
    TemporalReference,
    from_dict,
)


PARSER_VERSION = "phase7_requirement_rules_v1"
COLORS = ("red", "blue", "green", "yellow", "white", "black", "gray", "grey")
CATEGORY_SYNONYMS = {
    "cars": "car", "vehicles": "vehicle", "trucks": "truck", "buses": "bus",
    "people": "person", "persons": "person", "pedestrians": "pedestrian",
    "bicycles": "bicycle", "cyclists": "cyclist",
}
CATEGORIES = {
    "car", "vehicle", "truck", "bus", "person", "pedestrian", "bicycle", "cyclist",
    "traffic", "road", "congestion", "车辆",
}
MOTION_WORDS = ("moving", "moves", "stopped", "stops", "turning", "turns", "travelling", "traveling")
ENTRY_WORDS = ("enters", "enter", "entered", "exits", "exit", "leaves", "leave", "进入", "离开")
INTERACTION_WORDS = (
    "follows", "follow", "overtakes", "overtake", "approaches", "approach",
    "passes", "pass", "interacting", "interaction", "interact",
)
SPATIAL_PATTERNS = {
    "left_of": r"left\s+of", "right_of": r"right\s+of", "above": r"above",
    "below": r"below", "behind": r"behind", "in_front_of": r"in\s+front\s+of",
    "near": r"\bnear(?:by|est(?:\s+to)?)?\b|\bbeside\b|\bimmediately\s+around\b|\bclose\s+together\b|\bpositioned\s+relative\b",
    "overlap": r"overlap(?:s|ping)?",
}


class RequirementParseClient(Protocol):
    def parse(self, query: str) -> dict: ...


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _full_span(query: str) -> SourceTextSpan:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    return SourceTextSpan(0, len(query), query)


def _categories(query: str) -> list[tuple[str, tuple[str, ...]]]:
    matches = []
    for match in re.finditer(r"[\w]+", query, flags=re.UNICODE):
        raw = match.group(0).casefold()
        category = CATEGORY_SYNONYMS.get(raw, raw)
        if category in CATEGORIES:
            prefix = query[max(0, match.start() - 12):match.start()].casefold()
            modifiers = tuple(color for color in COLORS if re.search(rf"\b{color}\s+$", prefix))
            matches.append((category, modifiers))
    # Preserve mentions because repeated categories can be distinct semantic roles.
    return matches


def _role(index: int, category: str, modifiers: tuple[str, ...]) -> RoleReference:
    return RoleReference(f"role-{index + 1}", category=category, modifiers=modifiers)


def _rule_parse(query: str) -> QuestionRequirementSet:
    lower = query.casefold()
    span = _full_span(query)
    mentions = _categories(query)
    roles = [_role(index, category, modifiers) for index, (category, modifiers) in enumerate(mentions)]
    requirements: list[QuestionRequirement] = []

    def add(requirement_type: RequirementType, *, subject: RoleReference | None = None,
            predicate: str | None = None, reference: RoleReference | None = None,
            region: RegionReference | None = None, temporal: TemporalReference | None = None,
            attributes: tuple[str, ...] = (), depends: tuple[str, ...] = (),
            status: RequirementStatus = RequirementStatus.PARSED) -> QuestionRequirement:
        item = QuestionRequirement(
            requirement_id=f"req-{len(requirements) + 1:03d}",
            source_text_span=span,
            requirement_type=requirement_type,
            subject=subject,
            attributes=attributes,
            predicate=predicate,
            reference_object=reference,
            reference_region=region,
            temporal_relation=temporal,
            optionality=Optionality.MANDATORY,
            status=status,
            logic_group_id="root",
            logic_operator=LogicOperator.OR if " or " in lower else LogicOperator.AND,
            depends_on=depends,
        )
        requirements.append(item)
        return item

    count_query = bool(re.search(r"\bhow\s+many\b", lower))
    if count_query:
        add(RequirementType.COUNT, subject=roles[0] if roles else None, predicate="count")

    # Existence keeps each requested role explicit, including temporal anchors.
    if roles and not count_query:
        for role in roles:
            add(RequirementType.EXISTENCE, subject=role, predicate="exist")
    for role in roles:
        if role.modifiers:
            add(RequirementType.TARGET_ATTRIBUTE, subject=role, attributes=role.modifiers, predicate="attribute")

    motion = next((word for word in MOTION_WORDS if re.search(rf"\b{re.escape(word)}\b", lower)), None)
    if motion:
        add(RequirementType.MOTION, subject=roles[0] if roles else None, predicate=motion)

    interaction = next((word for word in INTERACTION_WORDS if re.search(rf"\b{word}\b", lower)), None)
    if interaction:
        add(RequirementType.INTERACTION, subject=roles[0] if roles else None,
            reference=roles[1] if len(roles) > 1 else None, predicate=interaction,
            status=RequirementStatus.PARSED if len(roles) > 1 else RequirementStatus.INCOMPLETE)

    spatial = next(((name, pattern) for name, pattern in SPATIAL_PATTERNS.items()
                    if re.search(pattern, lower)), None)
    if spatial:
        add(RequirementType.SPATIAL_RELATION, subject=roles[0] if roles else None,
            reference=roles[1] if len(roles) > 1 else None, predicate=spatial[0],
            status=RequirementStatus.PARSED if len(roles) > 1 else RequirementStatus.INCOMPLETE)

    entry = next((word for word in ENTRY_WORDS if word in lower), None)
    anchor = None
    if entry:
        anchor_subject = roles[0] if roles else None
        region = RegionReference("region-1", label="area") if "area" in lower else None
        anchor = add(RequirementType.ENTRY_EXIT, subject=anchor_subject, predicate=entry, region=region)

    if any(word in lower for word in ("congestion", "congested", "拥堵", "event", "occur")):
        add(RequirementType.EVENT, subject=roles[0] if roles else None,
            predicate="congestion" if any(word in lower for word in ("congestion", "congested", "拥堵")) else "occur")

    progression = bool(re.search(r"\bprogress(?:es|ed|ing)?\s+from\b.+\bto\b", lower))
    has_before = "before" in lower or progression or ("change" in lower and "after" in lower)
    has_after = "after" in lower or progression or "后" in query
    if has_before or has_after:
        if anchor is None:
            anchor = add(RequirementType.EVENT, subject=roles[0] if roles else None,
                         predicate="progress" if progression else "anchor",
                         status=RequirementStatus.PARSED if progression else RequirementStatus.INCOMPLETE)
        before_req = after_req = None
        if has_before:
            before_req = add(RequirementType.BEFORE_STATE, subject=roles[0] if roles else None,
                             predicate="state", temporal=TemporalReference(anchor.requirement_id, "before"),
                             depends=(anchor.requirement_id,))
        if has_after:
            after_req = add(RequirementType.AFTER_STATE, subject=roles[0] if roles else None,
                            predicate="state", temporal=TemporalReference(anchor.requirement_id, "after"),
                            depends=(anchor.requirement_id,))
        if before_req and after_req:
            add(RequirementType.STATE_CHANGE, subject=roles[0] if roles else None,
                predicate="change", temporal=TemporalReference(anchor.requirement_id, "before_after"),
                depends=(before_req.requirement_id, after_req.requirement_id))

    if progression and len(roles) > 1:
        add(RequirementType.COMPARISON, subject=roles[0], reference=roles[1],
            predicate="dominant" if "dominant" in lower else "progress")

    if any(label in lower for label in ("central", "center", "middle", "lower")):
        label = next(label for label in ("central", "center", "middle", "lower") if label in lower)
        add(RequirementType.SCENE_REGION, subject=roles[0] if roles else None,
            predicate=label, region=RegionReference("region-1", label=label))

    unknown_match = re.search(r"\b(?:why|shimmer(?:s|ed|ing)?|mysteriously)\b.*", query, flags=re.I)
    uncovered = (SourceTextSpan(unknown_match.start(), unknown_match.end(), unknown_match.group(0)),) if unknown_match else ()
    incomplete_fields = any(item.status is not RequirementStatus.PARSED for item in requirements)
    if not requirements:
        add(RequirementType.EVENT, predicate=None, status=RequirementStatus.INCOMPLETE)
        uncovered = (span,)
        incomplete_fields = True

    status = ParseStatus.INCOMPLETE if uncovered or incomplete_fields else ParseStatus.COMPLETE
    ids = tuple(item.requirement_id for item in requirements)
    contrastive_or = "whether" in lower and " or only " in lower
    if " or " in lower and len(ids) > 1 and not contrastive_or:
        root = BooleanExpression(LogicOperator.OR, ids)
    else:
        root = BooleanExpression(LogicOperator.AND, ids)
    return QuestionRequirementSet(
        query_hash=_query_hash(query), parser_version=PARSER_VERSION, parse_status=status,
        requirements=tuple(requirements), root_logic=root, uncovered_spans=uncovered,
        warnings=(),
    )


def _grounded(query: str, value: str | None) -> bool:
    if value is None:
        return True
    words = re.findall(r"[\w]+", value.casefold().replace("_", " ").replace("-", " "), flags=re.UNICODE)
    query_words = set(re.findall(r"[\w]+", query.casefold(), flags=re.UNICODE))
    approved = {CATEGORY_SYNONYMS.get(word, word) for word in query_words}
    approved.update({"exist", "attribute", "count", "state", "change", "anchor"})
    if "changes" in query_words:
        approved.add("change")
    if "change" in query_words and "after" in query_words:
        approved.add("before")
    if "progress" in query_words and "from" in query_words and "to" in query_words:
        approved.update({"before", "after", "change"})
    if any(phrase in query.casefold() for phrase in (
        "nearby", "beside", "immediately around", "close together", "positioned relative",
    )):
        approved.add("near")
    if "拥堵" in query:
        approved.add("congestion")
    if "后" in query:
        approved.add("after")
    return all(
        word in query_words
        or word in approved
        or (any(ord(character) > 127 for character in word) and word in query.casefold())
        for word in words
    )


def _logic_ids(expression: BooleanExpression) -> set[str]:
    result = set()
    for child in expression.children:
        if isinstance(child, str):
            result.add(child)
        else:
            result.update(_logic_ids(child))
    return result


def validate_requirement_parse(query: str, value: dict) -> QuestionRequirementSet:
    if not isinstance(value, dict):
        raise TypeError("requirement parse must be an object")
    result = from_dict(QuestionRequirementSet, value)
    if result.query_hash != _query_hash(query):
        raise ValueError("query hash mismatch")
    identifiers = {item.requirement_id for item in result.requirements}
    if _logic_ids(result.root_logic) != identifiers:
        raise ValueError("logic tree must reference every and only requirement ID")
    for item in result.requirements:
        span = item.source_text_span
        if span.end > len(query) or query[span.start:span.end] != span.text:
            raise ValueError("source span does not match original query")
        fields = [item.predicate, *item.attributes]
        for role in (item.subject, item.reference_object, item.comparison_target):
            if role:
                fields.extend([role.category, *role.modifiers, role.coreference_target])
        if item.reference_region:
            fields.append(item.reference_region.label)
        if item.temporal_relation:
            fields.extend([item.temporal_relation.direction, item.temporal_relation.aspect])
        if any(not _grounded(query, field) for field in fields):
            raise ValueError("requirement content is not grounded in query")
        if any(dependency not in identifiers for dependency in item.depends_on):
            raise ValueError("requirement dependency is unknown")
    for span in result.uncovered_spans:
        if span.end > len(query) or query[span.start:span.end] != span.text:
            raise ValueError("uncovered span does not match original query")
    return result


def parse_requirements(query: str, llm_client: RequirementParseClient | None = None) -> QuestionRequirementSet:
    local = _rule_parse(query)
    if local.parse_status is ParseStatus.COMPLETE or llm_client is None:
        return local
    try:
        candidate = llm_client.parse(query)
        return validate_requirement_parse(query, candidate)
    except Exception:
        return replace(local, warnings=local.warnings + ("fallback_rejected",))
