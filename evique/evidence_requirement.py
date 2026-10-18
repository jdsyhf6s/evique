"""Query demands only: these descriptors never assert an observed video fact."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class EvidenceRequirement:
    target_requirement: dict[str, object] = field(default_factory=dict)
    anchor_requirement: dict[str, object] = field(default_factory=dict)
    temporal_requirement: dict[str, object] = field(default_factory=dict)
    comparison_requirement: dict[str, object] = field(default_factory=dict)
    event_requirement: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_CATEGORIES = {
    "animal", "bag", "ball", "bicycle", "bus", "car", "cat", "chair", "dog",
    "motorcycle", "object", "person", "player", "table", "truck", "vehicle",
}
_PLURALS = {word + "s": word for word in _CATEGORIES}
_PLURALS.update({"buses": "bus", "people": "person"})
_ATTRIBUTES = {
    "black", "blue", "brown", "gray", "green", "grey", "large", "purple",
    "red", "small", "striped", "white", "yellow",
}
_EVENT_FORMS = {
    "enter": {"enter", "enters", "entered", "entering"},
    "exit": {"exit", "exits", "exited", "exiting", "leave", "leaves", "leaving", "depart", "departs", "departed"},
    "stop": {"stop", "stops", "stopped", "stopping"},
    "move": {"move", "moves", "moved", "moving"},
}
_SPEECH = {"speaker", "audio", "say", "said", "speak", "speech", "talk", "transcript"}


def parse_requirement(query: str) -> tuple[EvidenceRequirement, list[str]]:
    """Parse a bounded English vocabulary without grounding question premises.

    Unknown wording stays unspecified. A state change around an anchor requests
    both sides for validation; a simple before/after query requests only its side.
    Negation does not convert an event descriptor into an assertion of occurrence.
    """
    words = re.findall(r"[a-z]+", query.lower())
    terms = set(words)
    normalized = [_PLURALS.get(word, word) for word in words]
    categories = list(dict.fromkeys(word for word in normalized if word in _CATEGORIES))
    attributes = sorted(terms & _ATTRIBUTES)
    actions = [event for event, forms in _EVENT_FORMS.items() if terms & forms]
    congestion = bool(terms & {"congestion", "congested"}) or bool(
        re.search(r"\btraffic\s+jams?\b", query.lower())
    )
    temporal_terms = terms & {"before", "after", "then", "earlier", "later"}
    aspect = "traffic state" if "traffic" in terms or congestion else "state"
    requirement = EvidenceRequirement()
    roles = []
    if categories:
        requirement.target_requirement = {"categories": categories}
        if attributes:
            requirement.target_requirement["attributes"] = attributes
        roles.append("target_identity")

    if temporal_terms:
        if actions:
            requirement.anchor_requirement = {"event_types": actions}
            if categories:
                requirement.anchor_requirement["target_categories"] = categories
            roles.append("anchor_event")
        state_change = bool(terms & {"become", "becomes", "became", "change", "changes", "changed"})
        both = (
            {"before", "after"} <= terms
            or (congestion and bool(actions) and (state_change or "then" in terms))
            or (bool(terms & {"compare", "comparison"}) and bool(temporal_terms))
        )
        before = both or bool(terms & {"before", "earlier"})
        after = both or bool(terms & {"after", "then", "later"})
        if before:
            requirement.temporal_requirement["before"] = aspect
            roles.append("before_state")
        if after:
            requirement.temporal_requirement["after"] = aspect
            roles.append("after_state")
        relation = "before_after" if before and after else "before" if before else "after"
        requirement.temporal_requirement["relation"] = relation
        if before and after:
            requirement.comparison_requirement = {"mode": "before_after", "aspect": aspect}
    elif len(categories) == 2 and terms & {"compare", "comparison", "versus", "vs"}:
        requirement.comparison_requirement = {
            "mode": "targets", "left": {"categories": [categories[0]]},
            "right": {"categories": [categories[1]]},
        }
        roles.extend(["comparison_left", "comparison_right"])

    events = (["congestion"] if congestion else []) + ([] if temporal_terms else actions)
    if events:
        requirement.event_requirement = {"event_types": events}
        if not temporal_terms:
            roles.append("event_evidence")
    if terms & _SPEECH:
        roles.append("transcript")
    if not roles:
        roles.append("event_evidence" if terms & {"event", "events", "activity", "activities"} else "scene_context")
    return requirement, roles
