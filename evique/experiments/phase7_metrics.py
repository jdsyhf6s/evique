"""Evaluation-only Phase 7 reference handling and offline metrics.

This module is intentionally outside ``evique.phase7``. Runtime code never
imports annotations from here; evaluation begins only after trace byte hashes
have been verified.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from evique.phase7.schema import (
    Optionality,
    QuestionRequirementSet,
    RequirementType,
    from_dict,
)


REFERENCE_PROTOCOL = "phase7-reference-v1"
CATEGORIES = {"direct", "relational_motion", "temporal_event"}
CLAIM_LABELS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_VERIFIABLE")


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    numerator: float | int | None
    denominator: int
    unknown_count: int = 0
    status: str = "available"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value, "numerator": self.numerator,
            "denominator": self.denominator, "unknown_count": self.unknown_count,
            "status": self.status, "details": self.details or {},
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _load(path: Path) -> Any:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_exclusive(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _query_rows(value: Any) -> list[dict]:
    rows = value.get("queries") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise TypeError("query file must contain a query list")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("query_id"), str) or not row["query_id"]:
            raise ValueError("each query requires query_id")
        query = row.get("query") or row.get("question")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("each query requires query text")
        result.append({**row, "query": query})
    ids = [row["query_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query_id")
    return result


def emit_reference_template(query_file: Path, paraphrase_file: Path,
                            output_path: Path) -> dict:
    originals = _query_rows(_load(query_file))
    variations = _query_rows(_load(paraphrase_file))
    original_ids = {row["query_id"] for row in originals}
    for row in variations:
        if row.get("source_query_id") not in original_ids:
            raise ValueError("paraphrase/holdout source_query_id is not frozen")
        if row.get("split") not in {"paraphrase", "holdout"}:
            raise ValueError("variation split must be paraphrase or holdout")
    all_ids = [row["query_id"] for row in originals + variations]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("original and variation IDs overlap")
    entries = []
    for row, split in [(row, "original") for row in originals] + [(row, row["split"]) for row in variations]:
        query = row["query"]
        entries.append({
            "query_id": row["query_id"],
            "source_query_id": row.get("source_query_id") or row["query_id"],
            "query": query,
            "source_query_hash": _query_hash(query),
            "category": row.get("category"),
            "split": split,
            "annotation": {
                "requirements": [], "roles": [], "root_logic": None,
                "reviewer": "", "review_status": "blank",
            },
        })
    payload = {
        "protocol_version": REFERENCE_PROTOCOL,
        "annotation_version": "",
        "annotator": "",
        "review_status": "blank_template_not_reviewed",
        "entries": entries,
    }
    _write_exclusive(output_path, payload)
    return {
        "schema_version": "phase7_reference_template_manifest_v1",
        "review_status": "blank_template_not_reviewed",
        "entry_count": len(entries),
        "template_sha256": _sha256(output_path),
        "query_file_sha256": _sha256(query_file),
        "paraphrase_file_sha256": _sha256(paraphrase_file),
    }


def _forbidden_annotation_paths(value: Any, path: str = "annotation") -> list[str]:
    forbidden = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if (
                normalized.startswith("answer") or normalized.startswith("evidence")
                or normalized.startswith("retrieval") or normalized.startswith("runtime_fact")
                or normalized in {"facts", "ground_truth", "label_answer"}
            ):
                forbidden.append(f"{path}.{key}")
            forbidden.extend(_forbidden_annotation_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            forbidden.extend(_forbidden_annotation_paths(item, f"{path}[{index}]"))
    return forbidden


def _logic_ids(value: Any) -> set[str]:
    if not isinstance(value, dict) or value.get("operator") not in {"AND", "OR"}:
        raise ValueError("reference root_logic is invalid")
    children = value.get("children")
    if not isinstance(children, list) or not children:
        raise ValueError("reference root_logic requires children")
    result = set()
    for child in children:
        if isinstance(child, str):
            result.add(child)
        else:
            result.update(_logic_ids(child))
    return result


def _validate_entry(entry: dict) -> None:
    if not isinstance(entry, dict):
        raise TypeError("reference entry must be an object")
    query = entry.get("query")
    if not isinstance(entry.get("query_id"), str) or not isinstance(query, str) or not query:
        raise ValueError("reference entry identity/query is missing")
    if entry.get("source_query_hash") != _query_hash(query):
        raise ValueError("reference source query hash mismatch")
    annotation = entry.get("annotation")
    if not isinstance(annotation, dict) or annotation.get("review_status") != "reviewed":
        raise ValueError("reference annotation is not independently reviewed")
    forbidden = _forbidden_annotation_paths(annotation)
    if forbidden:
        raise ValueError("forbidden annotation fields: " + ", ".join(forbidden))
    roles = annotation.get("roles")
    requirements = annotation.get("requirements")
    if not isinstance(roles, list) or not isinstance(requirements, list) or not requirements:
        raise ValueError("reference requires roles and atomic requirements")
    role_ids = {role.get("role_id") for role in roles if isinstance(role, dict)}
    if None in role_ids or len(role_ids) != len(roles):
        raise ValueError("reference roles must have unique IDs")
    requirement_ids = {item.get("requirement_id") for item in requirements if isinstance(item, dict)}
    if None in requirement_ids or len(requirement_ids) != len(requirements):
        raise ValueError("reference requirements must have unique IDs")
    for item in requirements:
        try:
            RequirementType(item.get("requirement_type"))
        except (ValueError, TypeError):
            raise ValueError("reference requirement_type is invalid") from None
        if item.get("optionality") not in {"mandatory", "optional"}:
            raise ValueError("reference optionality is invalid")
        span = item.get("source_text_span")
        if not isinstance(span, dict) or not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
            raise ValueError("reference source span is invalid")
        if query[span["start"]:span["end"]] != span.get("text"):
            raise ValueError("reference source span does not match query")
        if any(role_id not in role_ids for role_id in item.get("role_ids") or ()):
            raise ValueError("reference requirement uses unknown role")
        if any(dependency not in requirement_ids for dependency in item.get("depends_on") or ()):
            raise ValueError("reference requirement has unknown dependency")
    if _logic_ids(annotation.get("root_logic")) != requirement_ids:
        raise ValueError("reference Boolean structure must cover all requirements")


def freeze_reference(input_path: Path, paraphrase_path: Path, output_dir: Path) -> dict:
    reviewed = _load(input_path)
    if not isinstance(reviewed, dict) or reviewed.get("protocol_version") != REFERENCE_PROTOCOL:
        raise ValueError("unsupported reference protocol")
    if not isinstance(reviewed.get("annotator"), str) or not reviewed["annotator"].strip():
        raise ValueError("reference annotator metadata is required")
    entries = reviewed.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reviewed reference entries are required")
    for entry in entries:
        _validate_entry(entry)
    ids = [entry["query_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reviewed query_id")
    variations = _query_rows(_load(paraphrase_path))
    variation_ids = {row["query_id"] for row in variations}
    if variation_ids and not variation_ids <= set(ids):
        raise ValueError("reviewed reference omits predeclared paraphrase/holdout IDs")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reference_path = output / "requirement_reference.json"
    _write_exclusive(reference_path, reviewed)
    manifest = {
        "schema_version": "phase7_frozen_reference_manifest_v1",
        "protocol_version": REFERENCE_PROTOCOL,
        "annotation_version": reviewed.get("annotation_version"),
        "annotator": reviewed["annotator"],
        "entry_count": len(entries),
        "input_sha256": _sha256(input_path),
        "paraphrase_sha256": _sha256(paraphrase_path),
        "reference_sha256": _sha256(reference_path),
        "review_status": "reviewed_frozen",
    }
    _write_exclusive(output / "manifest.json", manifest)
    return manifest


def _requirement_matches(system, reference: dict) -> bool:
    if system.requirement_type.value != reference.get("requirement_type"):
        return False
    attributes = set(reference.get("attributes") or ())
    if attributes and not attributes <= set(system.attributes):
        return False
    predicate = reference.get("predicate")
    return not predicate or predicate == system.predicate


def evaluate_requirement_parse(system: QuestionRequirementSet, reference: dict) -> MetricResult:
    if not isinstance(system, QuestionRequirementSet):
        raise TypeError("system parse must be QuestionRequirementSet")
    annotation = reference.get("annotation") if isinstance(reference, dict) else None
    rows = annotation.get("requirements") if isinstance(annotation, dict) else None
    if not isinstance(rows, list):
        return MetricResult(None, None, 0, status="unavailable")
    mandatory = [row for row in rows if row.get("optionality") == "mandatory"]
    unknown = [row for row in mandatory if row.get("adjudication_status") == "unknown"]
    checkable = [row for row in mandatory if row not in unknown]
    available = list(system.requirements)
    matched = 0
    for reference_requirement in checkable:
        position = next((index for index, candidate in enumerate(available)
                         if _requirement_matches(candidate, reference_requirement)), None)
        if position is not None:
            matched += 1
            available.pop(position)
    value = matched / len(checkable) if checkable else None
    return MetricResult(
        value, matched if checkable else None, len(checkable), len(unknown),
        "partial" if unknown and checkable else "unknown" if unknown else "available",
        {"reference_mandatory_count": len(mandatory)},
    )


def _measurement(numerator: int | float | None, denominator: int, *, unknown: int = 0,
                 status: str | None = None, **details) -> dict[str, Any]:
    value = numerator / denominator if numerator is not None and denominator else None
    return {
        "value": value, "numerator": numerator, "denominator": denominator,
        "unknown_count": unknown,
        "status": status or ("available" if value is not None else "unavailable"),
        **details,
    }


def _reject_embedded_reference(trace: dict) -> None:
    forbidden = []
    def visit(value, path="trace"):
        if isinstance(value, dict):
            for key, item in value.items():
                if "reference_requirement" in str(key).casefold() or str(key).casefold() == "human_annotation":
                    forbidden.append(f"{path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
    visit(trace)
    if forbidden:
        raise ValueError("trace contains embedded reference requirements")


def compute_phase7_evidence_metrics(trace: dict, reference: dict) -> dict:
    if not isinstance(trace, dict) or not isinstance(reference, dict):
        raise TypeError("trace and reference must be objects")
    _reject_embedded_reference(trace)
    requirements = from_dict(QuestionRequirementSet, trace["requirements"])
    parse = evaluate_requirement_parse(requirements, reference)
    support = trace.get("support") or {}
    supported = set(support.get("supported_requirement_ids") or ())
    mandatory_system = [item for item in requirements.requirements if item.optionality is Optionality.MANDATORY]
    coverage_numerator = sum(item.requirement_id in supported for item in mandatory_system)
    visible = trace.get("visible") or {}
    tokens = visible.get("actual_message_tokens")
    pairs = {tuple(pair) for pair in visible.get("requirement_fact_pairs") or () if isinstance(pair, list) and len(pair) == 2}
    density = len(pairs) * 1000 / tokens if isinstance(tokens, int) and tokens > 0 else None
    semantic_keys = visible.get("fact_semantic_keys") or []
    duplicate_count = len(semantic_keys) - len(set(semantic_keys))
    consistency = trace.get("consistency") or {}
    role_checks = consistency.get("role_checks") or []
    role_known = [item for item in role_checks if item.get("consistent") in {True, False}]
    role_unknown = len(role_checks) - len(role_known)
    temporal_checks = consistency.get("temporal_checks") or []
    temporal_known = [item for item in temporal_checks if item.get("consistent") in {True, False}]
    temporal_unknown = len(temporal_checks) - len(temporal_known)
    strict_complete = (
        not support.get("missing_requirement_ids")
        and support.get("parse_complete") is True
        and support.get("role_consistent") is True
        and support.get("temporal_consistent") is True
        and support.get("provenance_complete") is True
    )
    claimed_complete = support.get("complete") is True or support.get("compatibility_status") == "COMPLETE"
    false_positive = int(claimed_complete and not strict_complete)
    return {
        "category": trace.get("category") or reference.get("category"),
        "requirement_parse_recall": parse.to_dict(),
        "question_requirement_coverage": _measurement(coverage_numerator, len(mandatory_system)),
        "semantic_density_per_1000_tokens": _measurement(len(pairs) * 1000 if tokens else None, tokens or 0),
        "duplicate_evidence_ratio": _measurement(duplicate_count, len(semantic_keys)),
        "cross_role_consistency": _measurement(sum(item["consistent"] is True for item in role_known), len(role_known), unknown=role_unknown,
                                                  status="partial" if role_unknown else None),
        "temporal_consistency": _measurement(sum(item["consistent"] is True for item in temporal_known), len(temporal_known), unknown=temporal_unknown,
                                               status="partial" if temporal_unknown else None),
        "complete_false_positive_rate": _measurement(false_positive, 1 if claimed_complete else 0),
        "visible_bundle_count": int(visible.get("bundle_count") or 0),
        "visible_fact_count": int(visible.get("fact_count") or 0),
        "actual_visible_tokens": tokens,
        "latency_seconds": trace.get("latency_seconds"),
        "model_calls": dict(trace.get("model_calls") or {}),
    }


def compute_phase7_answer_metrics(trace: dict, judgments: dict, reference: dict) -> dict:
    if not isinstance(trace, dict) or not isinstance(judgments, dict) or not isinstance(reference, dict):
        raise TypeError("trace, judgments, and reference must be objects")
    _reject_embedded_reference(trace)
    claims = judgments.get("claims")
    if not isinstance(claims, list):
        claims = []
    distribution = Counter({label: 0 for label in CLAIM_LABELS})
    malformed = 0
    supported_requirements = set()
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("verdict") not in CLAIM_LABELS:
            malformed += 1
            continue
        verdict = claim["verdict"]
        distribution[verdict] += 1
        if verdict == "SUPPORTED":
            supported_requirements.update(
                identifier for identifier in claim.get("requirement_ids") or () if isinstance(identifier, str)
            )
    judged = sum(distribution.values())
    requirements = from_dict(QuestionRequirementSet, trace["requirements"])
    mandatory = {
        item.requirement_id for item in requirements.requirements
        if item.optionality is Optionality.MANDATORY
    }
    covered = len(mandatory & supported_requirements)
    return {
        "category": trace.get("category") or reference.get("category"),
        "supported_claim_rate": _measurement(distribution["SUPPORTED"], judged, unknown=malformed,
                                               status="partial" if malformed else None),
        "answer_requirement_coverage": _measurement(covered, len(mandatory)),
        "claim_distribution": {label: distribution[label] for label in CLAIM_LABELS},
        "malformed_claim_count": malformed,
        "latency_seconds": trace.get("latency_seconds"),
        "model_calls": dict(trace.get("model_calls") or {}),
    }


def load_verified_evaluation_inputs(trace_path: Path, expected_trace_hash: str,
                                    reference_path: Path) -> tuple[dict, dict]:
    trace_path = Path(trace_path)
    if _sha256(trace_path) != expected_trace_hash:
        raise ValueError("trace hash mismatch")
    trace = _load(trace_path)
    if not isinstance(trace, dict):
        raise TypeError("trace must be an object")
    _reject_embedded_reference(trace)
    reference = _load(reference_path)
    if not isinstance(reference, dict):
        raise TypeError("reference must be an object")
    return trace, reference
