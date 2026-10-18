"""Content-addressed, evidence-only Phase 7 Warsaw-12 diagnostic runner."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from evique.phase7.config import Phase7Config
from evique.phase7.gates import verify_phase7_inputs
from evique.phase7.requirement_parser import parse_requirements
from evique.phase7.schema import to_dict
from evique.phase7.serializer import ANSWER_INSTRUCTION


MAIN_ARMS = ("CaptionRich-Hybrid", "Repaired-Full-EVIQUE", "Phase7-C")
CATEGORIES = ("direct", "relational_motion", "temporal_event")
FORBIDDEN_CONFIG_KEYS = {
    "reference_requirements", "reference_requirement", "annotation_path", "annotations",
    "ground_truth", "answer", "answers", "api_key", "openai_api_key", "base_url", "endpoint",
    "raw_frames", "raw_frame_bytes", "frame_payloads",
}
SHARED_FIELDS = (
    "scope_sha256", "embedding_index_sha256", "embedding_model_id",
    "caption_model_id", "caption_prompt_version", "answer_model_id",
    "answer_temperature", "answer_max_tokens", "answer_instruction_sha256",
    "context_limit_chars",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _implementation_hash() -> str:
    evique_root = Path(__file__).resolve().parents[1]
    files = sorted((evique_root / "phase7").glob("*.py")) + [
        Path(__file__).resolve(), evique_root / "experiments" / "phase7_metrics.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(evique_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _forbidden_paths(value: Any, path: str = "config") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_CONFIG_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_paths(item, f"{path}[{index}]"))
    return found


def _queries_and_categories(config: dict) -> tuple[list[dict], dict[str, str]]:
    query_document = _load(config["query_file"])
    queries = query_document.get("queries") if isinstance(query_document, dict) else None
    if not isinstance(queries, list) or len(queries) != 12:
        raise ValueError("Phase 7 requires exactly 12 frozen queries")
    for row in queries:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key]
                                                for key in ("query_id", "query", "video_id")):
            raise ValueError("malformed frozen query")
    ids = [row["query_id"] for row in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frozen query ID")
    if _sha256(Path(config["query_file"])) != config.get("query_file_sha256"):
        raise ValueError("frozen query hash mismatch")
    order_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if order_hash != config.get("query_order_sha256"):
        raise ValueError("frozen query order hash mismatch")
    if _sha256(Path(config["classification_file"])) != config.get("classification_file_sha256"):
        raise ValueError("classification rule hash mismatch")
    selection = _load(config["selection_manifest"])
    selected = selection.get("selected_queries") if isinstance(selection, dict) else None
    if not isinstance(selected, list) or [row.get("query_id") for row in selected] != ids:
        raise ValueError("selection manifest differs from frozen query order")
    if selection.get("selected_query_hash") not in {None, config["query_file_sha256"]}:
        raise ValueError("selection manifest query hash mismatch")
    categories = {row["query_id"]: row.get("category") for row in selected}
    counts = Counter(categories.values())
    if counts != Counter({category: 4 for category in CATEGORIES}):
        raise ValueError("Phase 7 requires exactly 4/4/4 frozen categories")
    return queries, categories


def validate_phase7_config(config: dict, *, repo_root: Path | None = None) -> dict:
    if not isinstance(config, dict):
        raise TypeError("Phase 7 runner config must be an object")
    forbidden = _forbidden_paths(config)
    if forbidden:
        raise ValueError("forbidden runtime config fields: " + ", ".join(forbidden))
    if config.get("protocol_version") != "phase7-semantic-evidence-v1":
        raise ValueError("unsupported Phase 7 protocol")
    if tuple(config.get("main_arms") or ()) != MAIN_ARMS:
        raise ValueError("main arms must be exactly CaptionRich-Hybrid, Repaired-Full-EVIQUE, Phase7-C")
    if config.get("context_limit_chars") != 12000:
        raise ValueError("context_limit_chars must equal 12000")
    if any(config.get(flag) is not False for flag in (
        "legacy_raw_packing", "legacy_graph_expansion", "legacy_cost_pruning", "visual_completion",
    )):
        raise ValueError("main A/B/C requires every legacy and visual flag OFF")
    if config.get("external_execution_enabled") is not False:
        raise ValueError("external execution must be disabled for evidence-only Phase 7")
    shared = config.get("shared_basis")
    if not isinstance(shared, dict) or any(shared.get(field) in (None, "") for field in SHARED_FIELDS):
        raise ValueError("shared basis fields are incomplete")
    if shared.get("context_limit_chars") != 12000:
        raise ValueError("shared basis context limit differs")
    expected_instruction = hashlib.sha256(ANSWER_INSTRUCTION.encode()).hexdigest()
    if shared.get("answer_instruction_sha256") != expected_instruction:
        raise ValueError("shared answer instruction hash differs")
    arm_basis = config.get("arm_basis")
    if not isinstance(arm_basis, dict) or set(arm_basis) != set(MAIN_ARMS):
        raise ValueError("arm shared basis declarations are incomplete")
    if any(arm_basis[arm] != shared for arm in MAIN_ARMS):
        raise ValueError("A/B/C shared basis differs")
    queries, categories = _queries_and_categories(config)
    gate = None
    if config.get("input_ledger"):
        root = Path(repo_root or Path.cwd()).resolve()
        phase_config = Phase7Config.from_dict(config)
        report = verify_phase7_inputs(phase_config, root, Path(config["input_ledger"]))
        if not report.valid:
            raise ValueError("Phase 7 immutable input gate failed")
        gate = {"valid": True, "checked_count": len(report.checked_hashes)}
    return {
        "valid": True, "query_count": len(queries),
        "category_counts": dict(sorted(Counter(categories.values()).items())),
        "main_arms": list(MAIN_ARMS), "shared_basis_equal": True,
        "external_execution_enabled": False, "input_gate": gate,
        "config_sha256": _canonical_hash(config),
    }


def validate_visual_ablation_config(config: dict) -> dict:
    value = config.get("visual_ablation") if isinstance(config, dict) else None
    if not isinstance(value, dict) or value.get("name") != "Phase7-C-Visual-OFF-ON" or value.get("enabled") is not True:
        raise ValueError("visual ablation must be separately named and enabled")
    off, on = value.get("off_calls_per_query"), value.get("on_calls_per_query")
    if type(off) is not int or type(on) is not int or off < 0 or off != on:
        raise ValueError("visual OFF/ON variants require an equal nonnegative per-query call budget")
    if not isinstance(value.get("authorization_identity"), str) or not value["authorization_identity"].strip():
        raise ValueError("visual ablation requires a non-empty authorization identity")
    return {"valid": True, "name": value["name"], "calls_per_query": off}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _synthetic_trace(query: dict, category: str, arm: str, config: dict) -> dict:
    parsed = parse_requirements(query["query"])
    if arm == "CaptionRich-Hybrid":
        provenance = "scope_keyword_dense"
        items = [{
            "evidence_id": f"{query['query_id']}:scope", "source_kind": "offline_scope_fixture",
            "video_id": query["video_id"], "text": f"Local synthetic Scope observation for {query['query_id']}.",
        }]
        supported = []
    elif arm == "Repaired-Full-EVIQUE":
        provenance = "retained_full_evique"
        items = [{
            "evidence_id": f"{query['query_id']}:scope", "source_kind": "shared_scope_fixture",
            "video_id": query["video_id"], "text": f"Local synthetic Scope observation for {query['query_id']}.",
        }, {
            "evidence_id": f"{query['query_id']}:legacy", "source_kind": "retained_metadata_fixture",
            "video_id": query["video_id"], "text": "Local structural metadata fixture.",
        }]
        supported = []
    else:
        provenance = "phase7_bundle_construction"
        items = [{
            "evidence_id": f"{query['query_id']}:bundle", "source_kind": "phase7_bundle_fixture",
            "video_id": query["video_id"], "text": "Local provenance-backed bundle fixture.",
        }]
        supported = []
    mandatory = [item.requirement_id for item in parsed.requirements
                 if item.optionality.value == "mandatory"]
    visible_text = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "phase7_evidence_trace_v1",
        "query_id": query["query_id"], "question": query["query"],
        "video_id": query["video_id"], "category": category, "arm": arm,
        "shared_basis": config["shared_basis"], "arm_provenance": provenance,
        "requirements": to_dict(parsed) if arm == "Phase7-C" else None,
        "visible_evidence": items,
        "support": {
            "supported_requirement_ids": supported,
            "missing_requirement_ids": mandatory if arm == "Phase7-C" else [],
            "compatibility_status": "INSUFFICIENT" if arm == "Phase7-C" else "NOT_ASSESSED",
        },
        "budget": {
            "context_limit_chars": 12000, "context_chars_visible": len(visible_text),
            "estimated_visible_tokens": (len(visible_text) + 3) // 4,
            "bundle_fits": True,
        },
        "artifact_status": "complete", "external_requests": 0,
        "raw_frame_requests": 0, "network_client_instantiated": False,
    }


def _validate_resume(output: Path, config: dict, mode: str) -> dict:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"output exists without a resumable manifest: {output}")
    manifest = _load(manifest_path)
    if manifest.get("config_sha256") != _canonical_hash(config) or manifest.get("mode") != mode:
        raise ValueError("resume manifest configuration differs")
    if manifest.get("implementation_sha256") != _implementation_hash():
        raise ValueError("resume manifest implementation differs")
    for relative, expected in (manifest.get("artifact_hashes") or {}).items():
        path = output / Path(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    summary = _load(output / "summary.json")
    return {**summary, "resume_status": "validated_existing_artifacts"}


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    output = []
    for arm in MAIN_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for category in ("overall", *CATEGORIES):
            selected = arm_rows if category == "overall" else [row for row in arm_rows if row["category"] == category]
            output.append({
                "arm": arm, "category": category, "trace_count": len(selected),
                "visible_item_count": sum(row["visible_item_count"] for row in selected),
                "supported_requirement_count": sum(row["supported_requirement_count"] for row in selected),
                "missing_requirement_count": sum(row["missing_requirement_count"] for row in selected),
                "actual_visible_tokens": "",
                "estimated_visible_tokens": sum(row["estimated_visible_tokens"] for row in selected),
                "metric_status": "structural_only_no_reference",
            })
    return output


def run_phase7_diagnostic(config: dict, output_dir: Path, *, mode: str = "evidence_only") -> dict:
    if mode not in {"evidence_only", "visual_ablation"}:
        raise ValueError("Task 13 permits only evidence_only or separately gated visual_ablation")
    if mode == "evidence_only":
        validation = validate_phase7_config(config)
    else:
        validation = validate_phase7_config({**config, "visual_completion": False})
        validate_visual_ablation_config(config)
    output = Path(output_dir)
    if output.exists():
        return _validate_resume(output, config, mode)
    if not config.get("fixture_mode"):
        raise RuntimeError("real Phase 7 execution requires the post-implementation materialized index protocol")
    output.mkdir(parents=True, exist_ok=False)
    trace_dir = output / "per_query_trace"
    trace_dir.mkdir()
    queries, categories = _queries_and_categories(config)
    rows = []
    artifact_paths = []
    for query in queries:
        for arm in MAIN_ARMS:
            trace = _synthetic_trace(query, categories[query["query_id"]], arm, config)
            path = trace_dir / f"{query['query_id']}__{_slug(arm)}.json"
            _write_json(path, trace)
            artifact_paths.append(path)
            rows.append({
                "query_id": query["query_id"], "category": categories[query["query_id"]],
                "arm": arm, "status": trace["artifact_status"],
                "visible_item_count": len(trace["visible_evidence"]),
                "supported_requirement_count": len(trace["support"]["supported_requirement_ids"]),
                "missing_requirement_count": len(trace["support"]["missing_requirement_ids"]),
                "estimated_visible_tokens": trace["budget"]["estimated_visible_tokens"],
                "trace_file": path.relative_to(output).as_posix(),
            })
    per_query = output / "per_query_metrics.csv"
    _write_csv(per_query, rows, list(rows[0]))
    artifact_paths.append(per_query)
    aggregate = _aggregate_rows(rows)
    metrics = output / "evidence_metrics.csv"
    _write_csv(metrics, aggregate, list(aggregate[0]))
    artifact_paths.append(metrics)
    summary = {
        "schema_version": "phase7_diagnostic_summary_v1",
        "status": "evidence_only_complete" if mode == "evidence_only" else "visual_ablation_structural_complete",
        "mode": mode, "trace_count": len(rows), "query_count": len(queries),
        "arms": list(MAIN_ARMS), "category_counts": validation["category_counts"],
        "external_request_count": 0, "raw_frame_request_count": 0,
        "headline_status": "not_generated_evidence_only",
        "hard_stop": "warsaw12", "pilot_provisional": True,
        "shared_basis_equal": True,
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    artifact_paths.append(summary_path)
    artifact_hashes = {
        path.relative_to(output).as_posix(): _sha256(path) for path in sorted(artifact_paths)
    }
    manifest = {
        "schema_version": "phase7_diagnostic_manifest_v1", "mode": mode,
        "config_sha256": _canonical_hash(config), "query_file_sha256": config["query_file_sha256"],
        "query_order_sha256": config["query_order_sha256"],
        "implementation_sha256": _implementation_hash(),
        "shared_basis": config["shared_basis"], "artifact_hashes": artifact_hashes,
        "external_request_count": 0, "trace_count": len(rows),
    }
    _write_json(output / "run_manifest.json", manifest)
    return summary
