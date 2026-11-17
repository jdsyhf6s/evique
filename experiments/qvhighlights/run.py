"""Stable QVHighlights retrieval entry point for EVIQUE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from evaluation.qvhighlights.adapter import query_to_prediction, validate_submission
from .input_contract import normalize_inference_row


Retrieve = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} must contain a JSON object")
            rows.append(value)
    return rows


def run_queries(rows: Iterable[Mapping[str, Any]], retrieve: Retrieve, *, top_k: int = 10) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for source_row in rows:
        inference = normalize_inference_row(source_row)
        question = inference["question"]
        metadata = {key: value for key, value in inference.items() if key != "question"}
        retrieval = retrieve(question, metadata)
        official_query = {
            "qid": source_row.get("qid", inference["query_id"]),
            "query": question,
            "vid": inference["video_id"],
        }
        if "duration" in inference:
            official_query["duration"] = inference["duration"]
        predictions.append(query_to_prediction(official_query, retrieval, top_k=top_k))
    validate_submission(predictions)
    return predictions


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EVIQUE retrieval on QVHighlights query records.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-evidence", type=int, default=18)
    parser.add_argument("--token-budget", type=int, default=12000)
    args = parser.parse_args(argv)

    from evique.retriever import EvidenceRetriever

    retriever = EvidenceRetriever(
        args.index_dir,
        max_evidence=args.max_evidence,
        token_budget=args.token_budget,
    )
    predictions = run_queries(load_jsonl(args.queries), retriever.retrieve, top_k=args.top_k)
    write_jsonl(args.output, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
