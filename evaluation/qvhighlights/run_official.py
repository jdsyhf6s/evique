"""Invoke a user-provided official QVHighlights evaluator without bundling it."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .adapter import validate_submission


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("submission must contain JSON objects")
                rows.append(value)
    return rows


def run_official_evaluator(
    evaluator_script: Path,
    submission_path: Path,
    ground_truth_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the official command-line evaluator with explicit, checked paths."""
    evaluator_script = Path(evaluator_script).resolve()
    submission_path = Path(submission_path).resolve()
    ground_truth_path = Path(ground_truth_path).resolve()
    output_path = Path(output_path).resolve()
    for label, path in (
        ("evaluator", evaluator_script),
        ("submission", submission_path),
        ("ground truth", ground_truth_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    validate_submission(_load_jsonl(submission_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(evaluator_script),
        "--submission_path",
        str(submission_path),
        "--gt_path",
        str(ground_truth_path),
        "--save_path",
        str(output_path),
        "--not_verbose",
    ]
    subprocess.run(command, check=True, shell=False)
    with output_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("official evaluator output must be a JSON object")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a user-provided official QVHighlights evaluator.")
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_official_evaluator(args.evaluator, args.submission, args.ground_truth, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
