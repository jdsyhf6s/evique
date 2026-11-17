# QVHighlights experiment helpers

This supplement connects the published EVIQUE retriever to the QVHighlights
moment-retrieval prediction format. It contains only the dataset-facing input
contract, prediction conversion, command-line launchers, and synthetic tests.
It reuses the existing `evique.retriever` implementation from the main code
release and does not duplicate Planner, retrieval, or evidence-packing code.
Baseline time-window normalization reuses `baselines.temporal_adapter` from the
main release; this supplement does not duplicate that adapter or any baseline.

The query runner strips annotation-only fields before retrieval. Evidence is
accepted only when it carries an explicit interval and, when present, a video
identity matching the query. It never selects a window using ground truth.

## Retrieval

Install the main EVIQUE package, prepare an EVIQUE index, and provide the
official query JSONL file locally:

```bash
python scripts/run_qvhighlights.py \
  --index-dir /path/to/index \
  --queries /path/to/queries.jsonl \
  --output /path/to/predictions.jsonl
```

The output follows the Moment-DETR submission fields `qid`, `query`, `vid`,
and `pred_relevant_windows`. Existing output files are never overwritten.

## Official evaluation

Obtain Moment-DETR from its official upstream repository and use its evaluator
and QVHighlights annotations under their respective licenses. They are not
redistributed here.

```bash
python scripts/evaluate_qvhighlights.py \
  --evaluator /path/to/moment_detr/standalone_eval/eval.py \
  --submission /path/to/predictions.jsonl \
  --ground-truth /path/to/validation.jsonl \
  --output /path/to/metrics.json
```

The wrapper validates the submission and forwards the documented evaluator
arguments. It does not implement or alter the official metric definitions.

## Scope

The full private experiment scheduler is intentionally not included because it
contains machine-specific orchestration and calls development-only entry
points. This stable helper preserves the formal inference boundary while using
the already published EVIQUE core. No dataset, annotation, prediction, result,
model weight, index, cache, or complete third-party baseline is included.
