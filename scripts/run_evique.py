from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evique.retrieval import EvidenceRetriever
from evique.run_artifacts import RunSession, index_version
from evique.utils.query_io import QueryRecord, load_query_file
from evique.video_identity import EVIQUE_VERSION_LABEL
from evique.phase4_trace import finalize_phase4_package, write_phase4_trace, synchronize_debug_trace


def parse_args():
    p = argparse.ArgumentParser(description="Run EVIQUE retrieval/planning/packing for a query.")
    p.add_argument("--index-dir", type=Path, default=Path("outputs/demo_index"))
    p.add_argument("--query", default="")
    p.add_argument("--query-file", type=Path)
    p.add_argument("--output", type=Path, help="Filename/parent hint; saved inside a new run directory.")
    p.add_argument("--results-dir", type=Path, help="Run root (default: results, or --output's parent).")
    p.add_argument("--run-id", help="Unique run identifier; existing runs are never overwritten.")
    p.add_argument("--debug", action="store_true", help="Write query traces inside this run.")
    p.add_argument("--max-evidence", type=int, default=18)
    p.add_argument("--token-budget", type=int, default=12000)
    p.add_argument("--embedding-index-dir", type=Path, help="Dense sidecar index (default: INDEX_DIR/embedding_index if present).")
    p.add_argument("--no-hybrid", action="store_true", help="Use the preserved Phase 1 retrieval path for comparison.")
    p.add_argument("--no-construction", action="store_true", help="Use Phase 2 retrieval and packing without graph construction.")
    return p.parse_args()


def queries(a):
    if a.query_file:
        return load_query_file(a.query_file).queries
    if a.query:
        return [QueryRecord("adhoc_001", "adhoc", "", a.query)]
    raise SystemExit("provide --query or --query-file")


def main():
    a = parse_args()
    query_records = queries(a)
    results_root = a.results_dir or (a.output.parent if a.output else Path("results"))
    output_name = a.output.name if a.output else "evique_result.json"
    if output_name.casefold() == "config.json":
        raise SystemExit("--output filename config.json is reserved for the run manifest")
    run = RunSession.create(
        results_root,
        config={**vars(a), "execution_mode": "retrieval_only", "output_filename": output_name,
                "generation_model": "not_used", "model_version_kind": "retrieval_implementation"},
        index_version=index_version(a.index_dir), model_version=EVIQUE_VERSION_LABEL, run_id=a.run_id,
    )
    retriever = EvidenceRetriever(
        a.index_dir, max_evidence=a.max_evidence, token_budget=a.token_budget,
        debug_dir=run.run_dir / "debug" if a.debug else None,
        embedding_index_dir=a.embedding_index_dir, hybrid_enabled=not a.no_hybrid,
        construction_enabled=not a.no_construction,
    )
    rows = []
    for query_index, q in enumerate(query_records):
        metadata = {**q.metadata, "query_id": q.query_id, "video_id": q.video_id, "dataset": q.dataset}
        package = retriever.retrieve(q.query, metadata)
        package = finalize_phase4_package(package)
        if package.get('phase4'):
            trace_path = run.run_dir / 'phase4_trace' / f'query_{query_index:04d}.json'
            write_phase4_trace(package, trace_path, query_metadata=metadata)
            package['phase4_trace_path'] = str(trace_path)
        context = retriever.format_package(package)
        synchronize_debug_trace(package, context)
        rows.append({"query": q.to_dict(), "method": "EVIQUE", "evidence_package": package, "formatted_context": context})
    output = run.run_dir / output_name
    with output.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
