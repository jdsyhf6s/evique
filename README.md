# EVIQUE — Anonymous Code Artifact

This package contains a single existing EVIQUE implementation snapshot:
model interfaces, evidence views, retrieval, planning, packing, and local
inference logic. It excludes Git history, author metadata, credentials,
datasets, experiment results, logs, model weights, and baseline repositories.
Original first-party source files and licensing are preserved.

## Main components

| Path | Function |
|---|---|
| evique/builder.py; evique/views/ | Scope, Target, Track, Event views |
| evique/standalone_base_builder.py; evique/solo_*.py | Local video, Caption and ASR interfaces |
| evique/visual_builder.py; evique/visual_compactor.py | Visual evidence |
| evique/planner.py; evique/cost_planner.py | Planning and evidence budgets |
| evique/retriever.py; evique/embedding_index.py; evique/candidate_fusion.py | Retrieval and rank fusion |
| evique/evidence_construction.py; evique/evidence_graph_executor.py | Structured evidence construction |
| evique/evidence_packer.py | Evidence packing |
| evique/phase7/pipeline.py | Typed requirement-driven pipeline |
| evique/phase7/local_completion.py | Frozen-index completion source |
| evique/phase7/temporal_relation_bridge.py; temporal_refinement.py | Temporal reasoning components |
| evique/phase7/serializer.py | Visible context serialization |
| evique/experiments/ | Core integration, manifest and protocol helpers |

## Installation

Use Python 3.10 or newer, then install the base requirements:

    python -m pip install -r requirements.txt

Dense retrieval additionally uses requirements-retrieval.txt.
Video processing uses requirements-vision.txt, optional model-specific libraries
and locally supplied weights. DEPENDENCIES.md lists external import roots.
No dependencies or model weights were installed during packaging.

## Synthetic example

From the root of this artifact:

    python scripts/build_views.py --segments-json examples/demo_segments.json --output-dir outputs/demo_index
    python scripts/run_evique.py --index-dir outputs/demo_index --query-file examples/minimal_query.json --no-hybrid

The example is synthetic input, not a benchmark result. These commands expose
the retrieval/planning/packing CLI. The advanced Phase7 interfaces are in
evique/phase7/ and evique/experiments/phase7_5_runner.py; they require compatible
input records and configuration. This snapshot is not certified as the exact
revision of any particular published experimental run.

Dense index options:

    python scripts/build_embedding_index.py --help

Provision encoder weights locally and use local-files-only options where
supported if downloads are unwanted. No credential file is included.
Do not publish generated outputs containing private paths or credentials.

## Verification

ANONYMITY_CHECK.json records static syntax, import-target and metadata checks.
MANIFEST.json and SHA256SUMS.txt describe the upload contents.
Real model inference, API calls and end-to-end reproduction were not performed.
Public dependencies and original license notices retain their proper names.
See LICENSE for licensing and third-party terms.
