"""Build a fresh dense index without changing the original evidence views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evique.embedding_index import DEFAULT_MODEL, EmbeddingIndex, FastEmbedEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Fresh destination; default: INDEX_DIR/embedding_index")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true", help="Use cached model files without downloading")
    args = parser.parse_args()
    output_dir = (args.output_dir or args.index_dir / "embedding_index").resolve()
    encoder = FastEmbedEncoder(model_id=args.model, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    index = EmbeddingIndex.build(args.index_dir, output_dir, encoder)
    print(json.dumps({"output_dir": str(output_dir), "manifest": index.manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
