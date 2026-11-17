"""Command-line launcher for the official QVHighlights evaluator adapter."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.qvhighlights.run_official import main


if __name__ == "__main__":
    raise SystemExit(main())
