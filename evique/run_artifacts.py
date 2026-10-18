"""Create isolated, non-overwriting directories for retrieval runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RunSession:
    run_id: str
    run_dir: Path

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        config: dict[str, Any],
        index_version: str,
        model_version: str,
        run_id: str | None = None,
    ) -> "RunSession":
        now = datetime.now(timezone.utc)
        if run_id is None:
            run_id = f"run_{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"
        elif not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
            raise ValueError("run_id must contain only ASCII letters, digits, underscores or hyphens")
        elif not run_id.startswith("run_"):
            run_id = f"run_{run_id}"

        manifest = {
            "config": config,
            "run_id": run_id,
            "index_version": index_version,
            "model_version": model_version,
            "timestamp": now.isoformat(),
        }
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n"
        root = Path(root).resolve()
        run_dir = root / run_id
        if run_dir.resolve().parent != root:
            raise ValueError("run directory must remain inside the results root")
        root.mkdir(parents=True, exist_ok=True)
        # Atomic reservation: never reuse an existing run.
        run_dir.mkdir(exist_ok=False)
        with (run_dir / "config.json").open("x", encoding="utf-8") as handle:
            handle.write(payload)
        return cls(run_id=run_id, run_dir=run_dir)


def index_version(index_dir: Path) -> str:
    """Read the declared index version without constructing or changing it."""
    manifest_path = Path(index_dir) / "index_manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("index_version") or manifest.get("evique_version") or "unknown")
