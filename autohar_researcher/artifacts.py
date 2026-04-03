from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def ensure_round_dir(round_id: int, base_dir: Path | None = None) -> Path:
    root = base_dir or Path.cwd()
    round_dir = root / "artifacts" / f"round_{round_id}"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def write_json_artifact(path: Path, payload: object) -> str:
    if is_dataclass(payload):
        serializable = asdict(payload)
    else:
        serializable = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
