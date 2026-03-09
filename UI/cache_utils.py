from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def path_signature(path: Path) -> dict[str, Any]:
    resolved = str(path.resolve())
    if not path.exists():
        return {
            "path": resolved,
            "exists": False,
        }

    stat = path.stat()
    return {
        "path": resolved,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        return None
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)


def all_files_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() for path in paths)
