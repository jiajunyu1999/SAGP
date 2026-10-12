"""Small IO/config helpers shared by the stage CLIs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_config(value: str) -> dict[str, Any]:
    with config_path(value).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_root(config: dict[str, Any]) -> Path:
    raw = config.get("project_root")
    return Path(raw).expanduser().resolve() if raw else ROOT


def ensure_dirs(root: Path) -> None:
    for name in ("data/raw", "data/interim", "data/processed", "data/splits",
                 "outputs/audit", "outputs/manifests"):
        (root / name).mkdir(parents=True, exist_ok=True)


def db_path(root: Path) -> Path:
    configured = os.environ.get("CHEMBL37_SQLITE")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [root / "data/raw/chembl_37.db",
                  root / "data/raw/chembl_37/chembl_37_sqlite/chembl_37.db"]
    return next((path for path in candidates if path.exists()), candidates[0])


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for parquet outputs") from exc
    frame.to_csv(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
