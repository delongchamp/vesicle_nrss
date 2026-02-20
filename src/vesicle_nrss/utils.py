"""Utility helpers for deterministic sweep naming and serialization."""

from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any


def build_sweep_filename(
    filename_tags: str,
    base_filename: str,
    swept_arg: str,
    idx: int,
    filename_suffix: str,
) -> str:
    return f"{filename_tags}{base_filename}_{swept_arg}_{idx:03d}{filename_suffix}"


def ensure_result_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_lzma_pickle(path: Path, obj: Any) -> None:
    with lzma.open(path, "wb") as handle:
        pickle.dump(obj, handle)
