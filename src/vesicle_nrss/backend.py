"""Backend selection and cleanup helpers for numpy/cupy execution."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - cupy is optional
    cp = None


def get_array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "cupy":
        if cp is None:
            raise RuntimeError("backend='cupy' requested but CuPy is unavailable")
        return cp
    raise ValueError(f"Unsupported backend: {backend}")


def get_erf(backend: str):
    if backend == "numpy":
        from scipy.special import erf

        return erf

    if backend == "cupy":
        if cp is None:
            raise RuntimeError("backend='cupy' requested but CuPy is unavailable")
        from cupyx.scipy.special import erf

        return erf

    raise ValueError(f"Unsupported backend: {backend}")


def seed_backend(seed: int, backend: str) -> None:
    np.random.seed(seed)
    if backend == "cupy" and cp is not None:
        cp.random.seed(seed)


def to_numpy(value: Any):
    if cp is not None and isinstance(value, cp.ndarray):  # pragma: no cover - cupy optional
        return cp.asnumpy(value)
    return np.asarray(value)


def cleanup_gpu(backend: str) -> None:
    if backend != "cupy" or cp is None:
        gc.collect()
        return

    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()
