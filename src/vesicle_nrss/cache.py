"""RAM-only CPU radial cache utilities."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


_THREAD_LOCAL = threading.local()
_PROCESS_LOCAL_CACHE: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}


def _get_thread_cache() -> dict[tuple[Any, ...], dict[str, np.ndarray]]:
    cache = getattr(_THREAD_LOCAL, "radial_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.radial_cache = cache
    return cache


def _compute_radial_fields(
    vd: int,
    ld: int,
    PhysSize: float,
    radius_nm: float,
) -> dict[str, np.ndarray]:
    cz = (vd - 1) / 2.0
    cy = (ld - 1) / 2.0
    cx = (ld - 1) / 2.0

    z = np.arange(vd, dtype=np.float32)[:, None, None]
    y = np.arange(ld, dtype=np.float32)[None, :, None]
    x = np.arange(ld, dtype=np.float32)[None, None, :]

    dz_nm = (z - cz) * PhysSize
    dy_nm = (y - cy) * PhysSize
    dx_nm = (x - cx) * PhysSize

    radial_xy_nm = np.sqrt(dx_nm**2 + dy_nm**2)
    r_nm = np.sqrt(radial_xy_nm**2 + dz_nm**2)
    rho_nm = (r_nm - radius_nm).astype(np.float32)

    theta = np.arctan2(radial_xy_nm, dz_nm)
    psi = np.broadcast_to(np.arctan2(dy_nm, dx_nm), (vd, ld, ld))

    center_mask = (r_nm == 0.0)
    if np.any(center_mask):
        theta = np.where(center_mask, 0.0, theta)
        psi = np.where(center_mask, 0.0, psi)

    return {
        "rho_nm": rho_nm,
        "theta": theta.astype(np.float32),
        "psi": psi.astype(np.float32),
    }


def get_radial_fields(
    vd: int,
    ld: int,
    PhysSize: float,
    radius_nm: float,
    enable_cache: bool = True,
    radial_cache_scope: str = "thread_local",
    radial_cache_key_extra: str = "",
) -> dict[str, np.ndarray]:
    key = (vd, ld, float(PhysSize), float(radius_nm), radial_cache_key_extra)

    if not enable_cache:
        return _compute_radial_fields(vd=vd, ld=ld, PhysSize=PhysSize, radius_nm=radius_nm)

    if radial_cache_scope == "thread_local":
        cache = _get_thread_cache()
    elif radial_cache_scope == "process_local":
        cache = _PROCESS_LOCAL_CACHE
    else:
        raise ValueError("radial_cache_scope must be 'thread_local' or 'process_local'")

    if key not in cache:
        cache[key] = _compute_radial_fields(vd=vd, ld=ld, PhysSize=PhysSize, radius_nm=radius_nm)

    return cache[key]
