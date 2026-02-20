"""Argument dataclass and preset constructors for vesicle runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class VesiclePopulationMode(Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"


class RadiusSamplingMode(Enum):
    CONSTANT = "constant"
    NORMAL = "normal"
    LIST = "list"


def _default_energies() -> np.ndarray:
    return np.concatenate(
        [
            np.array([260.0, 270.0, 280.0], dtype=float),
            np.arange(282.0, 295.0, 0.1, dtype=float),
            np.arange(295.0, 330.0, 1.0, dtype=float),
        ]
    )


@dataclass
class VesicleArgs:
    # Geometry
    vd: int = 512
    ld: int = 512
    PhysSize: float = 1.5
    radius_nm: float = 40.0
    population_mode: VesiclePopulationMode = VesiclePopulationMode.SINGLE
    num_vesicles: int = 1
    radius_sampling_mode: RadiusSamplingMode = RadiusSamplingMode.CONSTANT
    radius_sigma_nm: float | None = None
    radii_nm_list: list[float] | None = None
    periodic_boundary_xyz: tuple[bool, bool, bool] = (True, True, True)
    placement_max_failures: int = 10000
    collision_buffer_sigma_multiplier: float = 6.0
    D_out_nm: float = 2.26
    sigma_nm: float = 0.22

    # Orientation
    S_lipid: float = 1.0

    # Resolution
    vfrac_supersample: int = 2
    enforce_supersample_on_vfrac_only: bool = True

    # Backend
    backend: Literal["numpy", "cupy"] = "numpy"

    # NRSS run inputs
    energies: np.ndarray = field(default_factory=_default_energies)
    EAngleRotation: list[float] = field(default_factory=lambda: [0.0, 15.0, 165.0])
    oc_lipid: Any = None
    oc_medium: Any = None

    # Output naming
    result_path: Path = field(default_factory=lambda: Path("results"))
    filename: str = "vesicle"
    filename_tags: str = ""
    filename_suffix: str = ""

    # Sweep/reproducibility
    base_seed: int = 0

    # Ray behavior
    ray_enable_serialization: bool = True
    ray_retry_on_failure: bool = False
    ray_max_retries: int = 2
    ray_retry_backoff_s: float = 5.0

    # Cache controls (RAM-only)
    enable_radial_cache: bool = True
    radial_cache_scope: Literal["thread_local", "process_local"] = "thread_local"
    radial_cache_key_extra: str = ""

    # Results payload controls
    save_data_in_results: bool = False
    save_args_in_results: bool = True
    save_pickle_in_run: bool = True
    return_I_para_perp: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.periodic_boundary_xyz) != 3:
            raise ValueError("periodic_boundary_xyz must have exactly 3 elements")
        self.periodic_boundary_xyz = tuple(bool(v) for v in self.periodic_boundary_xyz)

        if self.radius_nm <= 0:
            raise ValueError("radius_nm must be > 0")

        if self.radius_sampling_mode is RadiusSamplingMode.LIST:
            if self.radii_nm_list is None:
                raise ValueError("radii_nm_list is required when radius_sampling_mode='list'")
            self.num_vesicles = len(self.radii_nm_list)

        if self.num_vesicles < 1:
            raise ValueError("num_vesicles must be >= 1")

        if self.radius_sampling_mode is RadiusSamplingMode.NORMAL:
            if self.radius_sigma_nm is None or self.radius_sigma_nm <= 0:
                raise ValueError("radius_sigma_nm must be > 0 when radius_sampling_mode='normal'")

        if self.radii_nm_list is not None:
            if any(r <= 0 for r in self.radii_nm_list):
                raise ValueError("all values in radii_nm_list must be strictly positive")

        if self.population_mode is VesiclePopulationMode.MULTIPLE:
            if self.periodic_boundary_xyz != (True, True, True):
                raise ValueError(
                    "MULTIPLE mode currently supports only periodic_boundary_xyz == (True, True, True)"
                )


def default_vesicle_args() -> VesicleArgs:
    return VesicleArgs()


def small_test_args() -> VesicleArgs:
    args = VesicleArgs(
        vd=32,
        ld=32,
        PhysSize=2.0,
        radius_nm=12.0,
        vfrac_supersample=1,
        filename="vesicle_small",
    )
    return args


def highres_vesicle_args() -> VesicleArgs:
    args = VesicleArgs(
        vd=768,
        ld=768,
        PhysSize=1.0,
        radius_nm=40.0,
        vfrac_supersample=2,
        filename="vesicle_highres",
    )
    return args
