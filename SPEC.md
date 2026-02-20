# `vesicle_nrss` Repository Specification

## 1. Purpose
`vesicle_nrss` is an NRSS morphology/simulation framework for a **single vesicle** model, primarily for lipid vesicles, with flexibility for other shell-like systems.

The core design pattern is:
- define `VesicleArgs` dataclass
- build morphology from args
- run NRSS model
- return `VesicleResults` dataclass

## 2. Reference Implementations To Emulate
This specification intentionally mirrors behavior from the following repositories. These references are normative for design intent.

### 2.1 `fibril_models`
Emulate:
- dataclass-first model configuration (`arguments.py` style)
- default/preset argument constructor functions
- API split between `build_*_morph` and `run_*`
- `pyproject.toml` packaging style (`setuptools`, `setuptools-scm`, `src/` layout)
- deterministic filename/tag/suffix conventions for sweeps

### 2.2 `PBDF` (Ray implementation in `fibril_model02.py`)
Emulate:
- actor-based Ray sweep orchestration
- fractional-GPU actor variants (`1`, `0.5`, `0.33`, `0.25`, `0.2`, `0.1`)
- cluster/local init pathways
- greedy actor task assignment
- `ray.wait`/`ray.get` completion loop with progress reporting
- result ordering by original sweep index

### 2.3 `ray_glados`
Use as the preferred reusable Ray sweep backend.

Emulate/use:
- `run_generic_sweep_ray` orchestration pattern
- custom cluster configuration behavior
- actor class selection by `jobs_per_gpu`

Required integration note:
- `vesicle_nrss` should call `ray_glados` via a thin adapter layer to enforce this spec's filename, seed, ordering, and retry requirements.

### 2.4 `NRSS_tutorials/polymer_grafted_nanoparticles`
Emulate:
- radial Euler-angle calculation pattern using `arctan2`
- morphology construction flow for `Material` and `Morphology`

## 3. Design Constraints
- Keep OO minimal:
  - `VesicleArgs` dataclass
  - `Morphology` object
  - `VesicleResults` dataclass
- Two-material morphology only:
  - material 1: lipid
  - material 2: medium (typically water)
  - voxelwise: `vfrac_lipid + vfrac_medium = 1`
- No custom validation beyond `morph.validate_all(...)`.
- `theta`/`psi` are radial and computed globally (not wall-masked).
- `S` is global and flat for each material:
  - lipid: configurable `S_lipid`
  - medium: fixed `0` (not configurable)

## 4. Public API
```python
from vesicle_nrss import (
    VesicleArgs,
    VesicleResults,
    default_vesicle_args,
    small_test_args,
    highres_vesicle_args,
    build_vesicle_morph,
    run_vesicle,
    run_vesicle_sweep,
)
```

Required functions:
- `default_vesicle_args() -> VesicleArgs`
- `small_test_args() -> VesicleArgs`
- `highres_vesicle_args() -> VesicleArgs`
- `build_vesicle_morph(args: VesicleArgs) -> Morphology`
- `run_vesicle(args: VesicleArgs) -> VesicleResults`
- `run_vesicle_sweep(...)`

## 5. Dataclasses

### 5.1 `VesicleArgs`
- Geometry:
  - `vd: int`
  - `ld: int`
  - `PhysSize: float` (nm/voxel)
  - `radius_nm: float` (vesicle midplane radius)
  - `D_out_nm: float = 2.26` (half-thickness parameter)
  - `sigma_nm: float = 0.22` (interface width parameter)
- Orientation:
  - `S_lipid: float` (global constant over lipid material)
- Resolution:
  - `vfrac_supersample: int = 2` (applies only to `vfrac`)
  - `enforce_supersample_on_vfrac_only: bool = True`
- Backend:
  - `backend: Literal["numpy", "cupy"] = "numpy"`
- NRSS run inputs:
  - `energies: np.ndarray`
  - `EAngleRotation: list[float]`
  - `oc_lipid: NRSS.morphology.OpticalConstants`
  - `oc_medium: NRSS.morphology.OpticalConstants`
- Output naming:
  - `result_path: Path`
  - `filename: str`
  - `filename_tags: str`
  - `filename_suffix: str`
- Sweep/reproducibility:
  - `base_seed: int = 0`
- Ray behavior:
  - `ray_enable_serialization: bool = True`
  - `ray_retry_on_failure: bool = False`
  - `ray_max_retries: int = 2`
  - `ray_retry_backoff_s: float = 5.0`
- Cache controls (RAM-only):
  - `enable_radial_cache: bool = True`
  - `radial_cache_scope: Literal["thread_local", "process_local"] = "thread_local"`
  - `radial_cache_key_extra: str = ""`
 - Results payload controls:
  - `return_I_para_perp: bool = True`

### 5.2 `VesicleResults` (dataclass)
- `data`
- `remeshed_data`
- `I`
- `I_para`
- `I_perp`
- `A`
- `model_plots`
- `args`

`return_I_para_perp` behavior:
- when `True` (default), return `I_para` and `I_perp`, and set `A=None`
- when `False`, return `A`, and set `I_para=None`, `I_perp=None`

## 6. Units
All dimensional inputs in this repository are in **nm**.

Do not use Angstrom inputs in this API.
`D_out_nm` and `sigma_nm` are already nm-valued to avoid unit-conversion ambiguity.

## 7. Geometry and Volume Fraction Model

### 7.1 Centering
Use voxel-center coordinates:
- `cx = (ld - 1) / 2`
- `cy = (ld - 1) / 2`
- `cz = (vd - 1) / 2`

Distance field in nm:
- `dx_nm = (x - cx) * PhysSize`
- `dy_nm = (y - cy) * PhysSize`
- `dz_nm = (z - cz) * PhysSize`
- `r_nm = sqrt(dx_nm**2 + dy_nm**2 + dz_nm**2)`
- `rho_nm = r_nm - radius_nm`

### 7.2 Lipid wall profile (`volfrac.png` model)
Use explicit profile:
- `vfrac_lipid(rho_nm) = 0.5 * [erf((rho_nm + D_out_nm)/sigma_nm) - erf((rho_nm - D_out_nm)/sigma_nm)]`

Then:
- `vfrac_medium = 1 - vfrac_lipid`

Numerical cleanup:
- clamp only for floating-point roundoff
- enforce exact voxelwise sum-to-one after cleanup

Interpretation:
- wall full thickness is approximately `2 * D_out_nm`
- `radius_nm` is the bilayer midplane radius

## 8. Supersampling
- Supersample `vfrac` only.
- Default supersample multiplier is `2`.
- Use center-preserving subvoxel coordinates:
  - `x_sub = i + (k + 0.5)/m - 0.5` (same for `y_sub`, `z_sub`)
- Compute supersampled full-volume `vfrac` on `(vd*m, ld*m, ld*m)`.
- Downscale by box averaging using reshape/mean.
- Compute `theta`, `psi`, `S` at base resolution only.

## 9. Radial Euler and S Fields
Use radial formulas globally:
- `theta = arctan2(sqrt(dx_nm**2 + dy_nm**2), dz_nm)`
- `psi = arctan2(dy_nm, dx_nm)`

Center voxel convention:
- when `dx_nm = dy_nm = dz_nm = 0`: `theta = 0`, `psi = 0`

S fields:
- lipid: `S = S_lipid` everywhere
- medium: `S = 0` everywhere

## 10. Backend and Memory
- `backend="numpy"`: CPU path
- `backend="cupy"`: GPU path
- if CuPy unavailable and backend is `cupy`: hard error

GPU cleanup requirements:
- `del` large temporaries
- `cp.get_default_memory_pool().free_all_blocks()`
- `cp.get_default_pinned_memory_pool().free_all_blocks()` (if used)
- `gc.collect()`

## 11. Radial Cache Policy
- Cache is optional and **RAM-only on CPU**.
- No cache serialization to disk.
- No cache file I/O.
- No persistent GPU cache objects.
- In `cupy` mode, cached CPU arrays may be copied to GPU for active computation then freed.
- Per-thread/per-worker independent caches are acceptable.
- Cached artifacts are helper arrays only (distance/radial lookup), never simulation outputs.

## 12. Sweeps, Ray, and Serialization

### 12.1 Sweep naming and determinism
For sweep index `idx`:
- `seed = args.base_seed + idx`
- `base_filename = args.filename`
- `filename = filename_tags + base_filename + f"_{swept_arg}_{idx:03d}" + filename_suffix`

Always return/consume sweep outputs in sorted `idx` order.

### 12.2 Ray backend integration
Ray mode is optional via `parallel="ray"`.

Normative implementation path:
- use `ray_glados` library via a local adapter layer in `vesicle_nrss`

Hardcoded infra values (retain exactly):
- `RAY_HEAD_ADDR = "192.168.2.1"`
- `RAY_PORT = 6379`
- `RAY_TMPDIR = "/resdata/DeLongchamp/ray_temp"`

Custom cluster behavior to retain:
- `total_actors = 5 * jobs_per_gpu` in the cluster-oriented helper path
- fractional GPU actor classes: `1`, `0.5`, `0.33`, `0.25`, `0.2`, `0.1`

### 12.3 Serialization policy
- Non-Ray sweeps: serialization remains optional/decoupled.
- Ray sweeps: serialization is allowed and supported.
- If `args.ray_enable_serialization` is `True`, Ray tasks may write ordered pickle/lzma outputs with deterministic filenames.
- Serialization order must follow sorted `idx`.

### 12.4 Retry-on-failure policy
Optional, enabled by `args.ray_retry_on_failure`:
- on task failure, resubmit same `idx`/value up to `args.ray_max_retries`
- wait `args.ray_retry_backoff_s` between retries
- preserve deterministic filename and seed on retries

## 13. Repository Layout
```text
vesicle_nrss/
  pyproject.toml
  README.md
  LICENSE
  src/vesicle_nrss/
    __init__.py
    arguments.py          # VesicleArgs + default arg builders
    profile.py            # lipid/medium vfrac profile functions
    morphology.py         # build_vesicle_morph
    run.py                # run_vesicle
    sweep.py              # run_vesicle_sweep orchestration
    ray_adapter.py        # adapter between this repo and ray_glados
    backend.py            # numpy/cupy selection + cleanup helpers
    cache.py              # RAM-only CPU radial cache helpers
    results.py            # VesicleResults
    utils.py
  tests/
    test_centering.py
    test_vfrac_supersample.py
    test_orientation_radial.py
    test_backend_parity.py
    test_sweep_determinism.py
```

## 14. Packaging
Use `pyproject.toml` with `fibril_models`-style packaging:
- `setuptools` backend
- `setuptools-scm`
- `src/` package layout

Dependencies:
- required: `numpy`, `scipy`, `NRSS`, `PyHyperScattering`
- optional GPU: `cupy-cuda12x`
- optional Ray: `ray`, `ray_glados`

## 15. Acceptance Criteria
- API and dataclasses implemented as specified.
- Unit system is nm-only across public args.
- Explicit erf wall profile implemented exactly.
- Two-material sum-to-one guarantee enforced.
- Global radial Euler fields and center convention implemented.
- Medium S is fixed zero and not user-exposed.
- Supersampled `vfrac` path is full-volume and defaults to `2x`.
- Cache is CPU RAM-only with no serialization.
- Ray path uses `ray_glados` adapter and preserves deterministic naming/order/seed.
- Hardcoded Ray infra and custom actor scaling behavior retained.
- Optional retry-on-failure works as specified.
- Validation relies on `morph.validate_all(...)` only.
