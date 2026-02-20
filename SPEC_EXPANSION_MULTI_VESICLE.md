# `vesicle_nrss` Morphology Expansion Spec (Draft v0)

## 1. Goal
Define an additive morphology-generation expansion for a new repository, `vesicle_nrss`, that extends `vesicle_nrss` to support multiple vesicles in one simulation volume.

This draft is intentionally scoped to morphology generation only.

## 2. Compatibility and Scope
- Preserve existing single-vesicle behavior by default.
- Do not refactor stable non-morphology pathways from `vesicle_nrss` (run orchestration, serialization, sweep naming, Ray policy, result schema).
- In `SINGLE` mode, behavior must match current `vesicle_nrss` numerically (within existing tolerances).

## 3. New Mode Controls

### 3.1 Population mode enum
```python
class VesiclePopulationMode(Enum):
    SINGLE = "single"
    MULTIPLE = "multiple"
```

- `SINGLE`: delegates to existing single-vesicle morphology path.
- `MULTIPLE`: uses new multi-vesicle morphology generation path.

### 3.2 Radius sampling enum
```python
class RadiusSamplingMode(Enum):
    CONSTANT = "constant"
    NORMAL = "normal"
    LIST = "list"
```

- `CONSTANT`:
  - every vesicle uses `radius_nm`
- `NORMAL`:
  - sample radii from a truncated normal
  - mean = `radius_nm`
  - sigma = `radius_sigma_nm` (required)
  - `radius_sigma_nm` is distribution spread only; it is distinct from wall-profile `sigma_nm`
  - lower bound = `max(10.0, radius_nm - 3 * radius_sigma_nm)`
  - upper bound = `radius_nm + 3 * radius_sigma_nm`
- `LIST`:
  - use `radii_nm_list` directly
  - default `None`; required when mode is `LIST`
  - list length is authoritative in `LIST` mode
  - internal placement may reorder by radius, but reported/stored outputs must map back to original list order

## 4. Additive Argument Extensions

Add to the current `VesicleArgs` contract in `vesicle_nrss`:
- `population_mode: VesiclePopulationMode = VesiclePopulationMode.SINGLE`
- `num_vesicles: int = 1`
- `radius_sampling_mode: RadiusSamplingMode = RadiusSamplingMode.CONSTANT`
- `radius_sigma_nm: float | None = None`
- `radii_nm_list: list[float] | None = None`
- `periodic_boundary_xyz: tuple[bool, bool, bool] = (True, True, True)`
- `placement_max_failures: int = 10000`
- `collision_buffer_sigma_multiplier: float = 6.0`

Validation requirements:
- `num_vesicles >= 1`
- `radius_nm > 0`
- `radius_sigma_nm > 0` when `NORMAL`
- `radii_nm_list` present when `LIST`
- all radii strictly positive
- in `LIST` mode: set `num_vesicles = len(radii_nm_list)` (list length is source of truth)
- in `MULTIPLE` mode for v0, require `periodic_boundary_xyz == (True, True, True)`; other tuples are unsupported

## 5. Morphology Generation Behavior

### 5.1 `SINGLE` mode
- Use the current single-vesicle code path unchanged.
- Continue to allow radial cache behavior exactly as implemented now.

### 5.2 `MULTIPLE` mode high-level flow
1. Resolve radius list `r_i` from `radius_sampling_mode`.
2. Sample center list `c_i` uniformly in the periodic box (v0 supports sampled centers only).
3. Validate non-collision constraints under periodic boundaries.
4. Build vesicle wall fields and composite into master volume.
5. Build per-vesicle orientation fields and graft into master orientation arrays.
6. Apply cleanup and exact two-material closure (`vfrac_lipid + vfrac_medium = 1`).
7. Print placement summary to stdout including requested and placed counts.

### 5.3 Coordinate convention for `MULTIPLE`
Use physical nm coordinates on a periodic box:
- box lengths: `Lz = vd * PhysSize`, `Ly = ld * PhysSize`, `Lx = ld * PhysSize`
- base voxel-center coordinates:
  - `z_i = (i + 0.5) * PhysSize`
  - `y_j = (j + 0.5) * PhysSize`
  - `x_k = (k + 0.5) * PhysSize`
- sampled center coordinates:
  - `c_z ~ U[0, Lz)`, `c_y ~ U[0, Ly)`, `c_x ~ U[0, Lx)`

For signed minimum-image deltas (needed for Euler angle direction):
- `delta_signed = ((voxel_nm - center_nm + 0.5 * L) % L) - 0.5 * L`

For scalar distance checks:
- `d = sqrt(delta_z^2 + delta_y^2 + delta_x^2)` using minimum-image deltas.

RNG requirement for reproducibility:
- use a deterministic RNG initialized from `base_seed` (for example `np.random.default_rng(base_seed)`)
- the same seeded RNG stream must govern both radius sampling and center sampling in `MULTIPLE`
- for identical args/seed/backend, sampled radii and accepted centers must be reproducible

## 6. Collision Rule
Define effective exclusion distance for pair `(i, j)`:
- `d_excl(i, j) = r_i + r_j + collision_buffer_sigma_multiplier * sigma_nm`
- `sigma_nm` here is the wall-profile sigma used in the `erf` wall model, not `radius_sigma_nm`
- default `collision_buffer_sigma_multiplier = 6.0`, equivalent to `3*sigma_nm` clearance from each vesicle wall support boundary

Distance must use minimum-image periodic metric:
- `Lz = vd * PhysSize`, `Ly = ld * PhysSize`, `Lx = ld * PhysSize`
- for each axis:
  - `delta = abs(c_i - c_j)`
  - if periodic on axis: `delta = min(delta, L - delta)`
- `d_ij = sqrt(delta_z^2 + delta_y^2 + delta_x^2)`

Collision predicate:
- `collision = (d_ij < d_excl(i, j))`

## 7. Center Placement Strategy
Default approach for auto-placement:
- rejection sampling with sequential failure limit
- sample candidate centers as continuous `float` coordinates in physical nm space
- evaluate one candidate at a time
- for each candidate, compute collision checks against the full vector of accepted centers/radii using broadcasted pairwise distances

Algorithm:
1. build `(orig_idx, radius)` list and order by descending radius (largest first)
2. for current target radius, sample candidate center uniformly in box `[0, Lz) x [0, Ly) x [0, Lx)`
3. check candidate against already placed centers using periodic collision rule (broadcast over accepted centers)
4. accept candidate if non-colliding, store against its `orig_idx`, and move to next target radius
5. continue until either all targets are placed or consecutive failures exceed `placement_max_failures`

Failure behavior:
- rejected candidates are expected and handled silently during placement
- `placement_max_failures` is interpreted as a consecutive-failure threshold
- on each accepted placement, the consecutive-failure counter resets to `0`
- reaching `placement_max_failures` before placing all requested vesicles is a successful run with partial placement
- implementation must report `num_vesicles_requested` and `num_vesicles_placed` to stdout
- stdout report must also include `consecutive_failures_final` and `placement_max_failures`
- placement ends when either requested count is met or consecutive-failure threshold is reached
- reported/stored centers and radii must be in original index order (not working placement order)

Poisson-disc note:
- not default for variable radii (`NORMAL`/`LIST`) due implementation and performance complexity.

## 8. Supersampling With Arbitrary Float Centers
The current full-volume supersampling path is not reused for `MULTIPLE`.

For each vesicle `i`:
1. compute support radius
   - `r_support_i = r_i + 3 * sigma_nm`
2. construct local base-grid index window that covers support in each axis
   - `i_min = ceil((c_axis - r_support_i) / PhysSize - 0.5)`
   - `i_max = floor((c_axis + r_support_i) / PhysSize - 0.5)`
   - wrap destination indices by modulo for periodic axes
3. evaluate profile at supersampled local coordinates using the actual float center
   - for base index `i` and sub-index `k in [0, m-1]`:
   - `axis_sub_nm = (i + (k + 0.5)/m) * PhysSize`
4. downsample local supersampled wall by box averaging to base resolution (same `m` as existing `vfrac_supersample`)
5. apply support mask criterion in base resolution
   - include contribution only where minimum-image distance to vesicle center satisfies `d_i <= r_i + 3*sigma_nm`
   - set contribution to `0` outside this criterion
6. composite masked local result into global master volume with strict index bookkeeping and periodic wrapping

Compositing requirement:
- use additive composition into `vfrac_lipid_total` for masked contributions only
- because supports are non-overlapping by collision construction, this should prevent wall-tail accumulation from many vesicles
- apply only minimal floating-point cleanup (roundoff-level clipping) before final closure

Indexing requirements:
- global destination indices must be derived from physical coordinates and wrapped by modulo on periodic axes
- non-periodic axis behavior (if later enabled) must clip rather than wrap
- include explicit tests for edge-touching vesicles and wrapped insertion correctness

## 9. Orientation Fields in `MULTIPLE`
`theta` and `psi` are computed separately for each vesicle using that vesicle's own center and then grafted into master orientation arrays.

Grafting criterion for vesicle `i`:
- write `theta_i`, `psi_i` only where `d_i <= r_i + 3*sigma_nm` (minimum-image distance)
- write nothing outside this support mask
- this matches the vfrac support mask and avoids adjacent-vesicle overwrite under valid non-collision placement

- lipid material: `S = S_lipid` (same scalar as existing behavior)
- medium material: `S = 0`
- Euler formula per vesicle (inside support mask):
  - `theta_i = arctan2(sqrt(dx_i^2 + dy_i^2), dz_i)`
  - `psi_i = arctan2(dy_i, dx_i)`
  - where `dx_i`, `dy_i`, `dz_i` are signed minimum-image deltas to center `i`
- center singularity convention: when `dx_i = dy_i = dz_i = 0`, set `theta_i = 0`, `psi_i = 0`
- outside all support masks: explicitly set `theta = 0`, `psi = 0`

This keeps orientation physically local to each vesicle without changing the existing material model.

## 10. Periodic Boundary Conditions
For this expansion, PBC is enabled across all three dimensions in `MULTIPLE` mode.

`SINGLE` mode remains legacy behavior (no new PBC behavior added).
- v0 supports only full 3D periodicity in `MULTIPLE`; partial periodic tuples are out-of-scope and should raise validation error

PBC must be applied consistently in:
- center placement and collision checks
- distance evaluation during local vesicle field creation near boundaries
- subvolume-to-global composition/wrapping
- per-vesicle orientation mask/grafting

## 11. Caching Policy
- Keep current radial cache behavior for `SINGLE` mode.
- In `MULTIPLE`, disable/restrict reuse of single-center radial cache because centers are float and per-vesicle.
- Optional future optimization: cache per-radius local kernels centered at origin, then shift/wrap during insertion.

## 12. Tests Required for Expansion
Add tests in `vesicle_nrss` for:
- mode parity: `SINGLE` matches legacy outputs
- radius sampling:
  - `CONSTANT` deterministic
  - `NORMAL` truncation bounds and reproducibility from seed
  - `LIST` length drives `num_vesicles`
- collision logic under PBC (including across wrapped boundaries)
- placement termination behavior:
  - success when requested count is met
  - success with partial placement when failure threshold is hit first
  - reported requested vs placed counts are correct
  - consecutive-failure counter resets on accept
- float-center supersampling insertion correctness at boundary and near-corner wrap
- masked-addition rule: no contributions beyond `r_i + 3*sigma_nm`
- per-vesicle Euler grafting rule: write only within support mask; no overwrite under non-collision constraints
- Euler defaults: `theta=0`, `psi=0` outside all support masks
- `LIST` mode identity: internal reorder allowed, but reported/stored results remain in original list order
- two-material closure (`vfrac_lipid + vfrac_medium == 1` within tolerance)
- deterministic reproducibility from seed for sampled radii and centers

## 13. Non-Goals (This Iteration)
- no changes to NRSS run orchestration
- no changes to sweep APIs except additive args required for morphology
- no refactor of existing single-vesicle code paths that already work
- no Poisson-disc implementation for variable-radius placement in v0

## 14. Locked v0 Decisions

### 14.1 Resolved choices
1. v0 excludes user-provided centers and supports sampled centers only.
2. `LIST` mode makes list length authoritative for vesicle count.
3. Collisions, vfrac support masks, and Euler graft masks all depend on wall-profile `sigma_nm` (not `radius_sigma_nm`).
4. Candidate placement failures are silent; terminal failure threshold produces partial-placement success with reporting.
5. Centers are sampled as continuous physical coordinates in nm.
6. Placement may reorder vesicle processing (largest-first) for better packing.
7. Outside all support masks, orientation arrays are explicitly `theta=0`, `psi=0`.
8. In `LIST` mode, reported outputs must map back to original user list order.
9. Placement reporting target for v0 is stdout.
10. Placement failure threshold is consecutive, not total.
11. v0 placement is single-candidate sequential; vectorization applies only to candidate-vs-accepted collision checks.
