"""Morphology construction for single- and multi-vesicle NRSS models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import truncnorm

from .arguments import RadiusSamplingMode, VesicleArgs, VesiclePopulationMode
from .backend import cleanup_gpu, get_array_module, get_erf, to_numpy
from .cache import get_radial_fields
from .profile import cleanup_two_material_vfrac, lipid_wall_profile


def _subvoxel_axis(length: int, supersample: int, xp):
    axis = xp.arange(length * supersample, dtype=xp.float32)
    return ((axis + 0.5) / supersample) - 0.5


def _compute_single_vfrac_supersampled(args: VesicleArgs, xp, erf_func):
    m = int(args.vfrac_supersample)
    if m <= 1:
        raise ValueError("Supersampling requires vfrac_supersample > 1")

    cz = (args.vd - 1) / 2.0
    cy = (args.ld - 1) / 2.0
    cx = (args.ld - 1) / 2.0

    z_sub = _subvoxel_axis(args.vd, m, xp)[:, None, None]
    y_sub = _subvoxel_axis(args.ld, m, xp)[None, :, None]
    x_sub = _subvoxel_axis(args.ld, m, xp)[None, None, :]

    dz_nm = (z_sub - cz) * args.PhysSize
    dy_nm = (y_sub - cy) * args.PhysSize
    dx_nm = (x_sub - cx) * args.PhysSize

    r_nm = xp.sqrt(dx_nm**2 + dy_nm**2 + dz_nm**2)
    rho_nm = r_nm - args.radius_nm

    vfrac_lipid_hi = lipid_wall_profile(rho_nm, args.D_out_nm, args.sigma_nm, erf_func)
    vfrac_lipid = vfrac_lipid_hi.reshape(
        args.vd,
        m,
        args.ld,
        m,
        args.ld,
        m,
    ).mean(axis=(1, 3, 5))
    return vfrac_lipid.astype(xp.float32)


def _single_mode_center_nm(args: VesicleArgs) -> np.ndarray:
    return np.array(
        [
            (((args.vd - 1) / 2.0) + 0.5) * args.PhysSize,
            (((args.ld - 1) / 2.0) + 0.5) * args.PhysSize,
            (((args.ld - 1) / 2.0) + 0.5) * args.PhysSize,
        ],
        dtype=np.float32,
    )


def _generate_single_vesicle_fields(args: VesicleArgs, xp, erf_func) -> dict[str, Any]:
    radial_cpu = get_radial_fields(
        vd=args.vd,
        ld=args.ld,
        PhysSize=args.PhysSize,
        radius_nm=args.radius_nm,
        enable_cache=args.enable_radial_cache,
        radial_cache_scope=args.radial_cache_scope,
        radial_cache_key_extra=args.radial_cache_key_extra,
    )

    if args.vfrac_supersample > 1:
        vfrac_lipid = _compute_single_vfrac_supersampled(args, xp=xp, erf_func=erf_func)
    else:
        rho = radial_cpu["rho_nm"] if args.backend == "numpy" else xp.asarray(radial_cpu["rho_nm"])
        vfrac_lipid = lipid_wall_profile(rho, args.D_out_nm, args.sigma_nm, erf_func)

    vfrac_lipid, vfrac_medium = cleanup_two_material_vfrac(vfrac_lipid, xp)

    theta = radial_cpu["theta"] if args.backend == "numpy" else xp.asarray(radial_cpu["theta"])
    psi = radial_cpu["psi"] if args.backend == "numpy" else xp.asarray(radial_cpu["psi"])

    S_lipid = xp.full((args.vd, args.ld, args.ld), args.S_lipid, dtype=xp.float32)
    S_medium = xp.zeros((args.vd, args.ld, args.ld), dtype=xp.float32)

    return {
        "vfrac_lipid": vfrac_lipid.astype(xp.float32),
        "vfrac_medium": vfrac_medium.astype(xp.float32),
        "theta": theta.astype(xp.float32),
        "psi": psi.astype(xp.float32),
        "S_lipid": S_lipid,
        "S_medium": S_medium,
        "accepted_radii_nm": np.array([args.radius_nm], dtype=np.float32),
        "accepted_centers_nm": _single_mode_center_nm(args)[None, :],
        "num_vesicles_requested": 1,
        "num_vesicles_placed": 1,
        "consecutive_failures_final": 0,
        "placement_max_failures": args.placement_max_failures,
    }


def _box_lengths_nm(args: VesicleArgs) -> np.ndarray:
    return np.array(
        [args.vd * args.PhysSize, args.ld * args.PhysSize, args.ld * args.PhysSize],
        dtype=np.float64,
    )


def _signed_minimum_image_delta_nm(delta_nm, box_len_nm: float, periodic: bool):
    if periodic:
        return ((delta_nm + 0.5 * box_len_nm) % box_len_nm) - 0.5 * box_len_nm
    return delta_nm


def _resolve_radius_list_nm(args: VesicleArgs, rng: np.random.Generator) -> np.ndarray:
    if args.radius_sampling_mode is RadiusSamplingMode.CONSTANT:
        return np.full(args.num_vesicles, args.radius_nm, dtype=np.float64)

    if args.radius_sampling_mode is RadiusSamplingMode.NORMAL:
        sigma = float(args.radius_sigma_nm)
        lower = max(10.0, args.radius_nm - 3.0 * sigma)
        upper = args.radius_nm + 3.0 * sigma
        a = (lower - args.radius_nm) / sigma
        b = (upper - args.radius_nm) / sigma
        return truncnorm(a, b, loc=args.radius_nm, scale=sigma).rvs(size=args.num_vesicles, random_state=rng)

    if args.radius_sampling_mode is RadiusSamplingMode.LIST:
        return np.asarray(args.radii_nm_list, dtype=np.float64)

    raise ValueError(f"Unsupported radius_sampling_mode: {args.radius_sampling_mode}")


def _candidate_collides(
    candidate_center_nm: np.ndarray,
    candidate_radius_nm: float,
    accepted_centers_nm: np.ndarray,
    accepted_radii_nm: np.ndarray,
    args: VesicleArgs,
    box_lengths_nm: np.ndarray,
) -> bool:
    if accepted_centers_nm.size == 0:
        return False

    deltas = np.abs(accepted_centers_nm - candidate_center_nm[None, :])
    for axis_idx, (periodic, axis_len_nm) in enumerate(zip(args.periodic_boundary_xyz, box_lengths_nm)):
        if periodic:
            deltas[:, axis_idx] = np.minimum(deltas[:, axis_idx], axis_len_nm - deltas[:, axis_idx])

    d_ij = np.sqrt(np.sum(deltas**2, axis=1))
    d_excl = candidate_radius_nm + accepted_radii_nm + (args.collision_buffer_sigma_multiplier * args.sigma_nm)
    return bool(np.any(d_ij < d_excl))


def _place_vesicles(
    args: VesicleArgs,
    radii_nm: np.ndarray,
    rng: np.random.Generator,
    box_lengths_nm: np.ndarray,
) -> dict[str, Any]:
    requested = int(len(radii_nm))
    accepted_orig_idx: list[int] = []
    accepted_radii: list[float] = []
    accepted_centers: list[np.ndarray] = []

    placement_order = np.argsort(-radii_nm, kind="stable")
    target_ptr = 0
    consecutive_failures = 0

    while target_ptr < requested and consecutive_failures < args.placement_max_failures:
        orig_idx = int(placement_order[target_ptr])
        candidate_radius = float(radii_nm[orig_idx])
        candidate_center = np.array(
            [
                rng.uniform(0.0, box_lengths_nm[0]),
                rng.uniform(0.0, box_lengths_nm[1]),
                rng.uniform(0.0, box_lengths_nm[2]),
            ],
            dtype=np.float64,
        )

        accepted_centers_array = (
            np.asarray(accepted_centers, dtype=np.float64).reshape(-1, 3)
            if accepted_centers
            else np.empty((0, 3), dtype=np.float64)
        )
        accepted_radii_array = (
            np.asarray(accepted_radii, dtype=np.float64)
            if accepted_radii
            else np.empty((0,), dtype=np.float64)
        )

        if _candidate_collides(
            candidate_center_nm=candidate_center,
            candidate_radius_nm=candidate_radius,
            accepted_centers_nm=accepted_centers_array,
            accepted_radii_nm=accepted_radii_array,
            args=args,
            box_lengths_nm=box_lengths_nm,
        ):
            consecutive_failures += 1
            continue

        accepted_orig_idx.append(orig_idx)
        accepted_radii.append(candidate_radius)
        accepted_centers.append(candidate_center)
        target_ptr += 1
        consecutive_failures = 0

    if accepted_orig_idx:
        order = np.argsort(np.asarray(accepted_orig_idx), kind="stable")
        accepted_orig = np.asarray(accepted_orig_idx, dtype=np.int64)[order]
        accepted_radii_nm = np.asarray(accepted_radii, dtype=np.float64)[order]
        accepted_centers_nm = np.asarray(accepted_centers, dtype=np.float64)[order]
    else:
        accepted_orig = np.empty((0,), dtype=np.int64)
        accepted_radii_nm = np.empty((0,), dtype=np.float64)
        accepted_centers_nm = np.empty((0, 3), dtype=np.float64)

    print(
        "[vesicle_nrss] placement_summary "
        f"num_vesicles_requested={requested} "
        f"num_vesicles_placed={accepted_radii_nm.size} "
        f"consecutive_failures_final={consecutive_failures} "
        f"placement_max_failures={args.placement_max_failures}"
    )

    return {
        "accepted_orig_indices": accepted_orig,
        "accepted_radii_nm": accepted_radii_nm.astype(np.float32),
        "accepted_centers_nm": accepted_centers_nm.astype(np.float32),
        "num_vesicles_requested": requested,
        "num_vesicles_placed": int(accepted_radii_nm.size),
        "consecutive_failures_final": int(consecutive_failures),
        "placement_max_failures": int(args.placement_max_failures),
    }


def _axis_index_window(center_nm: float, support_nm: float, phys_size_nm: float) -> np.ndarray:
    i_min = int(np.ceil((center_nm - support_nm) / phys_size_nm - 0.5))
    i_max = int(np.floor((center_nm + support_nm) / phys_size_nm - 0.5))
    if i_max < i_min:
        return np.empty((0,), dtype=np.int64)
    return np.arange(i_min, i_max + 1, dtype=np.int64)


def _wrap_or_clip_indices(indices: np.ndarray, size: int, periodic: bool, xp):
    if periodic:
        mapped = np.mod(indices, size)
    else:
        mapped = np.clip(indices, 0, size - 1)
    return xp.asarray(mapped.astype(np.int64))


def _generate_multiple_vesicle_fields(args: VesicleArgs, xp, erf_func) -> dict[str, Any]:
    if args.vfrac_supersample < 1:
        raise ValueError("vfrac_supersample must be >= 1")

    rng = np.random.default_rng(args.base_seed)
    box_lengths_nm = _box_lengths_nm(args)
    sampled_radii_nm = _resolve_radius_list_nm(args, rng=rng)
    placement = _place_vesicles(
        args=args,
        radii_nm=sampled_radii_nm,
        rng=rng,
        box_lengths_nm=box_lengths_nm,
    )

    placed_radii_nm = placement["accepted_radii_nm"]
    placed_centers_nm = placement["accepted_centers_nm"]
    m = int(args.vfrac_supersample)

    vfrac_lipid_total = xp.zeros((args.vd, args.ld, args.ld), dtype=xp.float32)
    theta_total = xp.zeros((args.vd, args.ld, args.ld), dtype=xp.float32)
    psi_total = xp.zeros((args.vd, args.ld, args.ld), dtype=xp.float32)

    sub_offsets = ((xp.arange(m, dtype=xp.float32) + 0.5) / m).astype(xp.float32)
    periodic_z, periodic_y, periodic_x = args.periodic_boundary_xyz
    Lz, Ly, Lx = box_lengths_nm

    for radius_nm, center_nm in zip(placed_radii_nm, placed_centers_nm):
        cz_nm, cy_nm, cx_nm = [float(v) for v in center_nm]
        support_nm = float(radius_nm + 3.0 * args.sigma_nm)

        z_indices = _axis_index_window(cz_nm, support_nm, args.PhysSize)
        y_indices = _axis_index_window(cy_nm, support_nm, args.PhysSize)
        x_indices = _axis_index_window(cx_nm, support_nm, args.PhysSize)
        if z_indices.size == 0 or y_indices.size == 0 or x_indices.size == 0:
            continue

        z_dest = _wrap_or_clip_indices(z_indices, args.vd, periodic_z, xp)
        y_dest = _wrap_or_clip_indices(y_indices, args.ld, periodic_y, xp)
        x_dest = _wrap_or_clip_indices(x_indices, args.ld, periodic_x, xp)
        indexer = (z_dest[:, None, None], y_dest[None, :, None], x_dest[None, None, :])

        z_idx = xp.asarray(z_indices.astype(np.float32))
        y_idx = xp.asarray(y_indices.astype(np.float32))
        x_idx = xp.asarray(x_indices.astype(np.float32))

        z_base_nm = (z_idx + 0.5) * args.PhysSize
        y_base_nm = (y_idx + 0.5) * args.PhysSize
        x_base_nm = (x_idx + 0.5) * args.PhysSize

        z_sub_nm = ((z_idx[:, None] + sub_offsets[None, :]) * args.PhysSize).reshape(-1)
        y_sub_nm = ((y_idx[:, None] + sub_offsets[None, :]) * args.PhysSize).reshape(-1)
        x_sub_nm = ((x_idx[:, None] + sub_offsets[None, :]) * args.PhysSize).reshape(-1)

        dz_sub_nm = _signed_minimum_image_delta_nm(
            z_sub_nm[:, None, None] - cz_nm, box_len_nm=Lz, periodic=periodic_z
        )
        dy_sub_nm = _signed_minimum_image_delta_nm(
            y_sub_nm[None, :, None] - cy_nm, box_len_nm=Ly, periodic=periodic_y
        )
        dx_sub_nm = _signed_minimum_image_delta_nm(
            x_sub_nm[None, None, :] - cx_nm, box_len_nm=Lx, periodic=periodic_x
        )

        r_sub_nm = xp.sqrt(dx_sub_nm**2 + dy_sub_nm**2 + dz_sub_nm**2)
        rho_sub_nm = r_sub_nm - float(radius_nm)
        vfrac_hi = lipid_wall_profile(rho_sub_nm, args.D_out_nm, args.sigma_nm, erf_func)
        vfrac_local = vfrac_hi.reshape(
            z_indices.size,
            m,
            y_indices.size,
            m,
            x_indices.size,
            m,
        ).mean(axis=(1, 3, 5))

        dz_base_nm = _signed_minimum_image_delta_nm(
            z_base_nm[:, None, None] - cz_nm, box_len_nm=Lz, periodic=periodic_z
        )
        dy_base_nm = _signed_minimum_image_delta_nm(
            y_base_nm[None, :, None] - cy_nm, box_len_nm=Ly, periodic=periodic_y
        )
        dx_base_nm = _signed_minimum_image_delta_nm(
            x_base_nm[None, None, :] - cx_nm, box_len_nm=Lx, periodic=periodic_x
        )

        distance_base_nm = xp.sqrt(dx_base_nm**2 + dy_base_nm**2 + dz_base_nm**2)
        support_mask = distance_base_nm <= support_nm
        vfrac_local = xp.where(support_mask, vfrac_local, xp.float32(0.0))
        xp.add.at(vfrac_lipid_total, indexer, vfrac_local.astype(xp.float32))

        radial_xy_nm = xp.sqrt(dx_base_nm**2 + dy_base_nm**2)
        theta_local = xp.arctan2(radial_xy_nm, dz_base_nm)
        psi_local = xp.arctan2(dy_base_nm, dx_base_nm)
        center_mask = (dx_base_nm == 0.0) & (dy_base_nm == 0.0) & (dz_base_nm == 0.0)
        theta_local = xp.where(center_mask, 0.0, theta_local)
        psi_local = xp.where(center_mask, 0.0, psi_local)

        theta_view = theta_total[indexer]
        psi_view = psi_total[indexer]
        theta_total[indexer] = xp.where(support_mask, theta_local.astype(xp.float32), theta_view)
        psi_total[indexer] = xp.where(support_mask, psi_local.astype(xp.float32), psi_view)

    vfrac_lipid_total, vfrac_medium = cleanup_two_material_vfrac(vfrac_lipid_total, xp)
    S_lipid = xp.full((args.vd, args.ld, args.ld), args.S_lipid, dtype=xp.float32)
    S_medium = xp.zeros((args.vd, args.ld, args.ld), dtype=xp.float32)

    return {
        "vfrac_lipid": vfrac_lipid_total.astype(xp.float32),
        "vfrac_medium": vfrac_medium.astype(xp.float32),
        "theta": theta_total.astype(xp.float32),
        "psi": psi_total.astype(xp.float32),
        "S_lipid": S_lipid,
        "S_medium": S_medium,
        "accepted_radii_nm": placed_radii_nm.astype(np.float32),
        "accepted_centers_nm": placed_centers_nm.astype(np.float32),
        "num_vesicles_requested": placement["num_vesicles_requested"],
        "num_vesicles_placed": placement["num_vesicles_placed"],
        "consecutive_failures_final": placement["consecutive_failures_final"],
        "placement_max_failures": placement["placement_max_failures"],
    }


def generate_vesicle_fields(args: VesicleArgs) -> dict[str, Any]:
    args.validate()
    xp = get_array_module(args.backend)
    erf_func = get_erf(args.backend)

    if args.population_mode is VesiclePopulationMode.SINGLE:
        return _generate_single_vesicle_fields(args, xp=xp, erf_func=erf_func)
    if args.population_mode is VesiclePopulationMode.MULTIPLE:
        return _generate_multiple_vesicle_fields(args, xp=xp, erf_func=erf_func)
    raise ValueError(f"Unsupported population_mode: {args.population_mode}")


def _extract_opt_constants(oc_obj):
    if oc_obj is None:
        return None
    if hasattr(oc_obj, "opt_constants"):
        return oc_obj.opt_constants
    return oc_obj


def build_vesicle_morph(args: VesicleArgs):
    if args.oc_lipid is None or args.oc_medium is None:
        raise ValueError("oc_lipid and oc_medium placeholders must be replaced before building morphology")

    fields = generate_vesicle_fields(args)

    try:
        from NRSS.morphology import Material, Morphology
    except Exception as exc:
        raise RuntimeError("NRSS.morphology is required to build the morphology") from exc

    mat_lipid = Material(
        materialID=1,
        Vfrac=to_numpy(fields["vfrac_lipid"]),
        S=to_numpy(fields["S_lipid"]),
        theta=to_numpy(fields["theta"]),
        psi=to_numpy(fields["psi"]),
        NumZYX=(args.vd, args.ld, args.ld),
        energies=args.energies,
        opt_constants=_extract_opt_constants(args.oc_lipid),
        name="lipid",
    )

    mat_medium = Material(
        materialID=2,
        Vfrac=to_numpy(fields["vfrac_medium"]),
        S=to_numpy(fields["S_medium"]),
        theta=to_numpy(fields["theta"]),
        psi=to_numpy(fields["psi"]),
        NumZYX=(args.vd, args.ld, args.ld),
        energies=args.energies,
        opt_constants=_extract_opt_constants(args.oc_medium),
        name="medium",
    )

    morph = Morphology(
        2,
        {
            1: mat_lipid,
            2: mat_medium,
        },
        PhysSize=args.PhysSize,
    )

    if hasattr(morph, "EAngleRotation"):
        morph.EAngleRotation = list(args.EAngleRotation)

    morph.accepted_radii_nm = np.asarray(fields["accepted_radii_nm"], dtype=np.float32)
    morph.accepted_centers_nm = np.asarray(fields["accepted_centers_nm"], dtype=np.float32)
    morph.num_vesicles_requested = int(fields["num_vesicles_requested"])
    morph.num_vesicles_placed = int(fields["num_vesicles_placed"])
    cleanup_gpu(args.backend)
    return morph
