"""Volume-fraction profile helpers."""

from __future__ import annotations


def lipid_wall_profile(rho_nm, D_out_nm: float, sigma_nm: float, erf_func):
    return 0.5 * (
        erf_func((rho_nm + D_out_nm) / sigma_nm)
        - erf_func((rho_nm - D_out_nm) / sigma_nm)
    )


def cleanup_two_material_vfrac(vfrac_lipid, xp):
    lipid = xp.clip(vfrac_lipid, 0.0, 1.0)
    medium = 1.0 - lipid
    # Enforce exact voxelwise sum-to-one after cleanup.
    lipid = 1.0 - medium
    return lipid, medium
