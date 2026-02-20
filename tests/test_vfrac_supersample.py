import numpy as np
from scipy.special import erf

from vesicle_nrss.arguments import small_test_args
from vesicle_nrss.morphology import generate_vesicle_fields


def _manual_supersampled_vfrac(args):
    m = args.vfrac_supersample
    cz = (args.vd - 1) / 2.0
    cy = (args.ld - 1) / 2.0
    cx = (args.ld - 1) / 2.0

    z = ((np.arange(args.vd * m, dtype=np.float32) + 0.5) / m) - 0.5
    y = ((np.arange(args.ld * m, dtype=np.float32) + 0.5) / m) - 0.5
    x = ((np.arange(args.ld * m, dtype=np.float32) + 0.5) / m) - 0.5

    zz = (z[:, None, None] - cz) * args.PhysSize
    yy = (y[None, :, None] - cy) * args.PhysSize
    xx = (x[None, None, :] - cx) * args.PhysSize

    rho = np.sqrt(xx**2 + yy**2 + zz**2) - args.radius_nm
    hi = 0.5 * (erf((rho + args.D_out_nm) / args.sigma_nm) - erf((rho - args.D_out_nm) / args.sigma_nm))
    return hi.reshape(args.vd, m, args.ld, m, args.ld, m).mean(axis=(1, 3, 5))


def test_supersampled_vfrac_matches_manual_full_volume_path():
    args = small_test_args()
    args.vd = 8
    args.ld = 8
    args.vfrac_supersample = 2
    args.backend = "numpy"

    fields = generate_vesicle_fields(args)
    expected = _manual_supersampled_vfrac(args)

    assert np.allclose(fields["vfrac_lipid"], expected, atol=1e-6)
    assert np.allclose(fields["vfrac_lipid"] + fields["vfrac_medium"], 1.0, atol=1e-7)
