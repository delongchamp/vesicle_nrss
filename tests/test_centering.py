import numpy as np

from vesicle_nrss.cache import get_radial_fields


def test_center_voxel_convention_for_odd_grid():
    fields = get_radial_fields(vd=5, ld=5, PhysSize=1.0, radius_nm=2.0, enable_cache=False)

    assert np.isclose(fields["theta"][2, 2, 2], 0.0)
    assert np.isclose(fields["psi"][2, 2, 2], 0.0)
    assert np.isclose(fields["rho_nm"][2, 2, 2], -2.0)
