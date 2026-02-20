import numpy as np

from vesicle_nrss.cache import get_radial_fields


def test_radial_orientation_angles_follow_arctan2_definition():
    fields = get_radial_fields(vd=5, ld=5, PhysSize=1.0, radius_nm=0.0, enable_cache=False)

    assert np.isclose(fields["theta"][3, 2, 2], 0.0)
    assert np.isclose(fields["psi"][2, 2, 3], 0.0)
    assert np.isclose(fields["theta"][2, 3, 2], np.pi / 2)
    assert np.isclose(fields["psi"][2, 3, 2], np.pi / 2)
