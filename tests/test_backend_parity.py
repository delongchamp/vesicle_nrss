import numpy as np
import pytest

from vesicle_nrss.arguments import (
    RadiusSamplingMode,
    VesiclePopulationMode,
    small_test_args,
)
from vesicle_nrss.backend import to_numpy
from vesicle_nrss.morphology import generate_vesicle_fields


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cupy") is None,
    reason="cupy is not installed",
)
def test_numpy_and_cupy_field_generation_parity():
    args_np = small_test_args()
    args_np.vd = 8
    args_np.ld = 8
    args_np.backend = "numpy"
    args_np.vfrac_supersample = 1

    args_cp = small_test_args()
    args_cp.vd = 8
    args_cp.ld = 8
    args_cp.backend = "cupy"
    args_cp.vfrac_supersample = 1

    fields_np = generate_vesicle_fields(args_np)
    fields_cp = generate_vesicle_fields(args_cp)

    for key in ["vfrac_lipid", "vfrac_medium", "theta", "psi", "S_lipid", "S_medium"]:
        assert np.allclose(fields_np[key], to_numpy(fields_cp[key]), atol=1e-6)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cupy") is None,
    reason="cupy is not installed",
)
def test_numpy_and_cupy_multiple_mode_field_generation_parity():
    args_np = small_test_args()
    args_np.vd = 24
    args_np.ld = 24
    args_np.population_mode = VesiclePopulationMode.MULTIPLE
    args_np.num_vesicles = 2
    args_np.radius_sampling_mode = RadiusSamplingMode.CONSTANT
    args_np.radius_nm = 5.0
    args_np.vfrac_supersample = 1
    args_np.base_seed = 19
    args_np.backend = "numpy"

    args_cp = small_test_args()
    args_cp.vd = 24
    args_cp.ld = 24
    args_cp.population_mode = VesiclePopulationMode.MULTIPLE
    args_cp.num_vesicles = 2
    args_cp.radius_sampling_mode = RadiusSamplingMode.CONSTANT
    args_cp.radius_nm = 5.0
    args_cp.vfrac_supersample = 1
    args_cp.base_seed = 19
    args_cp.backend = "cupy"

    fields_np = generate_vesicle_fields(args_np)
    fields_cp = generate_vesicle_fields(args_cp)

    for key in ["vfrac_lipid", "vfrac_medium", "theta", "psi", "S_lipid", "S_medium"]:
        assert np.allclose(fields_np[key], to_numpy(fields_cp[key]), atol=1e-6)
    assert np.allclose(fields_np["accepted_radii_nm"], fields_cp["accepted_radii_nm"], atol=1e-6)
    assert np.allclose(fields_np["accepted_centers_nm"], fields_cp["accepted_centers_nm"], atol=1e-6)
