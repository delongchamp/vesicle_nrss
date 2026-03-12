import copy

import numpy as np
import pytest

from vesicle_nrss.arguments import (
    RadiusSamplingMode,
    VesiclePopulationMode,
    small_test_args,
)
from vesicle_nrss.morphology import (
    _candidate_collides,
    _place_vesicles,
    generate_vesicle_fields,
)


def _minimum_image_distances_nm(args, center_nm):
    Lz = args.vd * args.PhysSize
    Ly = args.ld * args.PhysSize
    Lx = args.ld * args.PhysSize

    z_nm = (np.arange(args.vd, dtype=np.float32) + 0.5) * args.PhysSize
    y_nm = (np.arange(args.ld, dtype=np.float32) + 0.5) * args.PhysSize
    x_nm = (np.arange(args.ld, dtype=np.float32) + 0.5) * args.PhysSize

    dz = ((z_nm[:, None, None] - center_nm[0] + 0.5 * Lz) % Lz) - 0.5 * Lz
    dy = ((y_nm[None, :, None] - center_nm[1] + 0.5 * Ly) % Ly) - 0.5 * Ly
    dx = ((x_nm[None, None, :] - center_nm[2] + 0.5 * Lx) % Lx) - 0.5 * Lx
    return np.sqrt(dx**2 + dy**2 + dz**2)


def _multiple_base_args():
    args = small_test_args()
    args.population_mode = VesiclePopulationMode.MULTIPLE
    args.radius_sampling_mode = RadiusSamplingMode.CONSTANT
    args.backend = "numpy"
    args.vfrac_supersample = 1
    args.periodic_boundary_xyz = (True, True, True)
    args.placement_max_failures = 200
    return args


def test_list_mode_length_drives_num_vesicles_and_preserves_original_order():
    args = _multiple_base_args()
    args.vd = 48
    args.ld = 48
    args.PhysSize = 2.0
    args.radius_sampling_mode = RadiusSamplingMode.LIST
    args.radii_nm_list = [9.0, 5.0, 7.0]
    args.num_vesicles = 999
    args.base_seed = 123

    fields = generate_vesicle_fields(args)

    assert args.num_vesicles == 3
    assert fields["num_vesicles_requested"] == 3
    assert fields["num_vesicles_generated"] == 3
    assert fields["num_vesicles_placed"] == 3
    assert np.allclose(fields["accepted_radii_nm"], np.array([9.0, 5.0, 7.0], dtype=np.float32))
    assert fields["accepted_centers_nm"].shape == (3, 3)


def test_normal_sampling_uses_truncnorm_bounds_and_seed_reproducibility():
    args = _multiple_base_args()
    args.vd = 80
    args.ld = 80
    args.PhysSize = 2.0
    args.radius_sampling_mode = RadiusSamplingMode.NORMAL
    args.num_vesicles = 4
    args.radius_nm = 30.0
    args.radius_sigma_nm = 4.0
    args.base_seed = 17

    fields_a = generate_vesicle_fields(args)
    fields_b = generate_vesicle_fields(copy.deepcopy(args))

    lower = max(10.0, args.radius_nm - 3.0 * args.radius_sigma_nm)
    upper = args.radius_nm + 3.0 * args.radius_sigma_nm
    assert fields_a["accepted_radii_nm"].min() >= lower
    assert fields_a["accepted_radii_nm"].max() <= upper
    assert np.allclose(fields_a["accepted_radii_nm"], fields_b["accepted_radii_nm"])
    assert np.allclose(fields_a["accepted_centers_nm"], fields_b["accepted_centers_nm"])


def test_constant_sampling_is_seed_deterministic_for_radii_and_centers():
    args = _multiple_base_args()
    args.vd = 40
    args.ld = 40
    args.PhysSize = 2.0
    args.num_vesicles = 3
    args.radius_nm = 8.0
    args.base_seed = 31

    fields_a = generate_vesicle_fields(args)
    fields_b = generate_vesicle_fields(copy.deepcopy(args))

    assert np.allclose(fields_a["accepted_radii_nm"], np.full(3, 8.0, dtype=np.float32))
    assert np.allclose(fields_a["accepted_radii_nm"], fields_b["accepted_radii_nm"])
    assert np.allclose(fields_a["accepted_centers_nm"], fields_b["accepted_centers_nm"])


def test_first_placed_vesicle_is_at_box_center():
    args = _multiple_base_args()
    args.vd = 40
    args.ld = 24
    args.PhysSize = 1.5
    args.num_vesicles = 3
    args.radius_nm = 6.0
    args.base_seed = 101

    fields = generate_vesicle_fields(args)
    expected_center = np.array(
        [0.5 * args.vd * args.PhysSize, 0.5 * args.ld * args.PhysSize, 0.5 * args.ld * args.PhysSize],
        dtype=np.float32,
    )
    assert np.allclose(fields["accepted_centers_nm"][0], expected_center, atol=1e-6)


def test_constant_sampling_prefix_stability_across_num_vesicles():
    args_two = _multiple_base_args()
    args_two.vd = 32
    args_two.ld = 32
    args_two.PhysSize = 1.5
    args_two.radius_nm = 7.0
    args_two.base_seed = 29
    args_two.num_vesicles = 2

    args_three = copy.deepcopy(args_two)
    args_three.num_vesicles = 3

    fields_two = generate_vesicle_fields(args_two)
    fields_three = generate_vesicle_fields(args_three)

    n_two = fields_two["accepted_radii_nm"].shape[0]
    assert fields_three["accepted_radii_nm"].shape[0] >= n_two
    assert np.allclose(fields_two["accepted_radii_nm"], fields_three["accepted_radii_nm"][:n_two], atol=1e-6)
    assert np.allclose(fields_two["accepted_centers_nm"], fields_three["accepted_centers_nm"][:n_two], atol=1e-6)


def test_normal_sampling_prefix_stability_across_num_vesicles():
    args_two = _multiple_base_args()
    args_two.vd = 60
    args_two.ld = 60
    args_two.PhysSize = 2.0
    args_two.radius_sampling_mode = RadiusSamplingMode.NORMAL
    args_two.radius_nm = 20.0
    args_two.radius_sigma_nm = 2.0
    args_two.base_seed = 9
    args_two.num_vesicles = 2

    args_three = copy.deepcopy(args_two)
    args_three.num_vesicles = 3

    fields_two = generate_vesicle_fields(args_two)
    fields_three = generate_vesicle_fields(args_three)

    n_two = fields_two["accepted_radii_nm"].shape[0]
    assert fields_three["accepted_radii_nm"].shape[0] >= n_two
    assert np.allclose(fields_two["accepted_radii_nm"], fields_three["accepted_radii_nm"][:n_two], atol=1e-6)
    assert np.allclose(fields_two["accepted_centers_nm"], fields_three["accepted_centers_nm"][:n_two], atol=1e-6)


def test_collision_logic_uses_periodic_minimum_image_metric():
    args = _multiple_base_args()
    args.sigma_nm = 0.22
    args.collision_buffer_sigma_multiplier = 6.0
    box_lengths = np.array([20.0, 20.0, 20.0], dtype=np.float64)

    accepted_centers = np.array([[0.1, 10.0, 10.0]], dtype=np.float64)
    accepted_radii = np.array([4.0], dtype=np.float64)
    candidate_center = np.array([19.9, 10.0, 10.0], dtype=np.float64)

    assert _candidate_collides(
        candidate_center_nm=candidate_center,
        candidate_radius_nm=4.0,
        accepted_centers_nm=accepted_centers,
        accepted_radii_nm=accepted_radii,
        args=args,
        box_lengths_nm=box_lengths,
    )


def test_partial_placement_hits_consecutive_failure_threshold_and_reports(capsys):
    args = _multiple_base_args()
    args.vd = 8
    args.ld = 8
    args.PhysSize = 1.0
    args.num_vesicles = 3
    args.radius_nm = 4.0
    args.placement_max_failures = 5
    args.base_seed = 4

    fields = generate_vesicle_fields(args)
    captured = capsys.readouterr().out

    assert fields["num_vesicles_requested"] == 3
    assert fields["num_vesicles_generated"] == 1
    assert fields["num_vesicles_placed"] == 1
    assert fields["consecutive_failures_final"] == 5
    assert fields["accepted_centers_nm"].shape == (1, 3)
    assert "num_vesicles_requested=3" in captured
    assert "num_vesicles_generated=1" in captured
    assert "num_vesicles_placed=1" in captured


def test_consecutive_failure_counter_resets_after_success():
    class StubRNG:
        def __init__(self, samples):
            self._samples = iter(samples)

        def uniform(self, _low, _high):
            return next(self._samples)

    args = _multiple_base_args()
    args.sigma_nm = 0.22
    args.placement_max_failures = 3
    radii = np.array([4.0, 4.0], dtype=np.float64)
    box_lengths = np.array([20.0, 20.0, 20.0], dtype=np.float64)

    # candidate 1 accept, candidate 2 reject (collision), candidate 3 accept
    stub_rng = StubRNG(
        [
            1.0,
            1.0,
            1.0,
            1.2,
            1.2,
            1.2,
            7.0,
            7.0,
            7.0,
        ]
    )
    placement = _place_vesicles(args=args, radii_nm=radii, rng=stub_rng, box_lengths_nm=box_lengths)

    assert placement["num_vesicles_generated"] == 2
    assert placement["accepted_radii_nm_all"].shape == (2,)
    assert placement["consecutive_failures_final"] == 0


def test_masked_addition_and_euler_defaults_outside_support():
    args = _multiple_base_args()
    args.vd = 32
    args.ld = 32
    args.PhysSize = 1.0
    args.num_vesicles = 1
    args.radius_nm = 8.0
    args.sigma_nm = 0.5
    args.base_seed = 12

    fields = generate_vesicle_fields(args)
    center_nm = fields["accepted_centers_nm"][0]
    support_nm = (
        fields["accepted_radii_nm"][0]
        + 0.5 * args.collision_buffer_sigma_multiplier * args.sigma_nm
    )

    distance_nm = _minimum_image_distances_nm(args, center_nm=center_nm)
    outside_mask = distance_nm > (support_nm + 1e-6)

    assert np.allclose(fields["vfrac_lipid"][outside_mask], 0.0, atol=1e-6)
    assert np.allclose(fields["theta"][outside_mask], 0.0, atol=1e-6)
    assert np.allclose(fields["psi"][outside_mask], 0.0, atol=1e-6)


def test_two_material_closure_in_multiple_mode():
    args = _multiple_base_args()
    args.vd = 28
    args.ld = 28
    args.PhysSize = 1.5
    args.num_vesicles = 2
    args.radius_nm = 6.0
    args.base_seed = 8

    fields = generate_vesicle_fields(args)
    assert np.allclose(fields["vfrac_lipid"] + fields["vfrac_medium"], 1.0, atol=1e-6)


def test_multiple_mode_requires_full_periodic_tuple():
    args = _multiple_base_args()
    args.periodic_boundary_xyz = (True, False, True)
    with pytest.raises(ValueError):
        generate_vesicle_fields(args)


def test_single_mode_does_not_require_full_periodic_tuple():
    args = small_test_args()
    args.population_mode = VesiclePopulationMode.SINGLE
    args.periodic_boundary_xyz = (False, False, False)
    fields = generate_vesicle_fields(args)
    assert fields["accepted_radii_nm"].shape == (1,)
