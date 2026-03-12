from pathlib import Path

from vesicle_nrss.arguments import small_test_args
from vesicle_nrss.sweep import run_vesicle_sweep


def test_serial_sweep_seed_order_and_filename_determinism(monkeypatch, tmp_path: Path):
    args = small_test_args()
    args.base_seed = 11
    args.filename_tags = "tag_"
    args.filename = "base"
    args.filename_suffix = "_sfx"
    args.result_path = tmp_path

    def _fake_run(task_args):
        return {
            "seed": task_args.base_seed,
            "radius": task_args.radius_nm,
            "filename": task_args.filename,
        }

    monkeypatch.setattr("vesicle_nrss.sweep.run_vesicle", _fake_run)

    results = run_vesicle_sweep(
        args,
        swept_arg="radius_nm",
        swept_values=[1.0, 2.0],
        parallel="serial",
        serialize=True,
    )

    assert [r["seed"] for r in results] == [11, 12]
    assert [r["radius"] for r in results] == [1.0, 2.0]
    assert [r["filename"] for r in results] == [
        "base_radius_nm_000",
        "base_radius_nm_001",
    ]

    assert (tmp_path / "tag_base_radius_nm_000_sfx.pickle").exists()
    assert (tmp_path / "tag_base_radius_nm_001_sfx.pickle").exists()


def test_serial_sweep_routes_vdb_exports_to_sweep_subdir(monkeypatch, tmp_path: Path):
    args = small_test_args()
    args.filename = "base"
    args.result_path = tmp_path / "pickles"
    args.vdb_folder = tmp_path / "vdb_root"

    observed_vdb_folders = []

    def _fake_run(task_args):
        observed_vdb_folders.append(Path(task_args.vdb_folder))
        return {"ok": True}

    monkeypatch.setattr("vesicle_nrss.sweep.run_vesicle", _fake_run)
    run_vesicle_sweep(
        args,
        swept_arg="radius_nm",
        swept_values=[1.0, 2.0, 3.0],
        parallel="serial",
        serialize=False,
    )

    expected = Path(args.vdb_folder) / Path(args.result_path).name
    assert expected.exists()
    assert expected.is_dir()
    assert observed_vdb_folders
    assert all(path == expected for path in observed_vdb_folders)
