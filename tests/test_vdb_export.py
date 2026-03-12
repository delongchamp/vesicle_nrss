from pathlib import Path

from vesicle_nrss.arguments import small_test_args
from vesicle_nrss.run import _save_vdb_if_requested
import vesicle_nrss.run as run_module


def test_save_vdb_if_requested_noop_when_folder_is_none():
    args = small_test_args()
    args.vdb_folder = None
    assert _save_vdb_if_requested(args, morph=object()) is None


def test_save_vdb_if_requested_uses_material1_and_all_fields(monkeypatch, tmp_path: Path):
    args = small_test_args()
    args.vdb_folder = tmp_path
    args.filename_tags = "tag_"
    args.filename = "base"
    args.filename_suffix = "_sfx"

    captured = {}

    def _fake_save_morph_to_vdb(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_module, "_has_morphtools", True)
    monkeypatch.setattr(run_module, "save_morph_to_vdb", _fake_save_morph_to_vdb)

    out_path = _save_vdb_if_requested(args, morph=object())
    expected_path = tmp_path / "tag_base_sfx.vdb"

    assert out_path == expected_path
    assert captured["output_path"] == str(expected_path)
    assert captured["component_indices"] == [1]
    assert captured["density_source"] == "vfrac"
    assert captured["save_fields"] == ["vfrac", "S", "psi", "theta"]
