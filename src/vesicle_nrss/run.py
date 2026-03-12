"""Vesicle simulation execution."""

from __future__ import annotations

import json
import socket
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import numpy as np

from .arguments import VesicleArgs
from .backend import seed_backend
from .morphology import build_vesicle_morph
from .results import VesicleResults
from .utils import dump_lzma_pickle, ensure_result_path

try:
    from morphtools.vdb import save_morph_to_vdb

    _has_morphtools = True
except Exception:  # pragma: no cover - optional dependency
    _has_morphtools = False


@lru_cache(maxsize=1)
def _get_vesicle_nrss_version() -> str | None:
    try:
        return metadata.version("vesicle_nrss")
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _get_git_commit_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _build_vdb_metadata_payload(args: VesicleArgs, vdb_path: Path) -> dict:
    model_parameters = {
        "vd": _json_value(args.vd),
        "ld": _json_value(args.ld),
        "PhysSize": _json_value(args.PhysSize),
        "radius_nm": _json_value(args.radius_nm),
        "num_vesicles": _json_value(args.num_vesicles),
        "radius_sigma_nm": _json_value(args.radius_sigma_nm),
        "S_lipid": _json_value(args.S_lipid),
    }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "vesicle_nrss_version": _get_vesicle_nrss_version(),
        "git_commit_sha": _get_git_commit_sha(),
    }
    return {
        "vdb_path": str(vdb_path),
        "model_parameters": model_parameters,
        "provenance": provenance,
    }


def _write_vdb_metadata_json(args: VesicleArgs, vdb_path: Path) -> Path:
    metadata_path = vdb_path.with_suffix(".json")
    payload = _build_vdb_metadata_payload(args, vdb_path)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return metadata_path


def _save_vdb_if_requested(args: VesicleArgs, morph) -> Path | None:
    if args.vdb_folder is None:
        return None

    vdb_dir = ensure_result_path(Path(args.vdb_folder) / Path(args.result_path).name)
    vdb_filename = f"{args.filename_tags}{args.filename}{args.filename_suffix}.vdb"
    vdb_path = vdb_dir / vdb_filename

    if not _has_morphtools:
        print(f"Warning: morphtools not available, skipping VDB save for {vdb_filename}")
        return None

    try:
        save_morph_to_vdb(
            morph=morph,
            output_path=str(vdb_path),
            component_indices=[1],
            density_source="vfrac",
            save_fields=["vfrac", "S", "psi", "theta"],
        )
    except Exception as exc:
        print(f"Warning: Could not save morphology to VDB: {exc}")
        return None

    _write_vdb_metadata_json(args, vdb_path)
    return vdb_path


def run_vesicle(args: VesicleArgs) -> VesicleResults:
    seed_backend(args.base_seed, args.backend)
    morph = build_vesicle_morph(args)

    if hasattr(morph, "inputData") and hasattr(morph.inputData, "windowingType"):
        morph.inputData.windowingType = 0

    if hasattr(morph, "EAngleRotation"):
        morph.EAngleRotation = list(args.EAngleRotation)
    morph.validate_all(quiet=False)

    data = morph.run(stdout=True, stderr=False)

    from PyHyperScattering.integrate import WPIntegrator

    integrator = WPIntegrator(use_chunked_processing=False)
    remeshed_data = integrator.integrateImageStack(data)

    I = remeshed_data.mean(dim="chi")
    if args.compute_polarized_traces:
        I_para = (
            remeshed_data.isel(chi=slice(0, 45)).sum(dim="chi")
            + remeshed_data.isel(chi=slice(135, 225)).sum(dim="chi")
            + remeshed_data.isel(chi=slice(315, 360)).sum(dim="chi")
        )
        I_perp = remeshed_data.isel(chi=slice(45, 135)).sum(dim="chi") + remeshed_data.isel(
            chi=slice(225, 315)
        ).sum(dim="chi")
        if args.return_I_para_perp:
            out_I_para = I_para
            out_I_perp = I_perp
            out_A = None
        else:
            out_I_para = None
            out_I_perp = None
            out_A = (I_para - I_perp) / (I_para + I_perp)
    else:
        out_I_para = None
        out_I_perp = None
        out_A = None

    if args.save_model_plots_in_results:
        model_plots = morph.visualize_materials(
            z_slice=args.vd // 2,
            runquiet=True,
            outputmat=[1, 2],
            outputplot=["vfrac", "S", "psi", "theta"],
            outputaxes=False,
            screen_euler=True,
            vfrac_range=[[0, 1]],
            plotstyle='dark',
        )
    else:
        model_plots = None

    results = VesicleResults(
        data=deepcopy(data) if args.save_data_in_results else None,
        remeshed_data=remeshed_data if args.save_remeshed_data_in_results else None,
        I=I,
        I_para=out_I_para,
        I_perp=out_I_perp,
        A=out_A,
        model_plots=model_plots,
        args=args if args.save_args_in_results else None,
        accepted_radii_nm=np.asarray(
            getattr(morph, "accepted_radii_nm", np.array([args.radius_nm], dtype=np.float32)),
            dtype=np.float32,
        ),
    )

    if args.save_pickle_in_run:
        result_path = ensure_result_path(Path(args.result_path))
        save_filename = f"{args.filename_tags}{args.filename}{args.filename_suffix}.pickle"
        dump_lzma_pickle(result_path / save_filename, results)

    _save_vdb_if_requested(args, morph)

    return results
