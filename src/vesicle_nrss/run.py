"""Vesicle simulation execution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .arguments import VesicleArgs
from .backend import seed_backend
from .morphology import build_vesicle_morph
from .results import VesicleResults
from .utils import dump_lzma_pickle, ensure_result_path


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

    model_plots = morph.visualize_materials(
        z_slice=args.vd // 2,
        runquiet=True,
        outputmat=[1, 2],
        outputplot=["vfrac", "S", "psi", "theta"],
        screen_euler=True,
        vfrac_range=[[0, 1]],
    )

    results = VesicleResults(
        data=deepcopy(data) if args.save_data_in_results else None,
        remeshed_data=remeshed_data,
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

    return results
