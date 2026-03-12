"""Deterministic sweep orchestration for serial and Ray execution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Literal

from .arguments import VesicleArgs
from .ray_adapter import run_sweep_with_ray_glados_actors
from .run import run_vesicle
from .utils import build_sweep_filename, dump_lzma_pickle, ensure_result_path


def _prepare_sweep_task(
    base_args: VesicleArgs,
    swept_arg: str,
    swept_value: Any,
    idx: int,
    base_filename: str,
    sweep_vdb_dir: Path | None = None,
) -> tuple[VesicleArgs, str]:
    task_args = copy.deepcopy(base_args)
    setattr(task_args, swept_arg, swept_value)
    task_args.base_seed = int(base_args.base_seed + idx)
    task_args.filename = f"{base_filename}_{swept_arg}_{idx:03d}"
    if sweep_vdb_dir is not None:
        task_args.vdb_folder = sweep_vdb_dir
    output_stem = build_sweep_filename(
        filename_tags=base_args.filename_tags,
        base_filename=base_filename,
        swept_arg=swept_arg,
        idx=idx,
        filename_suffix=base_args.filename_suffix,
    )
    return task_args, output_stem


def _ray_worker_run(args: VesicleArgs):
    return run_vesicle(args)


def _serialize_results(
    result_path: Path,
    ordered_results: list[Any],
    ordered_stems: list[str],
) -> None:
    ensure_result_path(result_path)
    for result, stem in zip(ordered_results, ordered_stems):
        dump_lzma_pickle(result_path / f"{stem}.pickle", result)


def run_vesicle_sweep(
    args: VesicleArgs,
    swept_arg: str,
    swept_values: Iterable[Any],
    *,
    parallel: Literal["serial", "ray"] = "serial",
    jobs_per_gpu: int = 1,
    return_results: bool = True,
    serialize: bool | None = None,
):
    values = list(swept_values)
    base_filename = args.filename
    sweep_vdb_dir = None
    if args.vdb_folder is not None:
        sweep_vdb_dir = Path(args.vdb_folder) / Path(args.result_path).name
        ensure_result_path(sweep_vdb_dir)

    indexed_args: list[tuple[int, VesicleArgs]] = []
    indexed_stems: list[tuple[int, str]] = []

    for idx, value in enumerate(values):
        task_args, stem = _prepare_sweep_task(
            base_args=args,
            swept_arg=swept_arg,
            swept_value=value,
            idx=idx,
            base_filename=base_filename,
            sweep_vdb_dir=sweep_vdb_dir,
        )
        indexed_args.append((idx, task_args))
        indexed_stems.append((idx, stem))

    if parallel == "serial":
        indexed_results = [(idx, run_vesicle(task_args)) for idx, task_args in indexed_args]
    elif parallel == "ray":
        indexed_results = run_sweep_with_ray_glados_actors(
            run_func=_ray_worker_run,
            indexed_args=indexed_args,
            jobs_per_gpu=jobs_per_gpu,
            pickle_path=args.result_path,
            retry_on_failure=args.ray_retry_on_failure,
            max_retries=args.ray_max_retries,
            retry_backoff_s=args.ray_retry_backoff_s,
        )
    else:
        raise ValueError("parallel must be 'serial' or 'ray'")

    indexed_results = sorted(indexed_results, key=lambda item: item[0])
    indexed_stems = sorted(indexed_stems, key=lambda item: item[0])

    ordered_results = [result for _, result in indexed_results]
    ordered_stems = [stem for _, stem in indexed_stems]

    do_serialize = serialize
    if do_serialize is None:
        do_serialize = (
            parallel == "ray"
            and args.ray_enable_serialization
            and not args.save_pickle_in_run
        )

    if do_serialize:
        _serialize_results(
            result_path=args.result_path,
            ordered_results=ordered_results,
            ordered_stems=ordered_stems,
        )

    if return_results:
        return ordered_results
    return ensure_result_path(args.result_path)
