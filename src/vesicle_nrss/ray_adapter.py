"""Thin adapter layer over ray_glados actor classes for sweep execution."""

from __future__ import annotations

import time
from typing import Any

RAY_HEAD_ADDR = "192.168.2.1"
RAY_PORT = 6379
RAY_TMPDIR = "/resdata/DeLongchamp/ray_temp"


def _get_actor_class(jobs_per_gpu: int):
    try:
        from ray_glados.sweep import (
            SweepActor,
            SweepActorFifthGPU,
            SweepActorHalfGPU,
            SweepActorQuarterGPU,
            SweepActorTenthGPU,
            SweepActorThirdGPU,
        )
    except Exception as exc:
        raise RuntimeError("ray_glados is required for parallel='ray' sweeps") from exc

    mapping = {
        1: SweepActor,
        2: SweepActorHalfGPU,
        3: SweepActorThirdGPU,
        4: SweepActorQuarterGPU,
        5: SweepActorFifthGPU,
        10: SweepActorTenthGPU,
    }
    if jobs_per_gpu not in mapping:
        raise ValueError("jobs_per_gpu must be one of: 1, 2, 3, 4, 5, 10")
    return mapping[jobs_per_gpu]


def initialize_ray_cluster_strict() -> Any:
    try:
        import ray
    except Exception as exc:
        raise RuntimeError("ray is required for parallel='ray' sweeps") from exc

    if ray.is_initialized():
        return ray

    try:
        ray.init(
            address=f"{RAY_HEAD_ADDR}:{RAY_PORT}",
            _node_ip_address=RAY_HEAD_ADDR,
            _temp_dir=RAY_TMPDIR,
            ignore_reinit_error=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to required Ray cluster at {RAY_HEAD_ADDR}:{RAY_PORT}"
        ) from exc

    return ray


def run_sweep_with_ray_glados_actors(
    *,
    run_func,
    indexed_args: list[tuple[int, Any]],
    jobs_per_gpu: int,
    pickle_path: Any,
    retry_on_failure: bool,
    max_retries: int,
    retry_backoff_s: float,
) -> list[tuple[int, Any]]:
    ray = initialize_ray_cluster_strict()
    ActorClass = _get_actor_class(jobs_per_gpu)

    total_actors = 5 * jobs_per_gpu
    actors = [ActorClass.remote(actor_id=i, pickle_path=pickle_path) for i in range(total_actors)]

    actor_load = {i: 0 for i in range(total_actors)}
    pending_queue = list(indexed_args)
    in_flight: dict[Any, tuple[int, int]] = {}
    retries = {idx: 0 for idx, _ in indexed_args}
    results: dict[int, Any] = {}

    def _submit_next(idx: int, args_obj: Any) -> None:
        if getattr(args_obj, "result_path", None) is None and pickle_path is not None:
            setattr(args_obj, "result_path", pickle_path)
        actor_idx = min(actor_load.keys(), key=lambda i: actor_load[i])
        future = actors[actor_idx].run_generic_task.remote(run_func, args_obj, task_id=idx)
        in_flight[future] = (idx, actor_idx)
        actor_load[actor_idx] += 1

    while pending_queue or in_flight:
        while pending_queue and len(in_flight) < len(actors):
            idx, args_obj = pending_queue.pop(0)
            _submit_next(idx, args_obj)

        done, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=5.0)
        if not done:
            continue

        for future in done:
            idx, actor_idx = in_flight.pop(future)
            actor_load[actor_idx] -= 1

            try:
                payload = ray.get(future)
                results[idx] = payload["result"]
            except Exception as exc:
                if retry_on_failure and retries[idx] < max_retries:
                    retries[idx] += 1
                    if retry_backoff_s > 0:
                        time.sleep(retry_backoff_s)
                    original_args = next(args_obj for task_idx, args_obj in indexed_args if task_idx == idx)
                    pending_queue.insert(0, (idx, original_args))
                else:
                    raise RuntimeError(f"Ray sweep task failed for idx={idx}") from exc

    return [(idx, results[idx]) for idx in sorted(results)]
