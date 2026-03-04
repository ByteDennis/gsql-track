"""Subprocess worker dispatch infrastructure for parallel execution.

Shared utilities for bench_manager and tune_manager parallel modes:
- Execution mode determination (GPU isolation vs CPU)
- Round-robin job distribution across workers
- Subprocess launching with CUDA_VISIBLE_DEVICES isolation
- Async worker output collection and failure handling
- Worker CLI argument parsing

Example
-------
>>> mode, n_workers, devices = determine_execution_mode(device_ids=[0,1], n_workers=2)
>>> jobs_per_worker = distribute_jobs(items, n_workers)
>>> procs, log_files = launch_workers(n_workers, devices, mode, import_path, worker_configs)
>>> output_queue = collect_worker_outputs(procs)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

from . import util as U

ProgressWriter = U.ProgressWriter


def determine_execution_mode(
    device_ids: Optional[List[int]],
    n_workers: int,
) -> Tuple[str, int, List[Optional[int]]]:
    """Determine GPU/CPU mode, actual worker count, and device assignment.

    Returns:
        (mode, n_workers, device_assignment) where mode is "gpu" or "cpu"
    """
    if device_ids and len(device_ids) > 0:
        device_ids = list(device_ids)  # convert from OmegaConf ListConfig if needed
        return "gpu", len(device_ids), device_ids
    return "cpu", n_workers, [None] * n_workers


def distribute_jobs(jobs: List[Any], n_workers: int) -> List[List[Any]]:
    """Round-robin distribute jobs across workers."""
    buckets = [[] for _ in range(n_workers)]
    for idx, job in enumerate(jobs):
        buckets[idx % n_workers].append(job)
    return buckets


def setup_worker_env(mode: str, device_id: Optional[int]) -> dict:
    """Build environment dict for a worker subprocess with CUDA isolation."""
    env = os.environ.copy()
    if mode == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        env.update({
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "TORCH_CUDA_ARCH_LIST": "",
        })
    return env


def setup_worker_process(mode: str, device_id: Optional[int], rank: int):
    """Setup environment inside a worker process (called within subprocess).

    Configures CUDA visibility and disables lazy init for CPU mode.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if mode == "cpu":
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "",
            "TORCH_CUDA_ARCH_LIST": "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        })
        if torch is not None:
            torch.cuda._lazy_init = torch.cuda.init = lambda: None
    elif mode == "gpu":
        if torch is None or not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(f"Worker {rank}: CUDA unavailable")


def launch_workers(
    n_workers: int,
    device_assignment: List[Optional[int]],
    mode: str,
    import_path: str,
    worker_configs: List[Dict],
    log_dir: Optional[Path] = None,
    writer: Optional[ProgressWriter] = None,
) -> Tuple[List[subprocess.Popen], List[Tuple[Path, Any]]]:
    """Launch subprocess workers with GPU isolation.

    Args:
        n_workers: Number of workers to launch
        device_assignment: GPU device ID per worker (None for CPU)
        mode: "gpu" or "cpu"
        import_path: Module path for worker entry (e.g., "src.common.bench_manager")
        worker_configs: Per-worker config dicts (will be JSON-serialized as CLI arg)
        log_dir: If set, redirect stdout to per-worker log files in this directory
        writer: Optional progress writer for launch messages

    Returns:
        (procs, log_files) where log_files is list of (path, file_handle) tuples
    """
    procs = []
    log_files = []

    for rank in range(n_workers):
        env = setup_worker_env(mode, device_assignment[rank])

        if writer:
            if mode == "gpu":
                writer.write_event(f"Launching worker {rank} with CUDA_VISIBLE_DEVICES={device_assignment[rank]}")
            else:
                writer.write_event(f"Launching worker {rank} in CPU mode")

        import_code = (
            f"import sys; sys.path.append('{os.getcwd()}'); "
            f"from {import_path} import _subprocess_worker_main; _subprocess_worker_main()"
        )

        popen_kwargs = dict(
            env=env, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

        if log_dir:
            log_path = Path(log_dir) / f"run_worker_{rank}.log"
            log_file = open(log_path, 'w', buffering=1)
            log_files.append((log_path, log_file))
            popen_kwargs['stdout'] = log_file
        else:
            popen_kwargs['stdout'] = subprocess.PIPE

        p = subprocess.Popen(
            [sys.executable, "-c", import_code,
             json.dumps(worker_configs[rank], cls=U.JSONEncoder)],
            **popen_kwargs,
        )
        procs.append(p)

    return procs, log_files


def collect_worker_outputs(procs: List[subprocess.Popen]) -> Queue:
    """Start threads to collect worker stdout/stderr asynchronously.

    Returns a Queue of (worker_idx, stdout, return_code, stderr) tuples.
    """
    output_queue = Queue()

    for i, p in enumerate(procs):
        def read_output(proc_idx, proc):
            try:
                stdout, stderr = proc.communicate()
                output_queue.put((proc_idx, stdout, proc.returncode, stderr))
            except Exception as e:
                output_queue.put((proc_idx, f"Worker {proc_idx} error: {e}", -1, None))
        threading.Thread(target=read_output, args=(i, p), daemon=True).start()

    return output_queue


def drain_worker_outputs(output_queue: Queue, writer: ProgressWriter, label: str = ""):
    """Process any pending worker output/failure messages from the queue."""
    while not output_queue.empty():
        worker_idx, output, return_code, stderr = output_queue.get_nowait()
        if return_code != 0:
            writer.write_event(f"\nWorker {worker_idx} failed (code {return_code})")
            if output:
                writer.write_event(f"Output: {output[:500]}")


def parse_worker_args() -> Dict:
    """Parse worker subprocess CLI arguments. Returns config dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument('worker_config', help='JSON config')
    args = parser.parse_args()
    return json.loads(args.worker_config)


def run_worker_loop(
    rank: int,
    jobs: list,
    run_one_fn,
    progress_prefix: str = "worker",
    fmt_job=None,
):
    """Generic worker job loop with progress file and cleanup.

    Shared by bench_manager and tune_manager subprocess workers.

    Args:
        rank: Worker rank/index
        jobs: List of job descriptors
        run_one_fn: Callable that takes a single job and executes it
        progress_prefix: Prefix for /tmp progress file (e.g., "bench_worker", "tune_worker")
        fmt_job: Optional callable to format job for display (default: str)
    """
    fmt = fmt_job or str
    progress_file = f"/tmp/{progress_prefix}_{rank}.txt"
    for i, job in enumerate(jobs):
        with open(progress_file, 'w') as f:
            f.write(f"[{i+1}/{len(jobs)}] {fmt(job)}...\n")
        try:
            run_one_fn(job)
        except Exception as e:
            with open(progress_file, 'w') as f:
                f.write(f"ERROR: {fmt(job)}: {e}\n")
            raise
        finally:
            U.cleanup_resources()
    with open(progress_file, 'w') as f:
        f.write(f"Done ({len(jobs)} jobs completed)\n")


def merge_worker_logs(log_paths: List[Path], main_log_path: Path):
    """Merge per-worker log files into a single main log file.

    Args:
        log_paths: List of per-worker log file paths
        main_log_path: Path to the main log file to append to
    """
    try:
        with open(main_log_path, 'a') as f:
            for log_path in log_paths:
                if log_path.exists() and log_path.stat().st_size > 0:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"Worker Log: {log_path.name}\n")
                    f.write(f"{'=' * 60}\n")
                    f.write(log_path.read_text())
    except Exception:
        pass


def run_worker_loop_queue(
    rank: int,
    db_path: str,
    claim_fn,
    run_one_fn,
    progress_prefix: str = "worker",
    fmt_job=None,
    retry_delay: float = 0.5,
    max_retries: int = 3,
):
    """Queue-based worker loop: atomically claims jobs until queue is empty.

    Args:
        rank: Worker rank/index
        db_path: Path to SQLite database
        claim_fn: Callable(db_path, worker_id) -> job_tuple or None
        run_one_fn: Callable that takes a job tuple and executes it
        progress_prefix: Prefix for /tmp progress file
        fmt_job: Optional callable to format job for display
        retry_delay: Seconds to wait on SQLite lock contention
        max_retries: Max retries on lock contention before giving up
    """
    import sqlite3
    fmt = fmt_job or str
    worker_id = f"{progress_prefix}_{rank}"
    progress_file = f"/tmp/{progress_prefix}_{rank}.txt"
    completed = 0

    while True:
        # Claim next job with retry on lock contention
        job = None
        for attempt in range(max_retries):
            try:
                job = claim_fn(db_path, worker_id)
                break
            except sqlite3.OperationalError:
                import time as _time
                _time.sleep(retry_delay * (attempt + 1))
        else:
            # All retries exhausted, try one final time (let it raise)
            job = claim_fn(db_path, worker_id)

        if job is None:
            break  # Queue exhausted

        completed += 1
        with open(progress_file, 'w') as f:
            f.write(f"[job {completed}] {fmt(job)}...\n")
        try:
            run_one_fn(job)
        except Exception as e:
            with open(progress_file, 'w') as f:
                f.write(f"ERROR: {fmt(job)}: {e}\n")
            raise
        finally:
            U.cleanup_resources()

    with open(progress_file, 'w') as f:
        f.write(f"Done ({completed} jobs completed)\n")


# ─── GPU Memory Utilities ────────────────────────────────────────────────────


def _is_oom_error(exc: Exception) -> bool:
    """Detect CUDA out-of-memory errors via type check and string matching."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__


def _get_nested(config: dict, dot_path: str):
    """Get value from nested dict using dot-path notation."""
    keys = dot_path.split('.')
    obj = config
    for k in keys:
        obj = obj[k]
    return obj


def _set_nested(config: dict, dot_path: str, value):
    """Set value in nested dict using dot-path notation."""
    keys = dot_path.split('.')
    obj = config
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


def oom_retry(train_fn, config: dict, batch_size_key: str = 'model.init_args.batch_size', max_retries: int = 2):
    """Wrap a training call with OOM retry logic.

    On OOM: clears GPU memory, halves batch_size, retries.
    After max_retries halvings, raises the original error.

    Args:
        train_fn: Training function that takes a config dict
        config: Training configuration dict (modified in-place on retry)
        batch_size_key: Dot-path to batch_size in config
        max_retries: Maximum number of batch_size halvings

    Returns:
        Result from train_fn

    Example:
        >>> from src.common.dispatch_manager import oom_retry
        >>> result = oom_retry(train, config, batch_size_key='model.init_args.batch_size')
    """
    import copy
    config = copy.deepcopy(config)
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            return train_fn(config)
        except Exception as e:
            if not _is_oom_error(e):
                raise
            last_exc = e
            if attempt == max_retries:
                raise last_exc

            # Clear GPU memory
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except (ImportError, RuntimeError):
                pass

            # Halve batch size
            try:
                current_bs = _get_nested(config, batch_size_key)
                new_bs = max(1, current_bs // 2)
                _set_nested(config, batch_size_key, new_bs)
                print(f"OOM retry {attempt + 1}/{max_retries}: batch_size {current_bs} → {new_bs}")
            except (KeyError, TypeError):
                raise last_exc  # Can't find batch_size key, re-raise original


def find_max_batch_size(
    train_fn,
    config: dict,
    batch_size_key: str = 'model.init_args.batch_size',
    probe_steps_key: str = 'model.init_args.n_steps',
    probe_steps: int = 2,
):
    """Find maximum batch size by doubling until OOM.

    Each probe injects probe_steps into config so training exits quickly.
    Returns last successful batch_size.

    Args:
        train_fn: Training function that takes a config dict
        config: Training configuration dict
        batch_size_key: Dot-path to batch_size in config
        probe_steps_key: Dot-path to n_steps in config (for short probes)
        probe_steps: Number of steps per probe

    Returns:
        int: Maximum batch size that doesn't OOM

    Example:
        >>> best_bs = find_max_batch_size(train, config)
        >>> config['model']['init_args']['batch_size'] = best_bs
    """
    import copy

    current_bs = _get_nested(config, batch_size_key)
    last_good_bs = current_bs

    while True:
        probe_config = copy.deepcopy(config)
        _set_nested(probe_config, batch_size_key, current_bs)
        try:
            _set_nested(probe_config, probe_steps_key, probe_steps)
        except (KeyError, TypeError):
            pass  # probe_steps_key not in config, run full training

        try:
            train_fn(probe_config)
            last_good_bs = current_bs
            current_bs *= 2
        except Exception as e:
            if _is_oom_error(e):
                # Clear GPU memory
                try:
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except (ImportError, RuntimeError):
                    pass
                break
            else:
                raise

    print(f"find_max_batch_size: best batch_size = {last_good_bs}")
    return last_good_bs


__all__ = [
    "determine_execution_mode",
    "distribute_jobs",
    "setup_worker_env",
    "setup_worker_process",
    "launch_workers",
    "collect_worker_outputs",
    "drain_worker_outputs",
    "parse_worker_args",
    "run_worker_loop",
    "run_worker_loop_queue",
    "merge_worker_logs",
    "oom_retry",
    "find_max_batch_size",
]
