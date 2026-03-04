"""Tests for gsql_track.dispatch — execution mode, job distribution, env setup."""
import os

import pytest
from gsql_track.dispatch import (
    determine_execution_mode,
    distribute_jobs,
    setup_worker_env,
    merge_worker_logs,
    _get_nested,
    _set_nested,
    _is_oom_error,
)


# ── determine_execution_mode ──

def test_gpu_mode():
    mode, n, devices = determine_execution_mode([0, 1], n_workers=2)
    assert mode == "gpu"
    assert n == 2
    assert devices == [0, 1]


def test_cpu_mode():
    mode, n, devices = determine_execution_mode(None, n_workers=3)
    assert mode == "cpu"
    assert n == 3
    assert devices == [None, None, None]


def test_empty_devices_is_cpu():
    mode, _, _ = determine_execution_mode([], n_workers=2)
    assert mode == "cpu"


# ── distribute_jobs ──

def test_distribute_even():
    result = distribute_jobs([1, 2, 3, 4], 2)
    assert result == [[1, 3], [2, 4]]


def test_distribute_uneven():
    result = distribute_jobs([1, 2, 3], 2)
    assert result == [[1, 3], [2]]


def test_distribute_single_worker():
    result = distribute_jobs([1, 2, 3], 1)
    assert result == [[1, 2, 3]]


# ── setup_worker_env ──

def test_gpu_env():
    env = setup_worker_env("gpu", 2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_cpu_env():
    env = setup_worker_env("cpu", None)
    assert env["CUDA_VISIBLE_DEVICES"] == ""


# ── nested dict helpers ──

def test_get_set_nested():
    d = {"a": {"b": {"c": 10}}}
    assert _get_nested(d, "a.b.c") == 10
    _set_nested(d, "a.b.c", 20)
    assert d["a"]["b"]["c"] == 20


# ── merge_worker_logs ──

def test_merge_worker_logs(tmp_path):
    log1 = tmp_path / "w0.log"
    log2 = tmp_path / "w1.log"
    log1.write_text("worker 0 output\n")
    log2.write_text("worker 1 output\n")
    main = tmp_path / "main.log"
    merge_worker_logs([log1, log2], main)
    content = main.read_text()
    assert "worker 0" in content
    assert "worker 1" in content


def test_merge_worker_logs_missing(tmp_path):
    main = tmp_path / "main.log"
    merge_worker_logs([tmp_path / "nope.log"], main)
    # Should not raise, file may not exist


# ── OOM detection ──

def test_oom_detection_string():
    assert _is_oom_error(RuntimeError("CUDA out of memory"))
    assert not _is_oom_error(RuntimeError("dimension mismatch"))


# ── run_worker_loop ──

def test_run_worker_loop(tmp_path, monkeypatch):
    from gsql_track.dispatch import run_worker_loop
    monkeypatch.setattr("gsql_track.util.cleanup_resources", lambda: None)
    results = []
    def run_one(job):
        results.append(job)
    run_worker_loop(rank=0, jobs=["a", "b", "c"], run_one_fn=run_one, progress_prefix="test_worker")
    assert results == ["a", "b", "c"]
    # Check progress file
    import os
    pf = "/tmp/test_worker_0.txt"
    assert os.path.exists(pf)
    assert "Done" in open(pf).read()
    os.unlink(pf)


def test_run_worker_loop_error(tmp_path, monkeypatch):
    from gsql_track.dispatch import run_worker_loop
    monkeypatch.setattr("gsql_track.util.cleanup_resources", lambda: None)
    def run_one(job):
        raise ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        run_worker_loop(rank=0, jobs=["x"], run_one_fn=run_one, progress_prefix="test_err")
    import os
    pf = "/tmp/test_err_0.txt"
    if os.path.exists(pf):
        os.unlink(pf)


# ── collect_worker_outputs ──

def test_collect_worker_outputs():
    import subprocess
    from gsql_track.dispatch import collect_worker_outputs
    proc = subprocess.Popen(["echo", "hello"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    q = collect_worker_outputs([proc])
    idx, stdout, rc, stderr = q.get(timeout=5)
    assert idx == 0
    assert rc == 0
    assert "hello" in stdout


# ── drain_worker_outputs ──

def test_drain_worker_outputs():
    from queue import Queue
    from gsql_track.dispatch import drain_worker_outputs
    from gsql_track.util import ProgressWriter
    q = Queue()
    q.put((0, "some output", 1, None))  # failed worker
    writer = ProgressWriter(silent=True)
    drain_worker_outputs(q, writer)
    assert q.empty()


# ── oom_retry ──

def test_oom_retry_success():
    from gsql_track.dispatch import oom_retry
    def train_fn(config):
        return config["model"]["init_args"]["batch_size"]
    config = {"model": {"init_args": {"batch_size": 32}}}
    result = oom_retry(train_fn, config)
    assert result == 32


def test_oom_retry_halves_batch():
    from gsql_track.dispatch import oom_retry
    call_count = [0]
    def train_fn(config):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("CUDA out of memory")
        return config["model"]["init_args"]["batch_size"]
    config = {"model": {"init_args": {"batch_size": 32}}}
    result = oom_retry(train_fn, config, max_retries=2)
    assert result == 16


def test_oom_retry_non_oom_error():
    from gsql_track.dispatch import oom_retry
    def train_fn(config):
        raise ValueError("bad config")
    config = {"model": {"init_args": {"batch_size": 32}}}
    with pytest.raises(ValueError, match="bad config"):
        oom_retry(train_fn, config)


def test_oom_retry_exhausted():
    from gsql_track.dispatch import oom_retry
    def train_fn(config):
        raise RuntimeError("CUDA out of memory")
    config = {"model": {"init_args": {"batch_size": 32}}}
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        oom_retry(train_fn, config, max_retries=1)


# ── find_max_batch_size ──

def test_find_max_batch_size():
    from gsql_track.dispatch import find_max_batch_size
    call_count = [0]
    def train_fn(config):
        call_count[0] += 1
        bs = config["model"]["init_args"]["batch_size"]
        if bs > 64:
            raise RuntimeError("CUDA out of memory")
    config = {"model": {"init_args": {"batch_size": 16, "n_steps": 10}}}
    best = find_max_batch_size(train_fn, config)
    assert best == 64


# ── run_worker_loop_queue ──

def test_run_worker_loop_queue(tmp_path, monkeypatch):
    from gsql_track.dispatch import run_worker_loop_queue
    monkeypatch.setattr("gsql_track.util.cleanup_resources", lambda: None)
    jobs = [("M1", "T1"), ("M2", "T2")]
    idx = [0]
    def claim_fn(db_path, worker_id):
        if idx[0] < len(jobs):
            job = jobs[idx[0]]
            idx[0] += 1
            return job
        return None
    results = []
    def run_one(job):
        results.append(job)
    run_worker_loop_queue(rank=0, db_path=str(tmp_path / "db"), claim_fn=claim_fn,
                          run_one_fn=run_one, progress_prefix="test_queue")
    assert results == [("M1", "T1"), ("M2", "T2")]
    import os
    pf = "/tmp/test_queue_0.txt"
    if os.path.exists(pf):
        os.unlink(pf)
