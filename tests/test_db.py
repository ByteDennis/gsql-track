"""Tests for gsql_track.db — benchmark and tune DB operations."""
import json
import time

import pytest
from gsql_track.db import (
    get_db_connection,
    update_state,
    get_state,
    bench_init_db,
    bench_start_run,
    bench_finish_run,
    bench_load_results,
    bench_get_pending_runs,
    bench_claim_next_job,
    tune_init_db,
    tune_start_job,
    tune_finish_job,
    tune_fail_job,
    tune_get_job_id,
    tune_update_best_config,
    tune_claim_next_job,
    tune_save_config,
    tune_get_config,
    tune_get_failed_jobs,
    tune_get_best_config,
)


@pytest.fixture
def bench_db(tmp_path):
    db = tmp_path / "bench.db"
    bench_init_db(db)
    return db


@pytest.fixture
def tune_db(tmp_path):
    db = tmp_path / "tune.db"
    tune_init_db(db)
    return db


# ── State get/set ──

def test_state_roundtrip(bench_db):
    update_state(bench_db, "benchmark_state", "status", "running")
    assert get_state(bench_db, "benchmark_state", "status") == "running"


def test_state_default(bench_db):
    assert get_state(bench_db, "benchmark_state", "missing", default="nope") == "nope"


# ── Benchmark lifecycle ──

def test_bench_start_finish_load(bench_db):
    run_id = bench_start_run(bench_db, "BERT", "SST2", 0, exe_cmd="python train.py")
    assert run_id > 0
    bench_finish_run(bench_db, run_id, {"acc": 0.92, "f1": 0.88})
    results = bench_load_results(bench_db)
    assert len(results) == 1
    assert results[0]["model"] == "BERT"
    assert results[0]["best_metrics"]["acc"] == pytest.approx(0.92)


def test_bench_start_existing_row(bench_db):
    """UPDATE path when row already exists (queue mode)."""
    with get_db_connection(bench_db) as conn:
        conn.execute(
            "INSERT INTO benchmark_runs (model_name, task_name, seed, status) VALUES (?, ?, ?, 'pending')",
            ("CNN", "MNIST", 1),
        )
    run_id = bench_start_run(bench_db, "CNN", "MNIST", 1)
    with get_db_connection(bench_db) as conn:
        row = conn.execute("SELECT status FROM benchmark_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "running"


def test_bench_finish_with_error(bench_db):
    run_id = bench_start_run(bench_db, "BERT", "SST2", 0)
    bench_finish_run(bench_db, run_id, {}, error="OOM")
    with get_db_connection(bench_db) as conn:
        row = conn.execute("SELECT status, error FROM benchmark_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "OOM"


def test_bench_pending_runs(bench_db):
    with get_db_connection(bench_db) as conn:
        for seed in range(3):
            conn.execute(
                "INSERT INTO benchmark_runs (model_name, task_name, seed, status) VALUES (?, ?, ?, 'pending')",
                ("M", "T", seed),
            )
    pending = bench_get_pending_runs(bench_db)
    assert len(pending) == 3


def test_bench_claim_next_job(bench_db):
    with get_db_connection(bench_db) as conn:
        conn.execute(
            "INSERT INTO benchmark_runs (model_name, task_name, seed, status) VALUES (?, ?, ?, 'pending')",
            ("A", "B", 0),
        )
    job = bench_claim_next_job(bench_db, "w0")
    assert job == ("A", "B", 0)
    assert bench_claim_next_job(bench_db, "w0") is None  # queue empty


# ── Tune lifecycle ──

def test_tune_start_finish(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute(
            "INSERT INTO tuning_jobs (model_name, task_name, status) VALUES (?, ?, 'pending')",
            ("BERT", "SST2"),
        )
    tune_start_job(tune_db, "BERT", "SST2")
    tune_finish_job(tune_db, "BERT", "SST2", completed_trials=10)
    with get_db_connection(tune_db) as conn:
        row = conn.execute("SELECT status, completed_trials FROM tuning_jobs WHERE model_name = 'BERT'").fetchone()
    assert row["status"] == "completed"
    assert row["completed_trials"] == 10


def test_tune_fail_job(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute(
            "INSERT INTO tuning_jobs (model_name, task_name, status) VALUES (?, ?, 'running')",
            ("CNN", "MNIST"),
        )
    tune_fail_job(tune_db, "CNN", "MNIST", error="crash")
    failed = tune_get_failed_jobs(tune_db)
    assert len(failed) == 1
    assert failed[0]["error"] == "crash"


def test_tune_update_best_config(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute(
            "INSERT INTO tuning_jobs (model_name, task_name, status) VALUES (?, ?, 'running')",
            ("M", "T"),
        )
    tune_update_best_config(tune_db, "M", "T", trial_number=5, value=0.95, params={"lr": 0.01})
    best = tune_get_best_config(tune_db, "M", "T")
    assert best["lr"] == 0.01
    # Higher value should overwrite
    tune_update_best_config(tune_db, "M", "T", trial_number=8, value=0.97, params={"lr": 0.005})
    assert tune_get_best_config(tune_db, "M", "T")["lr"] == 0.005
    # Lower value should NOT overwrite
    tune_update_best_config(tune_db, "M", "T", trial_number=9, value=0.90, params={"lr": 0.1})
    assert tune_get_best_config(tune_db, "M", "T")["lr"] == 0.005


def test_tune_get_job_id(tune_db):
    assert tune_get_job_id(tune_db, "X", "Y") is None
    with get_db_connection(tune_db) as conn:
        conn.execute(
            "INSERT INTO tuning_jobs (model_name, task_name) VALUES (?, ?)", ("X", "Y"),
        )
    assert tune_get_job_id(tune_db, "X", "Y") is not None


def test_tune_config_roundtrip(tune_db):
    tune_save_config(tune_db, "n_trials", 50)
    assert tune_get_config(tune_db, "n_trials") == 50
    assert tune_get_config(tune_db, "missing", default=0) == 0


def test_tune_claim_next_job(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute(
            "INSERT INTO tuning_jobs (model_name, task_name, status) VALUES (?, ?, 'pending')",
            ("A", "B"),
        )
    job = tune_claim_next_job(tune_db, "w0")
    assert job == ("A", "B")
    assert tune_claim_next_job(tune_db, "w0") is None


# ── save_run_epilogue ──

def test_save_run_epilogue(bench_db):
    from gsql_track.db import save_run_epilogue
    save_run_epilogue(bench_db, "benchmark_state", 125.0)
    assert get_state(bench_db, "benchmark_state", "status") == "completed"
    assert get_state(bench_db, "benchmark_state", "elapsed_time") == 125.0
    assert "2.1m" == get_state(bench_db, "benchmark_state", "elapsed_human")


# ── bench_log_progress ──

def test_bench_log_progress(bench_db):
    from gsql_track.db import bench_log_progress
    run_id = bench_start_run(bench_db, "M", "T", 0)
    bench_log_progress(bench_db, run_id, "step 5", current_step=5, total_steps=10, metrics={"loss": 0.3})
    with get_db_connection(bench_db) as conn:
        row = conn.execute("SELECT * FROM benchmark_progress WHERE run_id = ?", (run_id,)).fetchone()
    assert row["message"] == "step 5"
    assert row["current_step"] == 5
    assert json.loads(row["current_metrics"])["loss"] == 0.3


# ── bench_finish_run with config ──

def test_bench_finish_run_with_config(bench_db):
    run_id = bench_start_run(bench_db, "M", "T", 0)
    bench_finish_run(bench_db, run_id, {"acc": 0.9}, config={"lr": 0.01})
    results = bench_load_results(bench_db)
    assert results[0]["config"]["lr"] == 0.01


# ── tune_save_job_config ──

def test_tune_save_job_config(tune_db):
    from gsql_track.db import tune_save_job_config
    with get_db_connection(tune_db) as conn:
        conn.execute("INSERT INTO tuning_jobs (model_name, task_name) VALUES (?, ?)", ("M", "T"))
    tune_save_job_config(tune_db, "M", "T", {"lr": 0.01, "bs": 32})
    with get_db_connection(tune_db) as conn:
        row = conn.execute("SELECT config_json FROM tuning_job_configs WHERE model_name='M'").fetchone()
    assert json.loads(row["config_json"])["lr"] == 0.01


# ── tune_update_job_stats ──

def test_tune_update_job_stats(tune_db):
    from gsql_track.db import tune_update_job_stats
    with get_db_connection(tune_db) as conn:
        conn.execute("INSERT INTO tuning_jobs (model_name, task_name) VALUES (?, ?)", ("M", "T"))
    tune_update_job_stats(tune_db, "M", "T", "mean=0.8, std=0.02")
    with get_db_connection(tune_db) as conn:
        row = conn.execute("SELECT trials_stats FROM tuning_jobs WHERE model_name='M'").fetchone()
    assert row["trials_stats"] == "mean=0.8, std=0.02"


# ── tune_get_failed_jobs with filters ──

def test_tune_get_failed_jobs_filter_model(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute("INSERT INTO tuning_jobs (model_name, task_name, status, error, end_time) VALUES (?, ?, 'failed', 'err1', 1)", ("A", "T1"))
        conn.execute("INSERT INTO tuning_jobs (model_name, task_name, status, error, end_time) VALUES (?, ?, 'failed', 'err2', 2)", ("B", "T2"))
    assert len(tune_get_failed_jobs(tune_db, model_name="A")) == 1
    assert len(tune_get_failed_jobs(tune_db, task_name="T2")) == 1
    assert len(tune_get_failed_jobs(tune_db)) == 2


# ── tune_get_best_config when no params ──

def test_tune_get_best_config_none(tune_db):
    with get_db_connection(tune_db) as conn:
        conn.execute("INSERT INTO tuning_jobs (model_name, task_name) VALUES (?, ?)", ("M", "T"))
    assert tune_get_best_config(tune_db, "M", "T") is None
