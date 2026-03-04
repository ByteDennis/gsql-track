"""Shared SQLite database utilities for bench_manager and tune_manager.

Provides connection management, generic state persistence, and all SQL operations
for both benchmark and tuning workflows.

Example
-------
>>> with get_db_connection(db_path) as conn:
...     conn.execute("SELECT * FROM table")
>>> update_state(db_path, 'status', 'completed')
>>> get_state(db_path, 'status', default='pending')
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional


@contextmanager
def get_db_connection(db_path: Path):
    """SQLite connection context manager with auto-commit and row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def update_state(db_path: Path, table: str, key: str, value: Any):
    """Update a key-value state entry in a state table."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )


def get_state(db_path: Path, table: str, key: str, default: Any = None) -> Any:
    """Get a state value from a state table. Returns default if not found."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT value FROM {table} WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row['value']) if row else default


def save_run_epilogue(db_path: Path, state_table: str, elapsed_time: float):
    """Save standard run completion state (elapsed_time, elapsed_human, status=completed).

    Used by both bench_manager and tune_manager after run completion.
    """
    from . import util as U
    update_state(db_path, state_table, 'elapsed_time', elapsed_time)
    update_state(db_path, state_table, 'elapsed_human', U.fmt_duration(elapsed_time))
    update_state(db_path, state_table, 'status', 'completed')


# ─── Benchmark DB ───────────────────────────────────────────────────────────────

def bench_init_db(db_path: Path):
    """Initialize benchmark database schema."""
    with get_db_connection(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, task_name TEXT NOT NULL,
            seed INTEGER NOT NULL, status TEXT DEFAULT 'pending', start_time REAL, end_time REAL, error TEXT, debug TEXT,
            UNIQUE(model_name, task_name, seed))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_results (
            run_id INTEGER PRIMARY KEY,
            best_metrics TEXT NOT NULL,
            time_spent REAL,
            start_time TEXT,
            config TEXT,
            FOREIGN KEY(run_id) REFERENCES benchmark_runs(id))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_state (
            key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, timestamp REAL NOT NULL,
            current_step INTEGER, total_steps INTEGER,
            current_metrics TEXT, message TEXT,
            FOREIGN KEY(run_id) REFERENCES benchmark_runs(id))""")
        # Add new columns to existing databases (migration)
        for alter in [
            "ALTER TABLE benchmark_results ADD COLUMN start_time TEXT",
            "ALTER TABLE benchmark_results ADD COLUMN time_spent REAL",
            "ALTER TABLE benchmark_results ADD COLUMN time_spent_human TEXT",
            "ALTER TABLE benchmark_runs ADD COLUMN worker_id TEXT",
        ]:
            try:
                conn.execute(alter)
            except Exception:
                pass  # Column already exists


def bench_log_progress(db_path: Path, run_id: int, message: str,
                       current_step: int = None, total_steps: int = None, metrics: Dict = None):
    """Log detailed progress for a benchmark run."""
    with get_db_connection(db_path) as conn:
        conn.execute("""INSERT INTO benchmark_progress
                        (run_id, timestamp, current_step, total_steps, current_metrics, message)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     (run_id, time.time(), current_step, total_steps,
                      json.dumps(metrics) if metrics else None, message))


def bench_start_run(db_path: Path, model_name: str, task_name: str, seed: int, exe_cmd: str = None) -> int:
    """Record run start in database. Uses UPDATE if row exists (queue mode), INSERT otherwise."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM benchmark_runs WHERE model_name=? AND task_name=? AND seed=?",
            (model_name, task_name, seed)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE benchmark_runs SET status='running', start_time=?, debug=? WHERE id=?",
                (time.time(), exe_cmd, row['id'])
            )
            return row['id']
        else:
            cursor = conn.execute(
                "INSERT INTO benchmark_runs (model_name, task_name, seed, status, start_time, debug) VALUES (?, ?, ?, 'running', ?, ?)",
                (model_name, task_name, seed, time.time(), exe_cmd)
            )
            return cursor.lastrowid


def bench_finish_run(db_path: Path, run_id: int, result_summary: Dict, config: Dict = None, error: str = None):
    """Record run completion in database."""
    from . import util as U
    status = 'failed' if error else 'completed'

    # train_fn returns best_metrics directly
    best_metrics = result_summary if isinstance(result_summary, dict) else {}

    with get_db_connection(db_path) as conn:
        end_time = time.time()
        conn.execute("UPDATE benchmark_runs SET status = ?, end_time = ?, error = ? WHERE id = ?",
                    (status, end_time, error, run_id))
        if not error:
            # Get start_time from benchmark_runs table
            row = conn.execute("SELECT start_time FROM benchmark_runs WHERE id = ?", (run_id,)).fetchone()
            start_time = row['start_time'] if row else None
            # Convert to human-readable format
            start_time_text = None
            if start_time:
                start_time_text = dt.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            # Calculate time_spent from start_time and end_time
            time_spent = (end_time - start_time) if start_time else None
            time_spent_human = U.fmt_duration(time_spent) if time_spent else None

            conn.execute("""INSERT OR REPLACE INTO benchmark_results
                           (run_id, best_metrics, time_spent, time_spent_human, start_time, config)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (run_id,
                         json.dumps(best_metrics),
                         time_spent,
                         time_spent_human,
                         start_time_text,
                         json.dumps(config, cls=U.JSONEncoder) if config else None))


def bench_load_results(db_path: Path) -> List[Dict]:
    """Load all completed results from database."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute("""SELECT r.model_name, r.task_name, r.seed,
            res.best_metrics, res.time_spent, res.start_time, res.config
            FROM benchmark_runs r JOIN benchmark_results res ON r.id = res.run_id
            WHERE r.status = 'completed'""").fetchall()
        return [{
            'model': r['model_name'],
            'task': r['task_name'],
            'seed': r['seed'],
            'best_metrics': json.loads(r['best_metrics']),
            'time_spent': r['time_spent'],
            'start_time': r['start_time'],
            'config': json.loads(r['config']) if r['config'] else None
        } for r in rows]


def bench_get_pending_runs(db_path: Path) -> List[tuple]:
    """Get list of pending runs from database."""
    with get_db_connection(db_path) as conn:
        rows = conn.execute("SELECT model_name, task_name, seed FROM benchmark_runs WHERE status = 'pending'").fetchall()
        return [(r['model_name'], r['task_name'], r['seed']) for r in rows]


# ─── Tune DB ────────────────────────────────────────────────────────────────────

def tune_init_db(db_path: Path):
    """Schema design (simplified):
    - tuning_jobs: Job-level metadata and best trial info
    - tuning_state: Global state (e.g., elapsed_time)
    - tuning_config: Global tuning configuration (n_trials, timeout, sampler, etc.)
    - Trial history stored in per-job Optuna studies (not in this database)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection(db_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS tuning_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            task_name TEXT NOT NULL,
            study_path TEXT,
            status TEXT DEFAULT 'pending',
            n_trials INTEGER DEFAULT 0,
            completed_trials INTEGER DEFAULT 0,
            metric TEXT,
            direction TEXT,
            start_time REAL,
            end_time REAL,
            elapsed_human TEXT,
            error TEXT,
            best_trial_number INTEGER,
            best_value REAL,
            best_params TEXT,
            latest_debug_cmd TEXT,
            trials_stats TEXT,
            UNIQUE(model_name, task_name)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tuning_job_configs (
            model_name TEXT NOT NULL,
            task_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            PRIMARY KEY(model_name, task_name)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tuning_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tuning_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_jobs_status ON tuning_jobs(status)""")
        # Migration: add worker_id column
        try:
            conn.execute("ALTER TABLE tuning_jobs ADD COLUMN worker_id TEXT")
        except Exception:
            pass  # Column already exists


def tune_get_job_id(db_path: Path, model_name: str, task_name: str) -> Optional[int]:
    """Get the job ID for a given model and task name. Returns None if not found."""
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM tuning_jobs WHERE model_name=? AND task_name=?",
            (model_name, task_name)
        ).fetchone()
        return row['id'] if row else None


def tune_save_job_config(db_path: Path, model_name: str, task_name: str, config: dict):
    """Save per-job merged config to DB."""
    from . import util as U
    with get_db_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tuning_job_configs (model_name, task_name, config_json) VALUES (?, ?, ?)",
            (model_name, task_name, json.dumps(config, cls=U.JSONEncoder, indent=2))
        )


def tune_start_job(db_path: Path, model_name: str, task_name: str):
    """Mark job as running."""
    with get_db_connection(db_path) as conn:
        conn.execute("UPDATE tuning_jobs SET status='running', start_time=? WHERE model_name=? AND task_name=?",
                    (time.time(), model_name, task_name))


def tune_finish_job(db_path: Path, model_name: str, task_name: str, completed_trials: int = 0):
    """Mark job as completed."""
    from . import util as U
    end_time = time.time()
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT start_time FROM tuning_jobs WHERE model_name=? AND task_name=?",
                          (model_name, task_name)).fetchone()
        elapsed_human = ""
        if row and row['start_time']:
            elapsed_human = U.fmt_duration(end_time - row['start_time'])
        conn.execute("""UPDATE tuning_jobs SET status='completed', end_time=?, elapsed_human=?, completed_trials=?
                       WHERE model_name=? AND task_name=?""",
                    (end_time, elapsed_human, completed_trials, model_name, task_name))


def tune_fail_job(db_path: Path, model_name: str, task_name: str, error: str):
    """Mark job as failed."""
    with get_db_connection(db_path) as conn:
        conn.execute("UPDATE tuning_jobs SET status='failed', end_time=?, error=? WHERE model_name=? AND task_name=?",
                    (time.time(), error, model_name, task_name))


def tune_update_best_config(
    db_path: Path,
    model_name: str,
    task_name: str,
    trial_number: int,
    value: float,
    params: Dict,
    debug_cmd: str = None
):
    """Update best config in tuning_jobs (best_trial_number, best_value, best_params)."""
    with get_db_connection(db_path) as conn:
        current = conn.execute(
            "SELECT best_value FROM tuning_jobs WHERE model_name=? AND task_name=?",
            (model_name, task_name)
        ).fetchone()
        if not current or current['best_value'] is None or value > current['best_value']:
            conn.execute("""UPDATE tuning_jobs
                         SET best_trial_number=?, best_value=?, best_params=?, latest_debug_cmd=?
                         WHERE model_name=? AND task_name=?""",
                        (trial_number, value, json.dumps(params), debug_cmd, model_name, task_name))


def tune_update_job_stats(
    db_path: Path,
    model_name: str,
    task_name: str,
    stats_str: str,
):
    """Update trial statistics in tuning_jobs table."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """UPDATE tuning_jobs SET trials_stats=? WHERE model_name=? AND task_name=?""",
            (stats_str, model_name, task_name)
        )


def tune_save_config(db_path: Path, key: str, value: Any):
    """Save configuration to database."""
    from . import util as U
    if hasattr(value, '__dict__') and hasattr(value, '_metadata'):
        from omegaconf import OmegaConf
        value = OmegaConf.to_container(value, resolve=True)
    with get_db_connection(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO tuning_config (key, value) VALUES (?, ?)",
                    (key, json.dumps(value, cls=U.JSONEncoder)))


def tune_get_config(db_path: Path, key: str, default=None):
    """Read a value from the tuning_config table."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("SELECT value FROM tuning_config WHERE key=?", (key,)).fetchone()
        return json.loads(row['value']) if row else default


def tune_get_failed_jobs(db_path: Path, model_name: str = None, task_name: str = None) -> List[Dict]:
    """Get all failed jobs with debug commands.

    Example:
        >>> failed = tune_get_failed_jobs(db_path, model_name="BERT")
        >>> for job in failed:
        ...     print(f"{job['model']} on {job['task']} failed: {job['error']}")
        ...     print(f"Debug with: {job['debug']}")
    """
    with get_db_connection(db_path) as conn:
        query = """
            SELECT model_name, task_name, error, latest_debug_cmd
            FROM tuning_jobs
            WHERE status = 'failed'
        """
        params = []
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if task_name:
            query += " AND task_name = ?"
            params.append(task_name)
        query += " ORDER BY end_time DESC"
        rows = conn.execute(query, params).fetchall()
        return [ {
            'model': row['model_name'],
            'task': row['task_name'],
            'error': row['error'],
            'debug': row['latest_debug_cmd'],
        } for row in rows ]


def tune_get_best_config(db_path: Path, model_name: str, task_name: str) -> Optional[Dict]:
    """Get best hyperparameters for model+task."""
    with get_db_connection(db_path) as conn:
        row = conn.execute("""
            SELECT best_params FROM tuning_jobs
            WHERE model_name = ? AND task_name = ?
        """, (model_name, task_name)).fetchone()
        return json.loads(row['best_params']) if row and row['best_params'] else None


# ─── Atomic Job Claiming (Queue Mode) ────────────────────────────────────────

def bench_claim_next_job(db_path: Path, worker_id: str) -> Optional[tuple]:
    """Atomically claim the next pending benchmark run. Returns (model_name, task_name, seed) or None."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, model_name, task_name, seed FROM benchmark_runs WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE benchmark_runs SET status='running', start_time=?, worker_id=? WHERE id=?",
            (time.time(), worker_id, row['id'])
        )
        conn.commit()
        return (row['model_name'], row['task_name'], row['seed'])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def tune_claim_next_job(db_path: Path, worker_id: str) -> Optional[tuple]:
    """Atomically claim the next pending tuning job. Returns (model_name, task_name) or None."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, model_name, task_name FROM tuning_jobs WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE tuning_jobs SET status='claimed', start_time=?, worker_id=? WHERE id=?",
            (time.time(), worker_id, row['id'])
        )
        conn.commit()
        return (row['model_name'], row['task_name'])
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "get_db_connection",
    "update_state",
    "get_state",
    "save_run_epilogue",
    # Benchmark DB
    "bench_init_db",
    "bench_log_progress",
    "bench_start_run",
    "bench_finish_run",
    "bench_load_results",
    "bench_get_pending_runs",
    # Tune DB
    "tune_init_db",
    "tune_get_job_id",
    "tune_save_job_config",
    "tune_start_job",
    "tune_finish_job",
    "tune_fail_job",
    "tune_update_best_config",
    "tune_update_job_stats",
    "tune_save_config",
    "tune_get_config",
    "tune_get_failed_jobs",
    "tune_get_best_config",
    # Queue mode
    "bench_claim_next_job",
    "tune_claim_next_job",
]
