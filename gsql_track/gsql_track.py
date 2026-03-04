"""
gsql_track — lightweight experiment tracking backed by SQLite.

Two APIs:

1. Standalone (no dependencies beyond stdlib):

    t = GsqlTrack("mnist")
    run = t.start_run("cnn-001")
    run.log_params({"lr": 0.001, "batch_size": 64})
    run.log(step=0, loss=0.5, accuracy=0.85)    # step-level
    run.log(epoch=1, val_acc=0.91)               # epoch-level
    run.finish()

2. Wrapper for existing Tracker class:

    tracker = tracked(Tracker(config), experiment="mnist")
    # training loop unchanged — log_epoch/log_step/finalize auto-persist

All data goes to ~/.gsql/track.db — browse with `gsql ~/.gsql/track.db`.
"""

import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, experiment_id INTEGER, name TEXT DEFAULT '',
    status TEXT DEFAULT 'RUNNING',
    start_time TEXT DEFAULT (datetime('now')), end_time TEXT,
    git_commit TEXT, source TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE TABLE IF NOT EXISTS params (
    run_id TEXT, key TEXT, value TEXT, PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT, key TEXT, value REAL, step INTEGER, epoch INTEGER,
    timestamp TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS latest_metrics (
    run_id TEXT, key TEXT, value REAL, step INTEGER, epoch INTEGER,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT, example_id TEXT, prediction TEXT,
    confidence REAL DEFAULT 1.0, label TEXT, correct INTEGER,
    input_text TEXT,
    PRIMARY KEY (run_id, example_id)
);
CREATE INDEX IF NOT EXISTS idx_pred_example ON predictions(example_id);
CREATE VIEW IF NOT EXISTS run_summary AS
SELECT r.id, r.name, r.status, e.name as experiment,
       r.start_time, r.end_time, r.git_commit, r.source
FROM runs r JOIN experiments e ON r.experiment_id = e.id;
"""


def _open_db(path=None):
    """Open (and initialize) the track database."""
    if path is None:
        gsql_dir = Path.home() / ".gsql"
        gsql_dir.mkdir(parents=True, exist_ok=True)
        path = str(gsql_dir / "track.db")
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    # Migrate: add columns if missing (for existing databases)
    try:
        conn.execute("ALTER TABLE experiments ADD COLUMN metadata TEXT DEFAULT '{}'")
    except sqlite3.OperationalError:
        pass
    for table in ("metrics", "latest_metrics"):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN epoch INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def _get_git_commit():
    """Return short git commit hash, or empty string."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _flatten(obj, prefix="", sep="."):
    """Flatten a nested dict/object into dot-separated key-value strings."""
    items = {}
    if hasattr(obj, "__dict__"):
        obj = vars(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, (dict,)) or hasattr(v, "__dict__"):
                items.update(_flatten(v, key, sep))
            else:
                items[key] = str(v)
    return items


class _GsqlRun:
    """A single training run. Created via GsqlTrack.start_run()."""

    def __init__(self, conn, run_id):
        self._conn = conn
        self.id = run_id

    def log_params(self, params):
        """Record key-value parameters (dict or object with __dict__)."""
        flat = _flatten(params) if not isinstance(params, dict) else {
            k: str(v) for k, v in params.items()
        }
        for k, v in flat.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (self.id, k, v),
            )
        self._conn.commit()

    def log(self, step=None, epoch=None, **metrics):
        """Log one or more metrics at the given step and/or epoch.

        Args:
            step: training step (global step counter)
            epoch: training epoch (optional, for epoch-level metrics)
            **metrics: metric key-value pairs

        Examples::

            run.log(step=100, loss=0.5)              # step-level
            run.log(epoch=2, val_acc=0.91)            # epoch-level
            run.log(step=100, epoch=2, loss=0.5)      # both
        """
        for key, value in metrics.items():
            self._conn.execute(
                "INSERT INTO metrics (run_id, key, value, step, epoch) VALUES (?, ?, ?, ?, ?)",
                (self.id, key, float(value), step, epoch),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO latest_metrics (run_id, key, value, step, epoch) VALUES (?, ?, ?, ?, ?)",
                (self.id, key, float(value), step, epoch),
            )
        self._conn.commit()

    def finish(self, status="FINISHED"):
        """Mark the run as finished."""
        self._conn.execute(
            "UPDATE runs SET status = ?, end_time = datetime('now') WHERE id = ?",
            (status, self.id),
        )
        self._conn.commit()

    def fail(self):
        """Mark the run as failed."""
        self.finish(status="FAILED")

    def log_predictions(self, predictions, label_key="label", pred_key="prediction",
                        conf_key="confidence", id_key=None, text_key=None):
        """Log per-example predictions for disagreement analysis.

        Args:
            predictions: list of dicts, pandas DataFrame, or path to CSV/JSON file.
            label_key: column/key for ground truth label.
            pred_key: column/key for model prediction.
            conf_key: column/key for confidence (optional, defaults to 1.0).
            id_key: column/key for example ID (default: row index).
            text_key: column/key for input text (optional).

        Example::

            run.log_predictions([
                {"label": "Tech", "prediction": "Tech", "confidence": 0.94, "text": "Apple..."},
                {"label": "Biz",  "prediction": "Politics", "confidence": 0.58, "text": "Fed..."},
            ], text_key="text")
        """
        rows = self._normalize_predictions(predictions)
        batch = []
        for i, row in enumerate(rows):
            ex_id = str(row.get(id_key, i)) if id_key and id_key in row else str(i)
            label = str(row.get(label_key, ""))
            pred = str(row.get(pred_key, ""))
            conf = float(row.get(conf_key, 1.0)) if conf_key and conf_key in row else 1.0
            correct = 1 if pred == label else 0
            text = str(row.get(text_key, "")) if text_key and text_key in row else ""
            batch.append((self.id, ex_id, pred, conf, label, correct, text))
            if len(batch) >= 1000:
                self._insert_predictions_batch(batch)
                batch = []
        if batch:
            self._insert_predictions_batch(batch)

    def _insert_predictions_batch(self, batch):
        self._conn.executemany(
            "INSERT OR REPLACE INTO predictions "
            "(run_id, example_id, prediction, confidence, label, correct, input_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        self._conn.commit()

    @staticmethod
    def _normalize_predictions(predictions):
        """Convert predictions input to list of dicts."""
        if isinstance(predictions, (str, Path)):
            path = Path(predictions)
            if path.suffix == ".csv":
                import csv
                with open(path) as f:
                    return list(csv.DictReader(f))
            elif path.suffix == ".json":
                import json
                with open(path) as f:
                    return json.load(f)
        # pandas DataFrame
        if hasattr(predictions, "to_dict"):
            return predictions.to_dict("records")
        return list(predictions)


class GsqlTrack:
    """Experiment tracking entry point.

    Usage:
        t = GsqlTrack("my-experiment")
        run = t.start_run("run-name")
        run.log_params({"lr": 0.001})
        run.log(step=0, loss=0.5)
        run.finish()
    """

    def __init__(self, experiment, db_path=None):
        self._conn = _open_db(db_path)
        self._conn.execute(
            "INSERT OR IGNORE INTO experiments (name) VALUES (?)", (experiment,)
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id FROM experiments WHERE name = ?", (experiment,)
        ).fetchone()
        self._exp_id = row[0]
        self.experiment = experiment

    def log_experiment_metadata(self, metadata):
        """Store metadata on the experiment (command, models, tasks, etc.).

        Merges with existing metadata. Call once at the start of a benchmark/tune run.

        Args:
            metadata: dict of metadata to store

        Example::

            t = GsqlTrack("bench/my_bench")
            t.log_experiment_metadata({
                "command": "python -m src.common.bench_manager -c bench.yaml",
                "models": ["BERT_gold", "BERT_weak"],
                "tasks": ["AGNews", "CDR"],
                "n_seeds": 3,
                "primary_metric": "val.acc",
            })
        """
        import json
        # Read existing metadata and merge
        row = self._conn.execute(
            "SELECT metadata FROM experiments WHERE id = ?", (self._exp_id,)
        ).fetchone()
        existing = json.loads(row[0]) if row and row[0] else {}
        existing.update(metadata)
        self._conn.execute(
            "UPDATE experiments SET metadata = ? WHERE id = ?",
            (json.dumps(existing), self._exp_id),
        )
        self._conn.commit()

    def start_run(self, name="", source=None):
        """Start a new run and return a _GsqlRun handle."""
        run_id = uuid.uuid4().hex[:12]
        git_commit = _get_git_commit()
        if source is None:
            import inspect
            frame = inspect.stack()[1]
            source = os.path.basename(frame.filename)
        self._conn.execute(
            "INSERT INTO runs (id, experiment_id, name, git_commit, source) VALUES (?, ?, ?, ?, ?)",
            (run_id, self._exp_id, name, git_commit, source),
        )
        self._conn.commit()
        return _GsqlRun(self._conn, run_id)

    def log_completed_run(self, name="", params=None, metrics=None, status="FINISHED", source=None):
        """Create a run with all data at once (for async/batch workflows).

        Use this when results are collected after the run finishes — e.g.
        loading tune/bench results in bulk.

        Args:
            name: run name (e.g. "bert/sst2/seed_0")
            params: dict of hyperparameters
            metrics: dict of {key: value} for single-step results, or
                     list of dicts with "step" key for multi-step results
            status: final run status (default "FINISHED")
            source: source file label

        Returns:
            _GsqlRun handle (already finished)

        Example::

            t = GsqlTrack("bench/my_bench")
            # single-step (final results only)
            t.log_completed_run("bert/sst2/seed_0",
                params={"lr": 2e-5, "bs": 32},
                metrics={"acc": 0.923, "f1": 0.891})

            # multi-step (full history)
            t.log_completed_run("cnn/mnist/seed_0",
                metrics=[{"step": 0, "loss": 2.3}, {"step": 100, "loss": 0.1}])
            t.close()
        """
        run = self.start_run(name=name, source=source or "bulk_load")
        if params:
            run.log_params(params)
        if metrics:
            if isinstance(metrics, dict):
                # Single-step: log everything at step=0
                numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                if numeric:
                    run.log(step=0, **numeric)
            elif isinstance(metrics, list):
                # Multi-step: each entry must have "step" key
                for entry in metrics:
                    step = entry.get("step", 0)
                    numeric = {k: float(v) for k, v in entry.items()
                               if k != "step" and isinstance(v, (int, float))}
                    if numeric:
                        run.log(step=step, **numeric)
        run.finish(status=status)
        return run

    def close(self):
        """Close the database connection."""
        self._conn.close()


def tracked(tracker, experiment, run_name="", db_path=None):
    """Wrap an existing Tracker instance to auto-persist metrics.

    Monkey-patches log_epoch, log_step, and finalize to also write to
    ~/.gsql/track.db. The original methods are called first.

    Usage:
        tracker = tracked(Tracker(config), experiment="mnist")
        # then use tracker normally — metrics are auto-logged
    """
    t = GsqlTrack(experiment, db_path=db_path)
    run = t.start_run(name=run_name, source=None)

    # Try to capture config as params
    if hasattr(tracker, "config"):
        cfg = tracker.config
        try:
            # OmegaConf support
            from omegaconf import OmegaConf
            params = OmegaConf.to_container(cfg, resolve=True)
            run.log_params(_flatten(params))
        except Exception:
            flat = _flatten(cfg)
            if flat:
                run.log_params(flat)

    # Patch log_epoch
    orig_log_epoch = getattr(tracker, "log_epoch", None)
    if orig_log_epoch is not None:
        def patched_log_epoch(*args, **kwargs):
            result = orig_log_epoch(*args, **kwargs)
            epoch = getattr(tracker, "epoch", 0)
            step = getattr(tracker, "step", None)
            # Collect numeric metrics from tracker state
            metrics = {}
            for attr in ("train_loss", "valid_loss", "train_acc", "valid_acc"):
                val = getattr(tracker, attr, None)
                if val is not None and isinstance(val, (int, float)):
                    metrics[attr] = float(val)
            # Also log any kwargs that are numeric
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            if metrics:
                run.log(epoch=epoch, step=step, **metrics)
            return result
        tracker.log_epoch = patched_log_epoch

    # Patch log_step
    orig_log_step = getattr(tracker, "log_step", None)
    if orig_log_step is not None:
        def patched_log_step(*args, **kwargs):
            result = orig_log_step(*args, **kwargs)
            step = getattr(tracker, "step", 0)
            metrics = {}
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            if metrics:
                run.log(step=step, **metrics)
            return result
        tracker.log_step = patched_log_step

    # Patch finalize
    orig_finalize = getattr(tracker, "finalize", None)
    if orig_finalize is not None:
        def patched_finalize(*args, **kwargs):
            result = orig_finalize(*args, **kwargs)
            run.finish()
            return result
        tracker.finalize = patched_finalize

    # Stash references for manual access
    tracker._gsql_track = t
    tracker._gsql_run = run

    return tracker
