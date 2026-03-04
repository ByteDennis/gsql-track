"""Tests for GsqlTrack and _GsqlRun core API."""
import json
import sqlite3
import csv
from unittest.mock import MagicMock

import pytest
from gsql_track.gsql_track import GsqlTrack, tracked, _flatten


# ── Experiment lifecycle ──

def test_create_experiment(tracker):
    row = tracker._conn.execute(
        "SELECT name FROM experiments WHERE name = ?", ("test-exp",)
    ).fetchone()
    assert row[0] == "test-exp"


def test_duplicate_experiment_reuses_id(tmp_db):
    t1 = GsqlTrack("dup-exp", db_path=tmp_db)
    id1 = t1._exp_id
    t2 = GsqlTrack("dup-exp", db_path=tmp_db)
    id2 = t2._exp_id
    assert id1 == id2
    t1.close()
    t2.close()


# ── Run lifecycle ──

def test_start_run_creates_row(tracker):
    run = tracker.start_run("run-1")
    row = tracker._conn.execute(
        "SELECT status, name FROM runs WHERE id = ?", (run.id,)
    ).fetchone()
    assert row[0] == "RUNNING"
    assert row[1] == "run-1"


def test_finish_run(sample_run, tracker):
    sample_run.finish()
    row = tracker._conn.execute(
        "SELECT status, end_time FROM runs WHERE id = ?", (sample_run.id,)
    ).fetchone()
    assert row[0] == "FINISHED"
    assert row[1] is not None


def test_fail_run(tracker):
    run = tracker.start_run("fail-run")
    run.fail()
    row = tracker._conn.execute(
        "SELECT status FROM runs WHERE id = ?", (run.id,)
    ).fetchone()
    assert row[0] == "FAILED"


# ── log_params ──

def test_log_params_dict(sample_run, tracker):
    rows = tracker._conn.execute(
        "SELECT key, value FROM params WHERE run_id = ? ORDER BY key",
        (sample_run.id,)
    ).fetchall()
    params = {r[0]: r[1] for r in rows}
    assert params["lr"] == "0.001"
    assert params["bs"] == "64"


def test_log_params_object(tracker):
    run = tracker.start_run("obj-params")

    class Cfg:
        def __init__(self):
            self.lr = 0.01
            self.hidden = 256

    run.log_params(Cfg())
    rows = tracker._conn.execute(
        "SELECT key, value FROM params WHERE run_id = ?", (run.id,)
    ).fetchall()
    params = {r[0]: r[1] for r in rows}
    assert params["lr"] == "0.01"
    assert params["hidden"] == "256"


# ── log metrics ──

def test_log_step_metrics(sample_run, tracker):
    sample_run.log(step=0, loss=0.5, acc=0.8)
    sample_run.log(step=1, loss=0.3, acc=0.9)
    rows = tracker._conn.execute(
        "SELECT key, value, step FROM metrics WHERE run_id = ? ORDER BY step, key",
        (sample_run.id,)
    ).fetchall()
    assert len(rows) == 4  # 2 metrics × 2 steps


def test_log_epoch_metrics(sample_run, tracker):
    sample_run.log(epoch=1, val_acc=0.91)
    row = tracker._conn.execute(
        "SELECT epoch, value FROM metrics WHERE run_id = ? AND key = 'val_acc'",
        (sample_run.id,)
    ).fetchone()
    assert row[0] == 1
    assert row[1] == pytest.approx(0.91)


def test_latest_metrics_updated(sample_run, tracker):
    sample_run.log(step=0, loss=0.5)
    sample_run.log(step=1, loss=0.3)
    row = tracker._conn.execute(
        "SELECT value, step FROM latest_metrics WHERE run_id = ? AND key = 'loss'",
        (sample_run.id,)
    ).fetchone()
    assert row[0] == pytest.approx(0.3)
    assert row[1] == 1


# ── log_predictions ──

def test_log_predictions_list_of_dicts(sample_run, tracker):
    preds = [
        {"label": "A", "prediction": "A", "confidence": 0.9},
        {"label": "B", "prediction": "A", "confidence": 0.6},
    ]
    sample_run.log_predictions(preds)
    rows = tracker._conn.execute(
        "SELECT example_id, prediction, correct FROM predictions WHERE run_id = ? ORDER BY example_id",
        (sample_run.id,)
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][2] == 1  # correct
    assert rows[1][2] == 0  # incorrect


def test_log_predictions_csv(sample_run, tracker, tmp_path):
    csv_path = tmp_path / "preds.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "prediction", "confidence"])
        w.writeheader()
        w.writerow({"label": "X", "prediction": "X", "confidence": "0.95"})
    sample_run.log_predictions(str(csv_path))
    rows = tracker._conn.execute(
        "SELECT correct FROM predictions WHERE run_id = ?", (sample_run.id,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1


def test_log_predictions_json(sample_run, tracker, tmp_path):
    json_path = tmp_path / "preds.json"
    json_path.write_text(json.dumps([
        {"label": "Y", "prediction": "Z", "confidence": 0.4}
    ]))
    sample_run.log_predictions(str(json_path))
    rows = tracker._conn.execute(
        "SELECT correct FROM predictions WHERE run_id = ?", (sample_run.id,)
    ).fetchall()
    assert rows[0][0] == 0


# ── log_completed_run ──

def test_log_completed_run_single_step(tracker):
    run = tracker.log_completed_run(
        "bulk-run", params={"lr": 0.01}, metrics={"acc": 0.95, "f1": 0.88}
    )
    row = tracker._conn.execute(
        "SELECT status FROM runs WHERE id = ?", (run.id,)
    ).fetchone()
    assert row[0] == "FINISHED"
    metrics = tracker._conn.execute(
        "SELECT key, value FROM latest_metrics WHERE run_id = ? ORDER BY key",
        (run.id,)
    ).fetchall()
    assert {r[0]: r[1] for r in metrics} == pytest.approx({"acc": 0.95, "f1": 0.88})


def test_log_completed_run_multi_step(tracker):
    run = tracker.log_completed_run(
        "multi-step",
        metrics=[{"step": 0, "loss": 2.3}, {"step": 100, "loss": 0.1}],
    )
    rows = tracker._conn.execute(
        "SELECT step, value FROM metrics WHERE run_id = ? ORDER BY step",
        (run.id,)
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][1] == pytest.approx(2.3)
    assert rows[1][1] == pytest.approx(0.1)


# ── log_experiment_metadata ──

def test_log_experiment_metadata(tracker):
    tracker.log_experiment_metadata({"models": ["BERT"], "n_seeds": 3})
    tracker.log_experiment_metadata({"tasks": ["SST2"]})  # merge
    row = tracker._conn.execute(
        "SELECT metadata FROM experiments WHERE id = ?", (tracker._exp_id,)
    ).fetchone()
    meta = json.loads(row[0])
    assert meta["models"] == ["BERT"]
    assert meta["tasks"] == ["SST2"]
    assert meta["n_seeds"] == 3


# ── tracked() wrapper ──

def test_tracked_patches_methods(tmp_db):
    mock_tracker = MagicMock()
    mock_tracker.config = {"lr": 0.001}
    mock_tracker.log_epoch = MagicMock(return_value=None)
    mock_tracker.log_step = MagicMock(return_value=None)
    mock_tracker.finalize = MagicMock(return_value=None)
    mock_tracker.epoch = 1
    mock_tracker.step = 10

    result = tracked(mock_tracker, experiment="wrap-test", db_path=tmp_db)
    assert result is mock_tracker
    assert hasattr(result, "_gsql_track")
    assert hasattr(result, "_gsql_run")

    # Call patched methods
    result.log_epoch()
    result.finalize()

    # Verify run was finished
    conn = sqlite3.connect(tmp_db)
    row = conn.execute("SELECT status FROM runs").fetchone()
    assert row[0] == "FINISHED"
    conn.close()


# ── Edge cases ──

def test_finish_already_finished_run(sample_run, tracker):
    sample_run.finish()
    sample_run.finish()  # should not raise
    row = tracker._conn.execute(
        "SELECT status FROM runs WHERE id = ?", (sample_run.id,)
    ).fetchone()
    assert row[0] == "FINISHED"


def test_concurrent_runs(tracker):
    r1 = tracker.start_run("run-a")
    r2 = tracker.start_run("run-b")
    r1.log(step=0, loss=1.0)
    r2.log(step=0, loss=2.0)
    r1.finish()
    r2.fail()
    s1 = tracker._conn.execute("SELECT status FROM runs WHERE id = ?", (r1.id,)).fetchone()[0]
    s2 = tracker._conn.execute("SELECT status FROM runs WHERE id = ?", (r2.id,)).fetchone()[0]
    assert s1 == "FINISHED"
    assert s2 == "FAILED"


# ── _flatten helper ──

def test_flatten_nested_dict():
    result = _flatten({"a": {"b": 1, "c": 2}, "d": 3})
    assert result == {"a.b": "1", "a.c": "2", "d": "3"}


def test_flatten_object():
    class Obj:
        def __init__(self):
            self.x = 10
            self.y = 20

    result = _flatten(Obj())
    assert "x" in result
    assert result["x"] == "10"
