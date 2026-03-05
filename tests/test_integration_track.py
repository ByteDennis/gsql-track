"""Integration tests for Tracker and GsqlTrack lifecycle (Issue #4)."""
import concurrent.futures
import shutil
import sqlite3
from pathlib import Path

import pytest

from gsql_track.tracker import Tracker
from gsql_track.gsql_track import GsqlTrack
from tests.fixtures import make_train_config

OUTPUT_DIR = Path("/tmp/gsql_test_track")


@pytest.fixture(autouse=True, scope="module")
def setup_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


# ─── CPU Tests ─────────────────────────────────────────────────────────────


class TestTrackerEpochBased:
    def test_tracker_epoch_based(self):
        config = make_train_config(OUTPUT_DIR / "epoch_based", tracking_mode="epochs", n_total=10)
        tracker = Tracker(config=config)

        for epoch in range(10):
            loss = 1.0 - (epoch * 0.08)
            acc = 0.5 + (epoch * 0.04)
            tracker.log_epoch({"val.acc": acc, "val.loss": loss}, epoch=epoch)
            if tracker.should_stop:
                break

        tracker.finalize()
        assert tracker.best_metrics and "val.acc" in tracker.best_metrics
        assert tracker.best_epoch >= 0


class TestTrackerStepBased:
    def test_tracker_step_based(self):
        config = make_train_config(
            OUTPUT_DIR / "step_based", tracking_mode="steps", n_total=100,
            primary_metric="val.loss", direction="minimize",
        )
        config.pbar.steps_per_epoch = 10
        tracker = Tracker(config=config)

        for step in range(0, 100, 10):
            loss = 1.0 - (step * 0.008)
            acc = 0.5 + (step * 0.004)
            tracker.log_step({"val.loss": loss, "val.acc": acc}, step=step)
            if tracker.should_stop:
                break

        tracker.finalize()
        assert tracker.best_metrics and "val.loss" in tracker.best_metrics


class TestTrackerEarlyStop:
    def test_early_stopping(self):
        config = make_train_config(OUTPUT_DIR / "early_stop", tracking_mode="epochs", n_total=20)
        tracker = Tracker(config=config)

        for epoch in range(20):
            acc = 0.5 + epoch * 0.05 if epoch < 5 else 0.5
            tracker.log_epoch({"val.acc": acc, "val.loss": 1.0 - acc}, epoch=epoch)
            if tracker.should_stop:
                break

        assert tracker.epoch < 20
        assert tracker.best_epoch <= 5


class TestTrackerEvaluateFunction:
    def test_evaluate_function(self):
        config = make_train_config(OUTPUT_DIR / "evaluate", tracking_mode="epochs", n_total=5)
        tracker = Tracker(config=config)
        call_count = [0]
        dummy_dataset = [1, 2, 3]

        def evaluate_fn(dataset=None, **kwargs):
            call_count[0] += 1
            return {"val.acc": 0.8 + call_count[0] * 0.01, "val.loss": 0.5}

        tracker.add_evaluate_function(evaluate_fn, dataset=dummy_dataset)
        for epoch in range(5):
            result = tracker.evaluate_data()
            tracker.log_epoch(result, epoch=epoch)

        tracker.finalize()
        assert call_count[0] == 5


class TestGsqlTrackAPI:
    def test_full_lifecycle(self):
        db_path = str(OUTPUT_DIR / "track.db")
        t = GsqlTrack("test-experiment", db_path=db_path)
        run = t.start_run("run-001")
        run.log_params({"lr": 0.001, "batch_size": 64, "model": "bert"})
        for step in range(5):
            run.log(step=step, loss=1.0 - step * 0.15, accuracy=0.5 + step * 0.1)
        run.log_predictions([
            {"label": "A", "prediction": "A", "confidence": 0.9, "text": "example 1"},
            {"label": "B", "prediction": "A", "confidence": 0.6, "text": "example 2"},
        ], text_key="text")
        run.finish()
        t.close()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT status FROM runs WHERE id = ?", (run.id,)).fetchone()["status"] == "FINISHED"
        params = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM params WHERE run_id = ?", (run.id,)).fetchall()}
        assert params["lr"] == "0.001"
        assert conn.execute("SELECT COUNT(*) FROM metrics WHERE run_id = ?", (run.id,)).fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (run.id,)).fetchone()[0] == 2
        conn.close()


# ─── Experiment Detail Page Data ───────────────────────────────────────────

class TestExperimentWithSeedGroups:
    """Creates multi-group experiment data with seed variants for testing the experiment detail page."""

    def test_seed_groups(self):
        import random
        db_path = str(OUTPUT_DIR / "track.db")

        groups = [
            ("model_a/task_x", 3),
            ("model_a/task_y", 3),
            ("model_b/task_x", 2),
        ]

        for group_name, n_seeds in groups:
            for seed in range(n_seeds):
                random.seed(seed)
                t = GsqlTrack("bench-experiment", db_path=db_path)
                run = t.start_run(f"{group_name}/seed_{seed}")
                run.log_params({"model": group_name.split("/")[0], "task": group_name.split("/")[1], "seed": str(seed)})

                base_acc = 0.7 + random.uniform(-0.05, 0.05)
                base_loss = 0.6 + random.uniform(-0.1, 0.1)
                for epoch in range(20):
                    noise = random.uniform(-0.02, 0.02)
                    acc = base_acc + epoch * 0.012 + noise
                    loss = base_loss - epoch * 0.02 + noise
                    run.log(step=epoch, **{"val.acc": acc, "val.loss": loss, "val.f1": acc * 0.95 + noise})

                run.finish()
                t.close()

        # Also add an ungrouped run
        t = GsqlTrack("bench-experiment", db_path=db_path)
        run = t.start_run("baseline-single")
        run.log_params({"model": "baseline", "task": "all"})
        for epoch in range(20):
            run.log(step=epoch, **{"val.acc": 0.65 + epoch * 0.005, "val.loss": 0.8 - epoch * 0.01})
        run.finish()
        t.close()

        # Verify data
        conn = sqlite3.connect(db_path)
        exp_id = conn.execute("SELECT id FROM experiments WHERE name = 'bench-experiment'").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM runs WHERE experiment_id = ?", (exp_id,)).fetchone()[0]
        assert count == 9  # 3+3+2+1
        conn.close()


class TestMultiExperiment:
    """Creates a second experiment to ensure experiment filtering works."""

    def test_separate_experiment(self):
        db_path = str(OUTPUT_DIR / "track.db")

        for seed in range(3):
            t = GsqlTrack("other-experiment", db_path=db_path)
            run = t.start_run(f"config_a/seed_{seed}")
            run.log_params({"lr": "0.01", "seed": str(seed)})
            for step in range(10):
                run.log(step=step, **{"val.acc": 0.5 + step * 0.04 + seed * 0.01})
            run.finish()
            t.close()


# ─── GPU Variant ───────────────────────────────────────────────────────────


@pytest.mark.gpu
class TestTrackGPUAsync:
    def test_concurrent_writes(self):
        db_path = str(OUTPUT_DIR / "track.db")

        def run_experiment(worker_id):
            t = GsqlTrack("concurrent-test", db_path=db_path)
            run = t.start_run(f"worker-{worker_id}")
            run.log_params({"worker": worker_id, "lr": 0.001 * worker_id})
            for step in range(10):
                run.log(step=step, loss=1.0 / (step + 1), acc=step * 0.1)
            run.finish()
            t.close()
            return run.id

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            run_ids = [f.result() for f in [pool.submit(run_experiment, i) for i in range(4)]]

        assert len(set(run_ids)) == 17
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'FINISHED'").fetchone()[0] == 4
        for rid in run_ids:
            assert conn.execute("SELECT COUNT(*) FROM metrics WHERE run_id = ?", (rid,)).fetchone()[0] == 20
        conn.close()
