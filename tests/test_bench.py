"""Tests for gsql_track.bench — BenchmarkResult, BenchmarkResults, load_best_config."""
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from gsql_track.bench import (
    BenchmarkResult,
    BenchmarkResults,
    BenchmarkConfig,
    ModelSpec,
    BenchmarkProgressTracker,
    load_best_config,
)
from gsql_track import util as U


# ─── BenchmarkResult ────────────────────────────────────────────────────────

class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult("bert", "sst2", 1, {"acc": 0.9, "f1": 0.85}, time_spent=10.0)
        d = r.to_dict()
        assert d["model"] == "bert"
        assert d["seed"] == 1
        assert d["acc"] == 0.9
        assert d["time_spent"] == 10.0

    def test_to_dict_no_time(self):
        r = BenchmarkResult("mv", "yelp", 2, {"acc": 0.7})
        d = r.to_dict()
        assert "time_spent" not in d


# ─── BenchmarkResults ───────────────────────────────────────────────────────

class TestBenchmarkResults:
    def _make_results(self):
        return [
            BenchmarkResult("bert", "sst2", 1, {"acc": 0.90}),
            BenchmarkResult("bert", "sst2", 2, {"acc": 0.92}),
            BenchmarkResult("bert", "sst2", 3, {"acc": 0.88}),
            BenchmarkResult("mv", "sst2", 1, {"acc": 0.70}),
            BenchmarkResult("mv", "sst2", 2, {"acc": 0.72}),
        ]

    def test_aggregate(self):
        br = BenchmarkResults(self._make_results())
        agg = br.aggregate()
        assert len(agg) == 2  # bert, mv
        bert_row = agg[agg["model"] == "bert"].iloc[0]
        assert abs(bert_row["acc_mean"] - 0.9) < 0.01

    def test_empty_results(self):
        br = BenchmarkResults([])
        assert br.df.empty
        assert br.display_metrics == set()

    def test_to_csv(self, tmp_path):
        br = BenchmarkResults(self._make_results())
        csv_path = tmp_path / "results.csv"
        br.to_csv(csv_path, aggregated=False)
        assert csv_path.exists()
        br.to_csv(tmp_path / "agg.csv", aggregated=True)
        assert (tmp_path / "agg.csv").exists()

    def test_repr(self):
        br = BenchmarkResults(self._make_results())
        r = repr(br)
        assert "bert" in r


# ─── load_best_config ───────────────────────────────────────────────────────

class TestLoadBestConfig:
    def test_load(self, tmp_path):
        import yaml
        config = {"lr": 0.001, "batch_size": 32, "output": "/tmp", "seed": 42}
        config_path = tmp_path / "best.yaml"
        config_path.write_text(yaml.dump(config))
        loaded = load_best_config(str(config_path))
        assert loaded["lr"] == 0.001
        assert "output" not in loaded  # ignored by default
        assert "seed" not in loaded

    def test_load_with_base(self, tmp_path):
        import yaml
        config = {"lr": 0.01}
        config_path = tmp_path / "best.yaml"
        config_path.write_text(yaml.dump(config))
        base = {"lr": 0.001, "epochs": 10}
        loaded = load_best_config(str(config_path), base_config=base)
        assert loaded["lr"] == 0.01  # overridden
        assert loaded["epochs"] == 10  # preserved

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_best_config("/nonexistent/path.yaml")


# ─── BenchmarkProgressTracker ───────────────────────────────────────────────

class TestBenchmarkProgressTracker:
    def test_update_and_finalize(self):
        writer = U.ProgressWriter(silent=True)
        tracker = BenchmarkProgressTracker(3, writer=writer)
        tracker.update("completed", success=True, current_task="bert on sst2")
        assert tracker.completed == 1
        tracker.update("failed", success=False)
        assert tracker.failed == 1
        elapsed = tracker.finalize()
        assert elapsed > 0
