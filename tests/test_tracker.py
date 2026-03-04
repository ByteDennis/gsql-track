"""Tests for gsql_track.tracker — ModelSerializer, TrainingState, TuningState, SummaryReport."""
import json
import pickle
import tempfile
from pathlib import Path

import pytest

from gsql_track.tracker import (
    ModelSerializer,
    TrainingState,
    TuningState,
    SummaryReport,
    Checkpoints,
)
from gsql_track.enums import TrainingStatus


# ─── TrainingState ──────────────────────────────────────────────────────────

class TestTrainingState:
    def test_defaults(self):
        s = TrainingState()
        assert s.status == TrainingStatus.train
        assert s.current_epoch == 0
        assert s.current_step == 0
        assert s.best_metrics == {}

    def test_init_current_from_start(self):
        s = TrainingState(start_epoch=5, start_step=100)
        assert s.current_epoch == 5
        assert s.current_step == 100

    def test_explicit_current_overrides_start(self):
        s = TrainingState(start_epoch=5, current_epoch=10)
        assert s.current_epoch == 10

    def test_model_dump_roundtrip(self):
        s = TrainingState(current_epoch=3, best_metrics={"acc": 0.9})
        d = s.model_dump()
        s2 = TrainingState(**d)
        assert s2.current_epoch == 3
        assert s2.best_metrics["acc"] == 0.9

    def test_repr(self):
        s = TrainingState(current_epoch=5, best_metric_value=0.95)
        r = repr(s)
        assert "epoch=0=>5" in r
        assert "0.95" in r


# ─── TuningState ────────────────────────────────────────────────────────────

class TestTuningState:
    def test_defaults(self):
        s = TuningState()
        assert s.direction == "maximize"
        assert s.completed_trials == 0

    def test_repr_with_trials(self):
        s = TuningState(n_trials=50, completed_trials=10, best_value=0.85)
        r = repr(s)
        assert "10/50" in r
        assert "0.8500" in r

    def test_repr_no_total(self):
        s = TuningState(completed_trials=5)
        assert "5" in repr(s)


# ─── SummaryReport ──────────────────────────────────────────────────────────

class TestSummaryReport:
    def test_creation(self):
        s = SummaryReport(
            eval_config={"metric": "acc"},
            output_config={"folder": "/tmp"},
            model_info={"name": "bert"},
            execute_info={"time_per_epoch": 1.0},
            result_info={"best_epoch": 5},
        )
        assert s.model_info["name"] == "bert"
        d = s.model_dump()
        assert d["result_info"]["best_epoch"] == 5


# ─── ModelSerializer ────────────────────────────────────────────────────────

class TestModelSerializer:
    def test_save_load_dict_json(self, tmp_path):
        data = {"lr": 0.001, "batch_size": 64}
        path = tmp_path / "config"
        ModelSerializer.save(data, path)
        loaded = ModelSerializer.load(path)
        assert loaded == data

    def test_save_load_pkl(self, tmp_path):
        data = [1, 2, 3]
        path = tmp_path / "data"
        ModelSerializer.save(data, path)
        loaded = ModelSerializer.load(path)
        assert loaded == data

    def test_save_with_ext(self, tmp_path):
        data = {"a": 1}
        path = tmp_path / "config"
        ModelSerializer.save(data, path, ext="json")
        assert (tmp_path / "config.json").exists()
        loaded = ModelSerializer.load(tmp_path / "config.json")
        assert loaded == data

    def test_save_tuple_delegates(self, tmp_path):
        """Saving a (obj, ext) tuple delegates to save with ext."""
        data = {"x": 42}
        path = tmp_path / "test"
        ModelSerializer.save((data, "json"), path)
        assert (tmp_path / "test.json").exists()

    def test_load_auto_detect(self, tmp_path):
        """Load auto-detects extension when none given."""
        data = {"key": "value"}
        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps(data))
        loaded = ModelSerializer.load(tmp_path / "data")
        assert loaded == data

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(Exception):
            ModelSerializer.load(tmp_path / "nonexistent.json")

    def test_rng_state_roundtrip(self):
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        state = ModelSerializer._capture_rng_state()
        assert 'python' in state
        assert 'numpy' in state
        # Restore should not raise
        ModelSerializer._restore_rng_state(state)
