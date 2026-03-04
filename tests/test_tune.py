"""Tests for gsql_track.tune — sample_config, TuneJob, TuneResults, extract_tune_space."""
import pytest
import optuna

from gsql_track.tune import (
    sample_config,
    extract_tune_space,
    extract_tune_key,
    sample_random_params,
    JobResult,
    TuneResults,
    _fmt_val,
)


# ─── sample_config ──────────────────────────────────────────────────────────

class TestSampleConfig:
    def _trial(self):
        study = optuna.create_study(direction="maximize")
        return study.ask()

    def test_scalar_passthrough(self):
        trial = self._trial()
        assert sample_config(trial, 42, ["x"]) == 42
        assert sample_config(trial, "hello", ["x"]) == "hello"
        assert sample_config(trial, True, ["x"]) is True
        assert sample_config(trial, None, ["x"]) is None

    def test_range(self):
        trial = self._trial()
        val = sample_config(trial, ["range", 0.0, 1.0], ["lr"])
        assert 0.0 <= val <= 1.0

    def test_range_with_step(self):
        trial = self._trial()
        val = sample_config(trial, ["range", 0.0, 1.0, 0.1], ["dropout"])
        assert 0.0 <= val <= 1.0

    def test_range_log(self):
        trial = self._trial()
        val = sample_config(trial, ["range", "log", 1e-5, 1e-1], ["lr"])
        assert 1e-5 <= val <= 1e-1

    def test_choice(self):
        trial = self._trial()
        val = sample_config(trial, ["choice", "adam", "sgd"], ["optimizer"])
        assert val in ("adam", "sgd")

    def test_int(self):
        trial = self._trial()
        val = sample_config(trial, ["int", 16, 256], ["batch_size"])
        assert 16 <= val <= 256
        assert isinstance(val, int)

    def test_int_log(self):
        trial = self._trial()
        val = sample_config(trial, ["int", "log", 8, 1024], ["hidden"])
        assert 8 <= val <= 1024

    def test_nested_dict(self):
        trial = self._trial()
        space = {
            "lr": ["range", 1e-4, 1e-2],
            "layers": {"hidden": ["int", 64, 512]},
        }
        result = sample_config(trial, space, [])
        assert "lr" in result
        assert "layers" in result
        assert "hidden" in result["layers"]

    def test_optional_param(self):
        trial = self._trial()
        val = sample_config(trial, ["?range", 0.0, 0.1, 0.5], ["wd"])
        assert isinstance(val, float)

    def test_empty_list(self):
        trial = self._trial()
        assert sample_config(trial, [], ["x"]) == []

    def test_float_dist(self):
        trial = self._trial()
        val = sample_config(trial, ["float", 0.0, 1.0], ["x"])
        assert 0.0 <= val <= 1.0

    def test_log_dist(self):
        trial = self._trial()
        val = sample_config(trial, ["log", 1e-5, 1.0], ["x"])
        assert 1e-5 <= val <= 1.0

    def test_invalid_range_args(self):
        trial = self._trial()
        with pytest.raises(ValueError, match="range requires"):
            sample_config(trial, ["range", 1, 2, 3, 4], ["x"])

    def test_choice_empty(self):
        trial = self._trial()
        with pytest.raises(ValueError, match="choice requires"):
            sample_config(trial, ["choice"], ["x"])

    def test_unsupported_type(self):
        trial = self._trial()
        with pytest.raises(TypeError):
            sample_config(trial, object(), ["x"])


# ─── extract_tune_space ─────────────────────────────────────────────────────

class TestExtractTuneSpace:
    def test_flat(self):
        config = {
            "lr": ["range", 0.001, 0.1],
            "layers": 128,
            "dropout": ["choice", 0.1, 0.5],
        }
        space = extract_tune_space(config)
        assert "lr" in space
        assert "dropout" in space
        assert "layers" not in space

    def test_nested(self):
        config = {
            "model": {
                "lr": ["range", 0.001, 0.1],
                "hidden": 256,
            }
        }
        space = extract_tune_space(config)
        assert "model" in space
        assert "lr" in space["model"]


class TestExtractTuneKey:
    def test_flat(self):
        keys = extract_tune_key({"lr": ["range", 0.001, 0.1], "x": 5})
        assert keys == ["lr"]

    def test_nested(self):
        keys = extract_tune_key({"model": {"lr": ["range", 0.001, 0.1]}})
        assert keys == ["model.lr"]


# ─── sample_random_params ───────────────────────────────────────────────────

class TestSampleRandomParams:
    def test_deterministic(self):
        space = {"lr": ["range", 0.001, 0.1], "bs": ["choice", 16, 32, 64]}
        p1 = sample_random_params(space, seed=42)
        p2 = sample_random_params(space, seed=42)
        assert p1 == p2


# ─── Helpers ────────────────────────────────────────────────────────────────

class TestFmtVal:
    def test_float(self):
        assert _fmt_val(0.1234) == "0.1234"

    def test_none(self):
        assert _fmt_val(None) == "-"
