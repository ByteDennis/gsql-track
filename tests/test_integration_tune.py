"""Integration tests for Tuner (Issue #6)."""
import shutil
from pathlib import Path

import pytest
import optuna

from gsql_track.tune import TuneConfig, ModelTuneSpec, Tuner, extract_tune_space
from gsql_track.config import TaskConfig

DUMMY_FUNCTION_PATH = "tests.fixtures.dummy_train_fn"
OUTPUT_DIR = Path("/tmp/gsql_test_tune")


@pytest.fixture(autouse=True, scope="module")
def setup_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _make_tune_config(n_trials=10, n_workers=1, resume_mode="fresh"):
    return TuneConfig(
        models=[
            ModelTuneSpec(
                name="model_a",
                function=DUMMY_FUNCTION_PATH,
                train_config={
                    "lr": ["log", 0.001, 0.1],
                    "batch_size": ["choice", 8, 16, 32],
                    "eval": {"primary_metric": "val.acc", "direction": "maximize"},
                },
            ),
        ],
        tasks=[TaskConfig(name="task_x")],
        output=OUTPUT_DIR,
        n_trials=n_trials,
        timeout=300,
        metric="val.acc",
        direction="maximize",
        n_workers=n_workers,
        show_progress=False,
        auto_confirm=True,
        resume_mode=resume_mode,
    )


# ─── CPU Tests ─────────────────────────────────────────────────────────────


class TestTuneSequential:
    def test_run_sequential(self):
        tuner = Tuner(_make_tune_config(n_trials=10))
        results = tuner.run_sequential()

        assert len(results.results) == 1
        result = results.results[0]
        assert result.model == "model_a"
        assert result.task == "task_x"
        assert result.n_trials == 10
        assert result.best_value > 0

        # Optuna study accessible
        study_path = OUTPUT_DIR / "studies" / "model_a_task_x.db"
        assert study_path.exists()
        study = optuna.load_study(study_name="model_a_task_x", storage=f"sqlite:///{study_path}")
        assert len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 10

        # track.db alongside tuning.db
        assert (OUTPUT_DIR / "track.db").exists()
        assert (OUTPUT_DIR / "tuning.db").exists()


class TestTuneConfigExtraction:
    def test_extract_flat(self):
        space = extract_tune_space({"lr": ["log", 0.001, 0.1], "batch_size": ["choice", 8, 16], "hidden": 256})
        assert "lr" in space and "batch_size" in space and "hidden" not in space

    def test_extract_nested(self):
        space = extract_tune_space({"model": {"lr": ["range", 0.001, 0.1], "layers": {"hidden": ["int", 64, 512]}}, "seed": 42})
        assert "model" in space and "hidden" in space["model"]["layers"]


# ─── GPU Variant ───────────────────────────────────────────────────────────


@pytest.mark.gpu
class TestTuneGPUParallel:
    def test_parallel_tuning(self):
        config = TuneConfig(
            models=[ModelTuneSpec(name="model_a", function=DUMMY_FUNCTION_PATH,
                                   train_config={"lr": ["log", 0.001, 0.1], "batch_size": ["choice", 8, 16],
                                                  "eval": {"primary_metric": "val.acc", "direction": "maximize"}})],
            tasks=[TaskConfig(name="task_x")],
            output=OUTPUT_DIR,
            n_trials=20, timeout=300, metric="val.acc", direction="maximize",
            n_workers=4, device_ids=[0, 1],
            show_progress=False, auto_confirm=True, resume_mode="fresh",
        )
        results = Tuner(config).run_parallel()
        assert len(results.results) == 1
