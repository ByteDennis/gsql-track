"""Integration tests for BenchmarkRunner (Issue #5)."""
import shutil
from pathlib import Path

import pytest

from gsql_track.bench import BenchmarkConfig, BenchmarkRunner, ModelSpec
from gsql_track.config import TaskConfig
from tests.fixtures import DUMMY_FUNCTION_PATH

OUTPUT_DIR = Path("/tmp/gsql_test_bench")


@pytest.fixture(autouse=True, scope="module")
def setup_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _make_bench_config(n_seeds=2, n_workers=1, resume_mode="fresh"):
    return BenchmarkConfig(
        models=[
            ModelSpec(name="model_a", function=DUMMY_FUNCTION_PATH, n_seeds=n_seeds,
                      train_config={"eval": {"primary_metric": "val.acc", "direction": "maximize"}}),
            ModelSpec(name="model_b", function=DUMMY_FUNCTION_PATH, n_seeds=n_seeds,
                      train_config={"eval": {"primary_metric": "val.acc", "direction": "maximize"}}),
        ],
        tasks=[TaskConfig(name="task_x"), TaskConfig(name="task_y")],
        output=OUTPUT_DIR,
        n_workers=n_workers,
        show_progress=False,
        auto_confirm=True,
        resume_mode=resume_mode,
    )


# ─── CPU Tests ─────────────────────────────────────────────────────────────


class TestBenchSequential:
    def test_run_sequential(self):
        config = _make_bench_config(n_seeds=2)
        runner = BenchmarkRunner(config)
        results = runner.run_sequential()

        assert len(results.results) == 8
        assert set(results.df["model"].unique()) == {"model_a", "model_b"}
        assert set(results.df["task"].unique()) == {"task_x", "task_y"}

        # DONE files
        for model in ("model_a", "model_b"):
            for task in ("task_x", "task_y"):
                for seed in (1, 2):
                    assert (OUTPUT_DIR / model / task / f"seed_{seed}" / "DONE").exists()

        # track.db should exist alongside benchmark.db
        assert (OUTPUT_DIR / "track.db").exists()
        assert (OUTPUT_DIR / "benchmark.db").exists()


class TestBenchResumeMode:
    def test_resume_skips(self):
        runner = BenchmarkRunner(_make_bench_config(n_seeds=2))
        assert len(runner.run_sequential().results) == 8

        runner2 = BenchmarkRunner(_make_bench_config(n_seeds=2, resume_mode="resume"))
        assert len(runner2.run_sequential().results) == 8


class TestBenchFreshMode:
    def test_fresh_reruns(self):
        runner = BenchmarkRunner(_make_bench_config(n_seeds=1))
        assert len(runner.run_sequential().results) == 4

        runner2 = BenchmarkRunner(_make_bench_config(n_seeds=1, resume_mode="fresh"))
        assert len(runner2.run_sequential().results) == 4


# ─── GPU Variant ───────────────────────────────────────────────────────────


@pytest.mark.gpu
class TestBenchGPUParallel:
    def test_parallel_execution(self):
        config = BenchmarkConfig(
            models=[
                ModelSpec(name="model_a", function=DUMMY_FUNCTION_PATH, n_seeds=3,
                          train_config={"eval": {"primary_metric": "val.acc", "direction": "maximize"}}),
                ModelSpec(name="model_b", function=DUMMY_FUNCTION_PATH, n_seeds=3,
                          train_config={"eval": {"primary_metric": "val.acc", "direction": "maximize"}}),
            ],
            tasks=[TaskConfig(name="task_x"), TaskConfig(name="task_y")],
            output=OUTPUT_DIR,
            n_workers=4,
            device_ids=[0, 1],
            show_progress=False,
            auto_confirm=True,
            resume_mode="fresh",
        )
        results = BenchmarkRunner(config).run_parallel()
        assert len(results.results) == 12
