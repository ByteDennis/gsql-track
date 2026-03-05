"""Shared fixtures for integration tests — dummy train functions and config helpers."""
import random
import time

from omegaconf import OmegaConf
from gsql_track.config import (
    TrainConfig, ModelConfig, DataConfig, OutputConfig, EvalConfig, PbarConfig, LogConfig,
)
from gsql_track.enums import TrackingMode


DUMMY_FUNCTION_PATH = "tests.fixtures.dummy_train_fn"
DUMMY_FUNCTION_WITH_TRACKER_PATH = "tests.fixtures.dummy_train_fn_with_tracker"


def dummy_train_fn(config_dict: dict) -> dict:
    """Simulate training: return fake metrics based on seed."""
    seed = config_dict.get("seed", 42)
    rng = random.Random(seed)
    time.sleep(0.01)
    val_acc = round(rng.uniform(0.6, 0.95), 4)
    test_acc = round(rng.uniform(0.55, 0.90), 4)
    val_loss = round(rng.uniform(0.1, 1.0), 4)
    return {"val.acc": val_acc, "test.acc": test_acc, "val.loss": val_loss}


def dummy_train_fn_with_tracker(config_dict: dict) -> dict:
    """Full Tracker lifecycle: create, log epochs, finalize."""
    from gsql_track.tracker import Tracker
    from gsql_track.config import parse_config, create_config

    cfg = create_config(config_dict)
    tracker = Tracker(config=cfg)

    seed = config_dict.get("seed", 42)
    rng = random.Random(seed)
    n_epochs = config_dict.get("pbar", {}).get("total", 5)

    for epoch in range(n_epochs):
        loss = round(1.0 - (epoch / n_epochs) + rng.uniform(-0.05, 0.05), 4)
        acc = round(epoch / n_epochs + rng.uniform(-0.05, 0.05), 4)
        tracker.log_epoch({"val.loss": loss, "val.acc": acc, "test.acc": acc * 0.95}, epoch=epoch)
        if tracker.should_stop:
            break

    tracker.finalize()
    return tracker.best_metrics


def make_train_config(tmp_path, tracking_mode="epochs", n_total=10, primary_metric="val.acc", direction="maximize"):
    """Build a TrainConfig with all required fields pointing to tmp dirs."""
    output_dir = str(tmp_path / "output")
    cfg_dict = {
        "model": {"name": "dummy", "metric": "acc"},
        "data": {"init_args": {}, "process_args": {}},
        "output": {
            "folder": output_dir,
            "save_model": False, "save_model_only": False,
            "save_pred": False, "save_final": False, "save_best": False,
            "delete_best": True, "delete_last": True,
            "save_config": False, "save_analysis": False,
            "save_paths": {},
        },
        "eval": {
            "primary_metric": primary_metric,
            "patience": 5,
            "direction": direction,
        },
        "pbar": {"enabled": False, "total": n_total, "persist": False, "epoch_metrics": []},
        "log": {"mode": "logger"},
        "tracking_mode": tracking_mode,
        "seed": 42,
        "resume_from": "best_model",
        "exp_name": "E01",
    }
    return OmegaConf.create(cfg_dict)
