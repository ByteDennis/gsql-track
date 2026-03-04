"""
Essential type definitions and enums for ML experiment framework.

Provides enums for training status, logging modes, optimization direction,
and other framework-level types.
"""
import enum


class Enum(enum.Enum):
    """Base enum with string representation."""
    def __repr__(self) -> str:
        return self.value

    @classmethod
    def get(cls, value) -> str:
        return cls(value).value


class Direction(str, Enum):
    """Optimization direction."""
    minimize = "minimize"
    maximize = "maximize"


class TrackingMode(str, Enum):
    """Tracking granularity for training progress."""
    steps = "steps"
    epochs = "epochs"


class LogMode(str, Enum):
    """Logging backend options."""
    logger = "logger"
    wandb = "wandb"
    gsql = "gsql"


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    train = "train"
    test = "test"
    terminate = "terminate"


class SampleMode(str, Enum):
    """Hyperparameter sampling strategy (for Optuna)."""
    tpe = "tpe"
    random = "random"
    grid = "grid"
    cmaes = "cmaes"


class PruneMode(str, Enum):
    """Trial pruning strategy (for Optuna)."""
    nop = "nop"
    median = "median"
    hyperband = "hyperband"


class ResumeMode(str, Enum):
    """Checkpoint resume source."""
    best_model = "best_model"
    last_model = "last_model"
    best_pred = "best_pred"
    last_pred = "last_pred"


# Backwards compatibility
Patience = TrackingMode


__all__ = [
    "Direction",
    "TrackingMode",
    "LogMode",
    "TrainingStatus",
    "SampleMode",
    "PruneMode",
    "ResumeMode",
    "Patience",
]
