"""
gsql_track — lightweight experiment tracking and ML infrastructure.

Public API:
    Core Tracking:
    - GsqlTrack: Main tracking class
    - tracked: Wrapper for existing Tracker instances

    Modules:
    - util: Generic utilities (Timer, EarlyStopping, fmt_duration, etc.)
    - log: Logger abstraction (Logger, WandbLogger, GsqlLogger)
    - config: Config parsing and management
    - db: SQLite job queue and state persistence
    - dispatch: Parallel worker dispatch and GPU isolation
    - enums: Framework enums (Direction, TrackingMode, LogMode, etc.)
"""

"""
gsql_track — lightweight experiment tracking and ML infrastructure.

Public API:
    Core Tracking:
    - GsqlTrack: Main tracking class
    - tracked: Wrapper for existing Tracker instances

    Modules:
    - util: Generic utilities (Timer, EarlyStopping, fmt_duration, etc.)
    - log: Logger abstraction (Logger, WandbLogger, GsqlLogger)
    - config: Config parsing and management
    - db: SQLite job queue and state persistence
    - dispatch: Parallel worker dispatch and GPU isolation
    - enums: Framework enums (Direction, TrackingMode, LogMode, etc.)
    - tracker: Training loop tracker (Tracker, ModelSerializer)
    - bench: Benchmark runner (BenchmarkRunner, requires pandas/numpy)
    - tune: Hyperparameter tuning (Tuner, requires optuna/pandas/numpy)
"""

from .gsql_track import GsqlTrack, tracked
from . import util
from . import log
from . import config
from . import db
from . import dispatch
from . import enums
from . import tracker

# bench and tune have heavy deps (pandas, numpy, optuna) — import on demand
def __getattr__(name):
    if name == "bench":
        from . import bench
        return bench
    if name == "tune":
        from . import tune
        return tune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core tracking
    "GsqlTrack",
    "tracked",
    # Modules
    "util",
    "log",
    "config",
    "db",
    "dispatch",
    "enums",
    "tracker",
    "bench",
    "tune",
]

__version__ = "0.2.0"
