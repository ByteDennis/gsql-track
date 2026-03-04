"""
gsql.track — lightweight experiment tracking and ML infrastructure.

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

from .gsql_track import GsqlTrack, tracked
from . import util
from . import log
from . import config
from . import db
from . import dispatch
from . import enums

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
]

__version__ = "0.1.0"
