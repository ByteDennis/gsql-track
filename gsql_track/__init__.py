# ruff: noqa: F403 F405
"""
gsql_track — lightweight experiment tracking and ML infrastructure.

Public API re-exports all submodule contents for convenient access:
    import gsql_track as lib
    lib.Tracker(...)
    lib.METRIC_REGISTRY[...]
"""

from .gsql_track import GsqlTrack, tracked

# Re-export all submodule contents for `import gsql_track as lib` usage
from .config import *
from .db import *
from .dispatch import *
from .log import *
from .util import *
from .plan import *
from .enums import *
from .types import *
from .tracker import *
from .metric import *
from .prompt import *
from .data import *
from .testing import *

# bench and tune have heavy deps (pandas, numpy, optuna) — lazy import submodules
# but eagerly re-export their contents for lib.X access
from .bench import *
from .tune import *

__all__ = ["GsqlTrack", "tracked"]

__version__ = "0.3.0"
