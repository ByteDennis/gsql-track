"""
Generic utilities for ML experiments.

Provides timer, early stopping, progress tracking, data structures,
reproducibility, and I/O utilities.
"""
import os
import re
import sys
import json
import time
import random
import importlib
import threading
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime, date
from unittest.mock import patch
from contextlib import contextmanager, ExitStack
from typing import Any, Dict, Optional, List, Callable
from omegaconf import OmegaConf

try:
    import torch
except ImportError:
    torch = None

# Constants
ANSI_ESCAPE = re.compile(r'\x1B\[@-_][0-?]*[ -/]*[@-~]')


#>>> JSON Encoder <<<
class JSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types."""
    def default(self, obj):
        from collections.abc import Mapping, Sequence
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if isinstance(obj, Mapping) and not isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, list, tuple)):
            return list(obj)
        return super().default(obj)


#>>> Reproducibility <<<
def set_seed(seed: int = 42, benchmark=True, deterministic=True) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


#>>> Dictionary Utilities <<<
def unnest_dict(dct, parent_key="", sep="."):
    """Convert nested dict to flat dict with dot notation keys."""
    items = []
    for k, v in dct.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(unnest_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nest_dict(dct, sep="."):
    """Convert flat dict with dot notation keys to nested dict."""
    result = {}
    for key, value in dct.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def flat_dict(dict_data):
    """Convert dot-notation flat dict to nested dict using OmegaConf."""
    dotlist = [f"{k}={v}" for k, v in dict_data.items()]
    cfg = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)


def merge_dict(dct, override):
    """Merge override dict into dct (recursive for nested dicts)."""
    for k, v in override.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
            merge_dict(dct[k], v)
        else:
            dct[k] = v
    return dct


#>>> Import Utilities <<<
def import_function(function_path: str) -> Callable:
    """Import a function from a dot-separated string path."""
    try:
        module_path, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        main_func = getattr(module, function_name)
        return getattr(main_func, "__wrapped__", main_func)
    except Exception as e:
        raise ImportError(f"Failed to import {function_path}: {e}")


def pre_import_modules(modules: List[str], silent: bool = False):
    """Pre-import modules to avoid import overhead during parallel execution."""
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            if not silent:
                print(f"✓ Pre-imported: {module_name}")
        except ImportError as e:
            if not silent:
                print(f"⚠️  Failed to pre-import {module_name}: {e}")


#>>> Resource Cleanup <<<
def cleanup_resources():
    """Clean up GPU memory and run garbage collection."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


#>>> Formatting Utilities <<<
def fmt_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def fmt_float(value: float, decimals: int = 4) -> str:
    """Format float with fixed decimals."""
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else str(value)


def print_title(txt, width=60, logger=None):
    """Print a title with borders."""
    log = logger if logger is not None else print
    log(f"\n{'='*width}")
    log(txt)
    log(f"{'='*width}")


#>>> Timer Class <<<
class Timer:
    """Simple timer for tracking elapsed time."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def total(self):
        """Return total elapsed time (updates if still running)."""
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0

    @staticmethod
    def format_time(timestamp: float) -> str:
        """Format Unix timestamp to readable string."""
        if timestamp is None:
            return "-"
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


#>>> Early Stopping <<<
class EarlyStopping:
    """Early stopping utility with patience and best metric tracking."""

    def __init__(
        self,
        patience: int = 10,
        direction: str = "maximize",
        primary_metric: str = "val.acc",
        patience_type: str = "epochs",
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        buffer: int = 0,
        min_delta: float = 0.0,
        **kwargs,
    ):
        self.patience = patience
        self.direction = direction
        self.primary_metric = primary_metric
        self.patience_type = patience_type
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.steps_per_epoch = steps_per_epoch
        self.buffer = buffer
        self.min_delta = min_delta

        self._enable = patience > 0
        self.counter = 0
        self.update_count = 0
        self.best_value = float('-inf') if direction == "maximize" else float('inf')
        self._is_better = False

    def update(self, value: float):
        """Update with new metric value and check if improved."""
        self.update_count += 1

        if self.direction == "maximize":
            is_better = value > self.best_value + self.min_delta
        else:
            is_better = value < self.best_value - self.min_delta

        if is_better:
            self.best_value = value
            self.counter = 0
            self._is_better = True
        elif self.update_count > self.buffer:
            self.counter += 1
            self._is_better = False
        else:
            self._is_better = False

    def should_stop(self) -> bool:
        """Check if early stopping criteria met."""
        return self._enable and self.counter >= self.patience

    def should_end(self, step: int, epoch: int) -> bool:
        """Check if training should end based on max epochs/steps."""
        if self.n_epochs is not None and epoch >= self.n_epochs:
            return True
        if self.n_steps is not None and step >= self.n_steps:
            return True
        return False


#>>> Progress Tracking <<<
class ProgressWriter:
    """Thread-safe writer for progress updates."""

    def __init__(self, silent: bool = False):
        self.silent = silent
        self._lock = threading.Lock()

    def write_progress(self, msg: str, overwrite: bool = False):
        """Write progress message (optionally overwriting)."""
        if self.silent:
            return
        with self._lock:
            if overwrite:
                sys.stdout.write(f"\r{msg}")
            else:
                sys.stdout.write(f"{msg}\n")
            sys.stdout.flush()

    def write_event(self, msg: str):
        """Write event message (always on new line)."""
        if self.silent:
            return
        with self._lock:
            sys.stdout.write(f"{msg}\n")
            sys.stdout.flush()


class BaseProgressTracker:
    """Base class for progress trackers with ETA calculation."""

    def __init__(self, total: int, writer: ProgressWriter):
        self.total = total
        self.writer = writer
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_static_part = ""
        self._lock = threading.Lock()

    def calculate_eta(self, elapsed: float) -> str:
        """Calculate estimated time remaining."""
        if self.completed == 0:
            return ""
        rate = elapsed / self.completed
        remaining = (self.total - self.completed) * rate
        return f" | ETA: {self.fmt_time(remaining)}"

    @staticmethod
    def fmt_time(seconds: float) -> str:
        """Format time duration."""
        return fmt_duration(seconds)


#>>> Tqdm Utilities <<<
@contextmanager
def no_tqdm_pbar(disable: bool = True):
    """Context manager to optionally disable tqdm progress bars."""
    if not disable:
        yield
        return

    try:
        import tqdm.std

        class DisabledTqdm(tqdm.std.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **{**kwargs, "disable": True})

        module_items = list(sys.modules.items())
        paths = ["tqdm.tqdm", "tqdm.std.tqdm", "tqdm.auto.tqdm"] + [
            f"{name}.tqdm" for name, mod in module_items if mod and hasattr(mod, "tqdm")
        ]
        with ExitStack() as stack:
            for path in paths:
                try:
                    stack.enter_context(patch(path, DisabledTqdm))
                except (ImportError, AttributeError):
                    pass
            yield
    except ImportError:
        yield


#>>> Output Redirection <<<
class Tee:
    """Redirect output to both console and file, stripping ANSI codes from file."""

    def __init__(self, *files, strip_ansi=True):
        self.files = files
        self.strip_ansi = strip_ansi

    def write(self, data):
        for f in self.files:
            if self.strip_ansi and f not in (sys.__stdout__, sys.__stderr__):
                clean_data = ANSI_ESCAPE.sub('', data)
                if clean_data.startswith('\r'):
                    continue
                f.write(clean_data)
            else:
                f.write(data)

    def flush(self):
        [f.flush() for f in self.files]

    def isatty(self):
        return any(hasattr(f, 'isatty') and f.isatty() for f in self.files)

    def fileno(self):
        for f in self.files:
            if hasattr(f, 'fileno'):
                try:
                    return f.fileno()
                except Exception:
                    continue
        raise AttributeError("No underlying file has a fileno()")


def redirect_output_to_file(log_file_path: str | Path) -> Tee:
    """Redirect stdout and stderr to both console and a log file."""
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_path, "w", buffering=1)
    tee = Tee(sys.__stdout__, log_file, strip_ansi=True)
    sys.stdout = sys.stderr = tee
    return tee


__all__ = [
    "JSONEncoder",
    "set_seed",
    "unnest_dict",
    "nest_dict",
    "flat_dict",
    "merge_dict",
    "import_function",
    "pre_import_modules",
    "cleanup_resources",
    "fmt_duration",
    "fmt_float",
    "print_title",
    "Timer",
    "EarlyStopping",
    "ProgressWriter",
    "BaseProgressTracker",
    "no_tqdm_pbar",
    "Tee",
    "redirect_output_to_file",
]
