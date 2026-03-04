"""
Experiment logging system with multiple backends.

Provides unified logging interface supporting logger, wandb, and gsql
with automatic experiment tracking.
"""
import sys
import time
import uuid
import tempfile

from pathlib import Path
from dataclasses import dataclass, fields
from tqdm import tqdm
from functools import wraps
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal, TypeVar, Generic, Union

from omegaconf import DictConfig
from . import util


#>>> Logger registry <<<
C = TypeVar("C")
_logger_registry: Dict[str, 'BaseLogger'] = {}
_default_logger_name: Optional[str] = None


#>>> Logger base classes <<<
class BaseLogger(ABC, Generic[C]):
    """Base experiment logger.

    Example
    -------
    >>> @dataclass
    >>> class MyLogger(BaseLogger):
    ...     def _setup(self): pass
    ...     def log(self, metrics, step=None): pass
    """

    @dataclass
    class Config:
        verbose: bool = True  # Control console output in __call__

    def __init__(self, name: str, config: C = None):
        self.name = name
        config_cls = type(self).Config
        config = config or {}
        if isinstance(config, config_cls):
            self.config = config
        elif isinstance(config, Dict | DictConfig):
            config_fields = {f.name for f in fields(config_cls)}
            filtered_config = {k: v for k, v in config.items() if k in config_fields}
            self.config = config_cls(**filtered_config)
        else:
            raise NotImplementedError("Other type of config input is not supported")
        self.start_time = time.time()
        self._experiment_id = str(uuid.uuid4())
        self._setup()

    @abstractmethod
    def _setup(self):
        pass

    @property
    def experiment_id(self) -> str:
        """Unique experiment identifier"""
        return self._experiment_id

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameter configuration"""
        pass

    @abstractmethod
    def create_checkpoint(self, step: int, state: Dict[str, Any]):
        """Save experiment checkpoint state"""
        pass

    @abstractmethod
    def log_loss(self, loss: Union[int, float, str, dict], step: int, **kwargs):
        """Log training loss"""
        pass

    @abstractmethod
    def finish(self):
        pass

    def __call__(self, line_msg: str):
        """Print-like logging interface"""
        if self.config.verbose:
            tqdm.write(line_msg)

    def info(self, line_msg: str):
        return self(line_msg)

    def log_best_metric(self, metrics: Dict[str, Any]):
        """Log best metrics to gsql_track"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.finish()
        except Exception as e:
            print(f"Warning: Error during logger cleanup: {e}")
        return False


#>>> Logger implementations <<<
class Logger(BaseLogger['Logger.Config']):
    """Loguru enhanced logger with console output.

    Example
    -------
    >>> logger = Logger(name='exp', config=Logger.Config(log_path='./logs'))
    >>> logger.log({'loss': 0.5}, step=10)
    """
    @dataclass
    class Config:
        verbose: bool = True
        log_path: Path = None
        add_color: bool = False
        log_level: str = "INFO"
        format_string: str = "{time:MM/DD HH:mm} | {name}:{line:<4} | {level} | {message}"

    def _setup(self):
        from loguru import logger

        if self.config.log_path:
            self.log_dir = Path(self.config.log_path).parent
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "model").mkdir(exist_ok=True)
            (self.log_dir / "media").mkdir(exist_ok=True)

            # Add file sink with custom settings
            logger.add(
                sink=self.config.log_path,
                format=self.config.format_string,
                colorize=False,  # No colors in file
                level=self.config.log_level,
            )
        else:
            self.log_dir = None
            logger.remove()  # Remove default handler
            logger.add(
                sink=sys.stderr,
                format=self.config.format_string,
                colorize=self.config.add_color,
                level=self.config.log_level,
            )
        self.logger = logger
        self.best_metric = float("-inf")


    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # Track best and log improvements
        if "val_score" in metrics and metrics["val_score"] > self.best_metric:
            self.best_metric = metrics["val_score"]

    def log_loss(self, loss, step, **kwargs):
        pass

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameter configuration to file"""
        if self.log_dir:
            hp_path = self.log_dir / "hyperparameters.json"
            import json
            with open(hp_path, 'w') as f:
                json.dump(params, f, indent=2, cls=util.JSONEncoder)
        self.logger.info(f"Hyperparameters: {params}")

    def create_checkpoint(self, step: int, state: Dict[str, Any]):
        """Save experiment checkpoint state to file"""
        if self.log_dir:
            checkpoint_path = self.log_dir / "model" / f"checkpoint_step_{step}.pkl"
            import pickle
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        else:
            self.logger.info(f"Checkpoint step {step}: {state}")

    def save_figure(self, fig, step: Optional[int] = None, name: Optional[str] = None):
        """Save matplotlib figure"""
        if self.log_dir:
            filename = name or f"figure_step_{step or 0}"
            fig_path = self.log_dir / "media" / f"{filename}.png"
            fig.savefig(fig_path)
            self.logger.info(f"Figure saved: {fig_path}")

    def save_data(self, data: Any, step: Optional[int] = None, name: Optional[str] = None):
        """Save data array/tensor"""
        if self.log_dir:
            filename = name or f"data_step_{step or 0}"
            data_path = self.log_dir / "media" / f"{filename}.pkl"
            if hasattr(data, 'cpu'):  # torch tensor
                data = data.cpu().numpy()
            import pickle
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Data saved: {data_path}")

    def finish(self):
        return
        self.logger.info(
            f"Finished. Best: {self.best_metric:.4f}, Duration: {time.time() - self.start_time:.2f}s"
        )

    def __call__(self, line_msg: str):
        self.logger.info(line_msg)


class WandbLogger(BaseLogger["WandbLogger.Config"]):
    """Weights & Biases logger.

    Example
    -------
    >>> logger = WandbLogger(name='exp', config=WandbLogger.Config(project='my_project'))
    >>> logger.log({'loss': 0.5}, step=10)
    """
    @dataclass
    class Config:
        verbose: bool = True
        project: Optional[str] = None
        tags: Optional[list] = None
        notes: Optional[str] = None
        mode: Literal['online', 'offline', 'disabled'] = "online"
        save_code: bool = False
        resume: Literal['allow', 'never', 'must', 'auto'] = None
        id: Optional[str] = None
        group: Optional[str] = None
        job_type: Optional[str] = None

    def _setup(self):
        import wandb
        wandb_config = {
            "project": self.config.project or self.name,
            "tags": self.config.tags,
            "notes": self.config.notes,
            "mode": self.config.mode,
            "save_code": self.config.save_code,
            "resume": self.config.resume,
            "id": self.config.id,
            "group": self.config.group,
            "job_type": self.config.job_type,
            "dir": None,
            "reinit": True,
            'settings': wandb.Settings(
                x_disable_stats=True,
                x_disable_meta=False,
                console="auto",
                disable_git=True,
                disable_code=True,
            )
        }
        self.run = wandb.init(**wandb_config)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self.run.log(metrics, step=step)

    def log_loss(self, loss, step, **kwargs):
        self.log({'loss': loss}, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameter configuration to wandb"""
        self.run.config.update(params)

    def create_checkpoint(self, step: int, state: Dict[str, Any]):
        """Save experiment checkpoint state to wandb artifacts"""
        import wandb
        import pickle
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(state, f)
            temp_path = f.name
        try:
            artifact = wandb.Artifact(f"checkpoint-{step}", type="model")
            artifact.add_file(temp_path, f"checkpoint_step_{step}.pkl")
            self.run.log_artifact(artifact)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def log_image(self, image, step: Optional[int] = None, name: Optional[str] = None):
        """Log image to wandb"""
        import wandb
        key = name or f"image_step_{step or 0}"
        self.run.log({key: wandb.Image(image)}, step=step)

    def log_plot(self, plot, step: Optional[int] = None, name: Optional[str] = None):
        """Log plot to wandb"""
        key = name or f"plot_step_{step or 0}"
        self.run.log({key: plot}, step=step)

    def log_table(self, data, step: Optional[int] = None, name: Optional[str] = None):
        """Log table to wandb"""
        import wandb
        key = name or f"table_step_{step or 0}"
        table = wandb.Table(data=data) if isinstance(data, list) else data
        self.run.log({key: table}, step=step)

    def finish(self):
        self.run.finish()


class GsqlLogger(BaseLogger['GsqlLogger.Config']):
    """gsql_track logger — persists metrics to ~/.gsql/track.db.

    Example
    -------
    >>> logger = GsqlLogger(name='exp', config=GsqlLogger.Config(experiment='my_exp'))
    >>> logger.log({'loss': 0.5}, step=10)
    """
    @dataclass
    class Config:
        verbose: bool = True
        experiment: str = "default"
        run_name: str = ""
        db_path: str = None

    def _setup(self):
        from .gsql_track import GsqlTrack
        self._track = GsqlTrack(self.config.experiment, db_path=self.config.db_path)
        self._run = self._track.start_run(self.config.run_name or self.name)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if step is not None:
            numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            # Remove 'step' from numeric if present to avoid conflict with step parameter
            numeric.pop('step', None)
            if numeric:
                self._run.log(step=step, **numeric)

    def log_loss(self, loss, step, **kwargs):
        self._run.log(step=step, loss=float(loss))

    def log_hyperparameters(self, params: Dict[str, Any]):
        # Only log scalar params — skip nested dicts/lists to avoid noise
        scalar = {k: v for k, v in params.items()
                  if isinstance(v, (str, int, float, bool)) or v is None}
        if scalar:
            self._run.log_params(scalar)

    def create_checkpoint(self, step: int, state: Dict[str, Any]):
        pass  # gsql doesn't store checkpoints

    def finish(self):
        self._run.finish()
        self._track.close()


#>>> Logger factory functions <<<
def get_logger(
    mode: Literal['wandb', 'logger', 'gsql'] = None,
    name: Optional[str] = None,
    config=None,
    default_logger: bool = False
) -> BaseLogger:
    """Get or create a logger instance with registry-based singleton behavior.

    Example
    -------
    >>> logger = get_logger(name='my_experiment', default_logger=True)
    >>> logger = get_logger()  # Returns the same instance
    """
    global _default_logger_name

    if mode is None and name is None:
        if _default_logger_name is None:
            raise ValueError("No default logger set. Use set_default_logger() or provide logger_type/name.")
        if _default_logger_name not in _logger_registry:
            raise ValueError(f"Default logger '{_default_logger_name}' not found in registry.")
        return _logger_registry[_default_logger_name]

    is_default = default_logger

    if (not is_default) and (mode is None) and (name is not None):
        if name in _logger_registry:
            default_logger and _set_default_logger(name)
            return _logger_registry[name]
        raise ValueError(f"Logger '{name}' not found in registry. Provide logger_type to create it.")

    registry_key = name or mode
    default_logger and _set_default_logger(registry_key)

    if (not is_default) and (registry_key in _logger_registry):
        return _logger_registry[registry_key]

    logger_map = {
        'logger': Logger,
        'wandb': WandbLogger,
        'gsql': GsqlLogger,
    }
    logger_cls = logger_map.get(mode.lower())
    if not logger_cls:
        available_types = list(logger_map.keys())
        default_logger and _set_default_logger(None)
        raise ValueError(
            f"Unknown logger type: {mode}. Available types: {available_types}"
        )
    logger = logger_cls(name=registry_key, config=config)
    _logger_registry[registry_key] = logger
    return logger


def _set_default_logger(name: str):
    """Set the default logger by name."""
    global _default_logger_name
    _default_logger_name = name


def clear_logger_registry():
    """Clear all loggers from the registry and reset default.

    Example
    -------
    >>> clear_logger_registry()
    >>> logger = get_logger('wandb', name='my_experiment')
    """
    global _logger_registry, _default_logger_name
    _logger_registry.clear()
    _default_logger_name = None


def log_experiment(logger_cls: BaseLogger, **logger_kwargs):
    """Decorator for automatic logger lifecycle management in experiment functions.

    Example
    -------
    >>> @log_experiment(Logger, add_color=True, log_level="DEBUG")
    >>> def train_model(config: Dict, logger: BaseLogger):
    ...     logger.log({"loss": loss}, step=epoch)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(config: Dict[str, Any], *args, **kwargs):
            logger_config = logger_cls.Config(config=logger_kwargs)
            with logger_cls(func.__name__, logger_config) as logger:
                return func(config, logger, *args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    'BaseLogger',
    'Logger',
    'WandbLogger',
    'GsqlLogger',
    'get_logger',
    'clear_logger_registry',
    'log_experiment',
]
