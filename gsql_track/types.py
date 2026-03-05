"""
Type definitions and state models for ML experiment framework.

Provides Pydantic state models for training, tuning, and experiment tracking,
plus abstract base classes for preprocessing pipelines.
"""
import os
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, model_validator
from typing import Dict, Optional, List, Any, Type, Callable, Generic, TypeVar

from .enums import TrainingStatus


#>>> Pydantic models for experiment tracking <<<
class ModelInfo(BaseModel):
    """Model information."""
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    model_class: str


class ExecuteInfo(BaseModel):
    """Execution statistics."""
    time_per_epoch: float
    infer_per_epoch: float
    sample_size: int
    batch_size: int
    total_train_time: float
    total_infer_time: float


class ResultInfo(BaseModel):
    """Training results."""
    last_epoch: int
    last_metrics: Dict[str, float]
    best_epoch: int
    best_metrics: Dict[str, float]


class TrainingState(BaseModel):
    """Internal training state for the tracker.

    Example
    -------
    >>> state = TrainingState()
    >>> state.current_epoch = 5
    >>> state.best_metrics = {'acc': 0.85}
    """
    status: TrainingStatus = TrainingStatus.train
    start_epoch: int = 0
    start_step: int = 0
    current_epoch: int = 0
    current_step: int = 0
    best_epoch: int = 0
    best_step: int = 0
    best_metric_value: Optional[float] = None
    latest_metrics: Dict[str, Any] = {}
    best_metrics: Dict[str, Any] = {}

    @model_validator(mode="before")
    def init_current_from_start(cls, values: dict):
        values.setdefault("current_epoch", values.get('start_epoch', 0))
        values.setdefault("current_step", values.get('start_step', 0))
        return values

    def __repr__(self) -> str:
        def format_metrics(metrics: Dict[str, Any]) -> str:
            formatted = {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in metrics.items()
            }
            return str(formatted)

        return (
            f"TrainingState("
            f"status={self.status.value}, "
            f"epoch={self.start_epoch}=>{self.current_epoch}, "
            f"step={self.start_step}=>{self.current_step}, "
            f"best_epoch={self.best_epoch}, "
            f"best_step={self.best_step}, "
            f"best_metric={self.best_metric_value}, "
            f"latest_metrics={format_metrics(self.latest_metrics)}, "
            f"best_metrics={format_metrics(self.best_metrics)})"
        )


class TuningState(BaseModel):
    """State tracking for hyperparameter tuning.

    Example
    -------
    >>> state = TuningState(n_trials=100, metric='accuracy')
    >>> state.best_value = 0.85
    """
    direction: str = "maximize"
    n_trials: Optional[int] = None
    completed_trials: int = 0
    timeout: int = 0
    elapsed_time: float = 0
    opt_variables: List[str] = []
    metric: str = "accuracy"
    status: str = "running"
    best_value: float = 0
    best_trial: Optional[int] = None

    def __repr__(self) -> str:
        progress = (
            f"{self.completed_trials}/{self.n_trials}"
            if self.n_trials
            else f"{self.completed_trials}"
        )
        elapsed = f"{self.elapsed_time:.1f}s"
        time_info = f"{elapsed}/{self.timeout}s" if self.timeout else elapsed

        best_val = (
            f"{self.best_value:.4f}"
            if isinstance(self.best_value, float)
            else str(self.best_value)
        )
        vars_display = ", ".join(self.opt_variables[:3])
        if len(self.opt_variables) > 3:
            vars_display += f"... (+{len(self.opt_variables) - 3} more)"
        return (
            f"TuningState("
            f"trials={progress}, "
            f"time={time_info}, "
            f"status={self.status}, "
            f"{self.direction}({self.metric})={best_val}, "
            f"vars=[{vars_display}])"
        )


class SummaryReport(BaseModel):
    """Complete training summary report."""
    eval_config: Dict[str, Any]
    output_config: Dict[str, Any]
    model_info: Dict[str, Any]
    execute_info: Dict[str, Any]
    result_info: Dict[str, Any]


#>>> Base class for preprocessing steps (fit/transform pattern) <<<
@dataclass
class StepConfig:
    """Configuration for cache key generation.

    Example
    -------
    >>> cfg = StepConfig(type='feature', params={'extract_fn': 'bert'})
    >>> cfg.config_hash()
    'a1b2c3d4e5f6'
    """
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def config_hash(self) -> str:
        normalized = json.dumps(self.params, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]


class PreprocessStep(ABC):
    """Abstract base for preprocessing steps.

    Example
    -------
    >>> class CustomStep(PreprocessStep):
    ...     @property
    ...     def step_type(self):
    ...         return 'custom'
    ...     def fit(self, train_data):
    ...         return lambda x: x * 2
    ...     def transform(self, data, fitted_state):
    ...         return fitted_state(data)
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self._fitted_state = None

    @property
    @abstractmethod
    def step_type(self) -> str:
        pass

    @property
    def cache_key(self) -> str:
        config = StepConfig(type=self.step_type, params=self.params)
        return f"{self.step_type}_{config.config_hash()}"

    @abstractmethod
    def fit(self, train_data: Any) -> Any:
        pass

    @abstractmethod
    def transform(self, data: Any, fitted_state: Any) -> Any:
        pass

    def fit_transform(self, train_data: Any) -> Any:
        self._fitted_state = self.fit(train_data)
        return self.transform(train_data, self._fitted_state)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


#>>> Data registry infrastructure <<<
DatasetT = TypeVar("DatasetT")


class BaseData(ABC, Generic[DatasetT]):
    """Base class for data processors with registry support.

    Example
    -------
    >>> @register_data("wrench_data")
    >>> class WrenchData(BaseData):
    ...     def process(self, **kwargs):
    ...         return train, valid, test
    >>> processor = WrenchData(name='yelp', data_path='./datasets/yelp')
    >>> train, valid, test = processor.process()
    """

    meta_info = property(lambda self: self._meta)

    def __init__(self, name: str, data_path: str = "", **kwargs):
        self.name = name
        self.data_path = os.path.expandvars(data_path) if data_path else ""
        self.init_args = kwargs
        self._meta = {'data_path': self.data_path, **kwargs}

    @abstractmethod
    def process(self, **kwargs):
        """Process and return the dataset."""
        pass

    @classmethod
    def process_data(cls, init_args, process_args, return_meta=False, **kwargs):
        """Process data from init and process arguments."""
        obj = cls(**init_args)
        if return_meta:
            return obj.process(**process_args), obj.meta_info
        return obj.process(**process_args)


class DataRegistry(dict):
    """Registry for data processor classes.

    Example
    -------
    >>> registry = DataRegistry()
    >>> registry['wrench_data']  # Access registered data class
    """

    def __call__(
        self,
        data_class: Type["BaseData"],
        data_name: str = None,
        init_args: dict = None,
        process_args: dict = None,
    ):
        process_args = process_args or {}
        data_obj = self[data_class].process_data(
            init_args=init_args or {},
            process_args=process_args or {}
        )
        data_obj.name = data_name or data_obj.__class__.__name__
        return data_obj


DATA_REGISTRY: DataRegistry = DataRegistry()


def register_data(name: str):
    """Decorator to register a data processor class.

    Example
    -------
    >>> @register_data("wrench_data")
    >>> class WrenchData(BaseData):
    ...     ...
    """
    def decorator(cls: Type["BaseData"]):
        DATA_REGISTRY[name] = cls
        return cls
    return decorator


class DataConfigurationNotDefined(Exception):
    """Exception raised when data configuration is not defined."""
    pass


#>>> Preprocess step registry <<<
PREPROCESS_STEP_REGISTRY: Dict[str, type] = {}


def register_pp(name: str):
    """Decorator to register a preprocessing step class.

    Example
    -------
    >>> @register_pp("custom")
    >>> class CustomStep(PreprocessStep):
    ...     ...
    """
    def decorator(cls):
        PREPROCESS_STEP_REGISTRY[name] = cls
        return cls
    return decorator


__all__ = [
    # State models
    "ModelInfo",
    "ExecuteInfo",
    "ResultInfo",
    "TrainingState",
    "TuningState",
    "SummaryReport",
    # Preprocessing
    "PreprocessStep",
    "StepConfig",
    # Data registry
    "DatasetT",
    "BaseData",
    "DataRegistry",
    "DATA_REGISTRY",
    "register_data",
    "DataConfigurationNotDefined",
    # Preprocess registry
    "PREPROCESS_STEP_REGISTRY",
    "register_pp",
]
