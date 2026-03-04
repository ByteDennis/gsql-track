"""
Experiment tracking for ML training workflows.

Tracker class handles state management, metrics logging, early stopping,
checkpointing, and summary generation with <10 lines integration.
"""
import re
import json
import pickle
import inspect

from pathlib import Path
from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator, validate_call
from typing import (
    Dict, Any, Callable, Union, Optional, TypedDict, Unpack, Tuple, Literal, List
)

from . import util as U
from . import config as C
from .enums import TrainingStatus
from .log import get_logger

try:
    import torch
except ImportError:
    torch = None

try:
    import joblib
except ImportError:
    joblib = None

logger = None


# ─── Pydantic State Models ──────────────────────────────────────────────────

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


# ─── Model Serialization ────────────────────────────────────────────────────

class Checkpoints(TypedDict, total=False):
    model: Any
    optimizer: Any
    scheduler: Any
    ema: Any


class ModelSerializer:
    """Serialize/deserialize any model type via duck typing.

    Example
    -------
    >>> ModelSerializer.save(model, Path("checkpoint"))
    >>> ModelSerializer.load(Path("checkpoint.pt"), obj=model)
    """

    @staticmethod
    def save(obj: Any, path: Path, ext: str = None) -> None:
        path = Path(path)
        if ext:
            ext = ext.lstrip('.')
            getattr(U, f"save_{ext}")(obj, path.with_suffix(f".{ext}"))
            return
        if isinstance(obj, tuple) and len(obj) == 2:
            ModelSerializer.save(obj[0], path, obj[1])
        elif isinstance(obj, dict):
            U.save_json(obj, path.with_suffix(".json"))
        elif hasattr(obj, "state_dict"):
            if torch is None:
                raise ImportError("torch required to save state_dict objects")
            torch.save(obj.state_dict(), path.with_suffix(".pt"))
        elif hasattr(obj, "save_pretrained"):
            obj.save_pretrained(path.parent)
        elif hasattr(obj, "get_params"):
            if joblib is None:
                raise ImportError("joblib required to save sklearn-like objects")
            joblib.dump(obj, path.with_suffix(".joblib"))
        else:
            U.save_pkl(obj, path.with_suffix(".pkl"))

    @staticmethod
    def save_json(obj, path):
        json.dump(obj, open(path, "w"), indent=2)

    @staticmethod
    def save_pt(obj, path):
        if torch is None:
            raise ImportError("torch required for save_pt")
        torch.save(obj, path)

    @staticmethod
    def save_joblib(obj, path):
        if joblib is None:
            raise ImportError("joblib required for save_joblib")
        joblib.dump(obj, path)

    @staticmethod
    def save_pkl(obj, path):
        pickle.dump(obj, open(path, "wb"))

    @staticmethod
    def load(path: Path, obj: Any = None) -> Any:
        """Load object using appropriate method."""
        path = Path(path)
        if not path.suffix:
            for ext in [".json", ".pt", ".joblib", ".pkl"]:
                if (candidate := path.with_suffix(ext)).exists():
                    path = candidate
                    break
        try:
            if path.suffix == ".json":
                return U.load_json(path)

            if path.suffix == ".pt":
                if torch is None:
                    raise ImportError("torch required to load .pt files")
                state_dict = torch.load(path, map_location="cpu", weights_only=False)
                if obj and hasattr(obj, "load_state_dict"):
                    obj.load_state_dict(state_dict)
                    return obj
                return state_dict

            if path.suffix == ".joblib":
                if joblib is None:
                    raise ImportError("joblib required to load .joblib files")
                return joblib.load(path)

            if path.suffix == ".pkl":
                return U.load_pickle(path)

            return U.load_pickle(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture all RNG states for reproducibility."""
        import random
        import numpy as np

        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        if torch is not None:
            rng_state['torch'] = torch.get_rng_state()
            if torch.cuda.is_available():
                rng_state['torch_cuda'] = torch.cuda.get_rng_state_all()
        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state: dict) -> None:
        """Restore all RNG states."""
        import random
        import numpy as np

        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        if torch is not None and 'torch' in rng_state:
            torch.set_rng_state(rng_state['torch'])
            if 'torch_cuda' in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['torch_cuda'])


# ─── Progress Bar Stub ──────────────────────────────────────────────────────
# Tracker uses U.ProgressBar and U.EarlyStopping from gsql_track.util.
# If ProgressBar doesn't exist in util, we provide a minimal stub.

if not hasattr(U, 'ProgressBar'):
    class _ProgressBarStub:
        def __init__(self, **kwargs): pass
        def start_epoch(self, epoch, step): pass
        def update(self, **kwargs): pass
        def update_epoch_metrics(self, metrics): pass
        def update_best(self): pass
        def set_description(self, desc): pass
        def close(self): pass
        def __getstate__(self): return {}
        def __setstate__(self, state): pass
        def __repr__(self): return "ProgressBar(stub)"
    U.ProgressBar = _ProgressBarStub

if not hasattr(U, 'parse_epoch_step_string'):
    def _parse_epoch_step_string(value, default_type='epochs'):
        if isinstance(value, int):
            return default_type, value
        if isinstance(value, str):
            import re
            m = re.match(r'^(\d+)\s*(e(?:pochs?)?|s(?:teps?)?)$', value.strip(), re.IGNORECASE)
            if m:
                n = int(m.group(1))
                unit = m.group(2).lower()
                return ('steps' if unit.startswith('s') else 'epochs'), n
            if value.strip().isdigit():
                return default_type, int(value.strip())
        raise ValueError(f"Cannot parse '{value}' as epoch/step string")
    U.parse_epoch_step_string = _parse_epoch_step_string

if not hasattr(U, 'save_json'):
    def _save_json(obj, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, cls=U.JSONEncoder)
    U.save_json = _save_json

if not hasattr(U, 'load_json'):
    def _load_json(path):
        with open(path) as f:
            return json.load(f)
    U.load_json = _load_json

if not hasattr(U, 'save_pkl'):
    def _save_pkl(obj, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    U.save_pkl = _save_pkl

if not hasattr(U, 'load_pickle'):
    def _load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    U.load_pickle = _load_pickle

if not hasattr(U, 'stop_when_error'):
    U.stop_when_error = lambda: None

if not hasattr(U, 'ensure_dir'):
    def _ensure_dir(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    U.ensure_dir = _ensure_dir


# ─── Main Tracker ────────────────────────────────────────────────────────────

class Tracker:
    """Comprehensive experiment tracker with <10 lines training loop integration.

    Example
    -------
    >>> tracker = Tracker(config)
    >>> for epoch in range(10):
    ...     tracker >> 1
    ...     val_metrics = validate(model)
    ...     tracker.log_epoch(val_metrics)
    ...     if tracker.is_better:
    ...         tracker.save_checkpoint(model=model)
    ...     if tracker.should_stop:
    ...         break
    >>> tracker.finalize()
    """
    state  = property(lambda self: self._state)
    status = property(lambda self: self._state.status)
    epoch  = property(lambda self: self._state.current_epoch)
    step   = property(lambda self: self._state.current_step)
    best_metrics = property(lambda self: self._state.best_metrics or self._state.latest_metrics)
    best_epoch   = property(lambda self: self._state.best_epoch)
    best_step    = property(lambda self: self._state.best_step)
    best_metric_value = property(lambda self: self._state.best_metric_value)
    latest_metrics = property(lambda self: self._state.latest_metrics)

    def __init__(self, config: C.TrainConfig = None, overrides: dict = None):
        if overrides:
            config = C.parse_config(config, overrides=overrides)
        self.config = config

        global logger
        logger = get_logger(**OmegaConf.structured(config.log))

        self.eval_config = config.eval
        self.output_config = config.output
        self.model_config = config.model
        self.primary_metric = self.eval_config.primary_metric

        self._state = TrainingState()
        self.timer = U.Timer()
        self.inference_timer = U.Timer()
        self.tracking_mode = config.tracking_mode

        # Configure early stopping with tracking mode
        eval_dict = OmegaConf.to_container(config.eval, resolve=True)
        pbar_dict = OmegaConf.to_container(config.pbar, resolve=True)
        if self.tracking_mode == 'steps':
            eval_dict["n_steps"] = pbar_dict.get("total")
        else:
            eval_dict["n_epochs"] = pbar_dict.get("total")
        self.early_stopping = U.EarlyStopping(
            **eval_dict,
            patience_type=self.tracking_mode,
            steps_per_epoch=pbar_dict.get('steps_per_epoch')
        )

        self.update_state(status=TrainingStatus.train)
        self.test_evaluate_fn = None
        self.pbar = U.ProgressBar(**pbar_dict, tracking_mode=self.tracking_mode)

        hyperparams = C.resolve_config(config)
        logger.log_hyperparameters(hyperparams)
        if self.output_config.save_config:
            C.save_config(config, self.output_config.save_paths['config_path'])

        self.timer.start()
        self.reset()

    def log_info(self, msg: str):
        """Log an informational message using logger API."""
        logger(msg)

    @validate_call
    def reset(self, epoch: Optional[int] = None, step: Optional[int] = None) -> None:
        """Mark start of new training epoch."""
        self.update_state(epoch=epoch or getattr(self._state, "start_epoch", 0))
        self.update_state(step=step or getattr(self._state, "start_step", 0))
        self.pbar.start_epoch(self.epoch, self.step)

    @validate_call
    def log_loss(self, loss: float, step: Optional[int] = None, epoch: Optional[int] = None, name: str = "Loss"):
        """Log step-level loss values during training."""
        logger.log_loss(loss, step=step or self.step, epoch=epoch or self.epoch, name=name)

    def _log_metrics_common(self, metrics: Dict[str, Union[int, float]], log_type: str = 'epochs') -> None:
        """Common logic for logging metrics and updating state."""
        self.update_state(latest_metrics=metrics.copy())
        step_value = self.step if log_type == 'steps' else self.epoch
        log_data = {'epoch': self.epoch, 'step': self.step, **metrics}
        logger.log(log_data, step=step_value)

        if self.primary_metric in metrics:
            metric_value = metrics[self.primary_metric]
            should_update_es = (self.tracking_mode == log_type)
            if should_update_es:
                self.early_stopping.update(metric_value)

            if self.is_better:
                self.update_state(
                    best_epoch=self.epoch,
                    best_step=self.step,
                    best_metric_value=metric_value,
                    best_metrics=metrics.copy()
                )
                logger.log_best_metric(metrics)
                logger(f"New best {self.primary_metric}: {metric_value:.4f} {self._format_progress('best')}")

        elif self.early_stopping._enable:
            raise KeyError(
                f"Primary metric '{self.primary_metric}' not found in calculated metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )

    @validate_call
    def log_epoch(self, metrics: Dict[str, Union[int, float]], epoch: Optional[int] = None) -> None:
        """Log epoch-level metrics and check for improvements."""
        epoch is not None and self.update_state(epoch=epoch)
        self._log_metrics_common(metrics, log_type='epochs')
        self.pbar.update_epoch_metrics(metrics)

    @validate_call
    def log_step(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """Log step-level metrics during training."""
        step is not None and self.update_state(step=step)
        self._log_metrics_common(metrics, log_type='steps')

    @property
    def should_stop(self) -> bool:
        """Check if early stopping criteria met."""
        if self.early_stopping.should_stop():
            self.update_state(status=TrainingStatus.terminate)
            logger(f"Early stop {self._format_progress()} evaluated by {self.primary_metric}")
            return True
        if self.early_stopping.should_end(self.step, self.epoch):
            self.update_state(status=TrainingStatus.terminate)
            logger(f"Finish training {self._format_progress(include_both=True)}")
            return True
        return False

    @property
    def is_better(self) -> bool:
        """Check if current epoch is the best seen so far."""
        if (is_better := self.early_stopping._is_better):
            self.pbar.update_best()
        return is_better

    def _update_progress(self, attr: str, value: int, is_relative: bool = False, from_start: bool = False) -> int:
        """Generic helper to update epoch or step."""
        old_val = getattr(self, attr)
        if is_relative:
            new_val = old_val + value
        elif from_start:
            new_val = getattr(self._state, f"start_{attr}") + value
        else:
            new_val = value
        setattr(self._state, f"current_{attr}", new_val)
        return new_val - old_val

    @validate_call
    def update_state(
        self,
        status: Optional[TrainingStatus] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        is_relative: bool = False,
        from_start: bool = False,
        best_epoch: Optional[int] = None,
        best_step: Optional[int] = None,
        best_metric_value: Optional[float] = None,
        latest_metrics: Optional[dict] = None,
        best_metrics: Optional[dict] = None
    ) -> None:
        """Update tracker state with new values."""
        if status is not None:
            self._state.status = status
        if epoch is not None:
            delta_epoch = self._update_progress("epoch", epoch, is_relative)
            self.pbar.update(epoch=delta_epoch)
        if step is not None:
            delta_step = self._update_progress("step", step, is_relative, from_start)
            self.pbar.update(step=delta_step)
        if best_epoch is not None:
            self._state.best_epoch = best_epoch
        if best_step is not None:
            self._state.best_step = best_step
        if best_metric_value is not None:
            self._state.best_metric_value = best_metric_value
        if latest_metrics is not None:
            self._state.latest_metrics = latest_metrics
        if best_metrics is not None:
            self._state.best_metrics = best_metrics

    def save_checkpoint(self, is_last=False, verbose=False, **state_components) -> None:
        """Save checkpoint with training state and model components.

        Example
        -------
        >>> tracker.save_checkpoint(model=model, optimizer=optimizer)
        >>> tracker.save_checkpoint(is_last=True, model=model)
        """
        if (save_paths := self.output_config.save_paths) == {}:
            return

        checkpoint_data = {
            'state': self._state.model_dump(),
            'timer': self.timer.__getstate__(),
            'early_stop': self.early_stopping.__getstate__(),
            'pbar': self.pbar.__getstate__() if self.pbar else None,
            'config': C.resolve_config(self.config),
            'rng_state': (ModelSerializer._capture_rng_state(), '.pkl'),
            **state_components
        }
        if self.is_better and 'best_model_path' in save_paths:
            self._save_state_components(checkpoint_data, save_paths['best_model_path'])
            verbose and logger(f"Saved best checkpoint {self._format_progress()}")

        if is_last and 'last_model_path' in save_paths:
            self._save_state_components(checkpoint_data, save_paths['last_model_path'])
            verbose and logger(f"Saved last checkpoint {self._format_progress()}")

    def _save_state_components(self, checkpoint_data: Dict[str, Any], base_path: Path) -> None:
        """Save individual state components using appropriate serializers."""
        (base_path := Path(base_path)).mkdir(parents=True, exist_ok=True)
        for key, value in checkpoint_data.items():
            if value is None:
                continue
            component_path = base_path / f"{key}.pt"
            try:
                ModelSerializer.save(value, component_path)
            except Exception as e:
                logger(f"Failed to save {key}: {e}")

    def load_checkpoint(self, checkpoint_path: Path, load_config: bool = False, **target_objects) -> Dict[str, Any]:
        """Load checkpoint and return state dictionary.

        Example
        -------
        >>> state = tracker.load_checkpoint(Path("checkpoint"), model=model, optimizer=optimizer)
        """
        loaded_state = {}
        checkpoint_path = Path(checkpoint_path)

        for file_path in checkpoint_path.glob("*"):
            if file_path.suffix not in ['.pt', '.joblib', '.pkl', '.json']:
                continue
            component_name = file_path.stem
            try:
                loaded_obj = ModelSerializer.load(file_path)
                if component_name in target_objects and file_path.suffix == '.pt':
                    target_objects[component_name].load_state_dict(loaded_obj)
                    loaded_state[component_name] = target_objects[component_name]
                else:
                    loaded_state[component_name] = loaded_obj
                self._load_sanity_check(loaded_state[component_name], component_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load {component_name}") from e

        if load_config and 'config' in loaded_state:
            self.config = OmegaConf.create(loaded_state['config'])

        return loaded_state

    def _load_sanity_check(self, obj: Any, component_name: str) -> None:
        """Sanity check loaded objects to ensure they are legitimate."""
        match component_name:
            case 'state':
                required_keys = ['current_epoch', 'current_step', 'best_epoch', 'start_epoch', 'start_step']
                if not all(key in obj for key in required_keys):
                    raise ValueError("Training state missing required keys")
            case 'early_stop':
                if "_best_value" not in obj and "_n_consequtive_bad_updates" not in obj:
                    raise ValueError("Early stop state missing required keys")
            case 'timer':
                required_keys = ['total_elapsed', 'begin_time']
                if not all(key in obj for key in required_keys):
                    raise ValueError("Timer state missing required keys")
            case 'config':
                if not isinstance(obj, dict):
                    raise ValueError("Config must be a dictionary")
            case _:
                if hasattr(obj, 'get_params') and hasattr(obj, 'fit'):
                    from sklearn.utils.validation import check_is_fitted
                    check_is_fitted(obj)
                elif hasattr(obj, 'state_dict') and hasattr(obj, 'parameters'):
                    param_count = sum(p.numel() for p in obj.parameters())
                    if param_count == 0:
                        raise ValueError(f"PyTorch model {component_name} has no parameters")
                elif hasattr(obj, 'param_groups') and hasattr(obj, 'state'):
                    if not obj.param_groups or obj.param_groups[0].get('lr', 0) <= 0:
                        raise ValueError(f"PyTorch optimizer {component_name} has invalid state")
                elif hasattr(obj, 'get_lr') and hasattr(obj, 'optimizer'):
                    if obj.optimizer is None or not obj.get_lr() or any(lr <= 0 for lr in obj.get_lr()):
                        raise ValueError(f"PyTorch scheduler {component_name} has invalid state")

    def end_epoch(self) -> None:
        """Mark end of epoch and update timing."""
        label = self._get_primary_progress_label().capitalize()
        value = self._get_primary_progress_value()
        self.pbar.set_description(f"{label} {value} completed")

    def finalize(self) -> None:
        """Finalize tracking and cleanup."""
        self.timer.stop()
        self.pbar.close()

        if self.best_metrics and self.primary_metric in self.best_metrics:
            best_metric = self.best_metrics[self.primary_metric]
            logger(f"Training completed. Best {self.primary_metric}: {best_metric:.4f}")
        else:
            logger(f"Training completed (no metrics recorded)")

        save_paths = self.output_config.save_paths
        import shutil

        if self.output_config.delete_best and 'best_model_path' in save_paths:
            best_path = Path(save_paths['best_model_path'])
            if best_path.exists():
                shutil.rmtree(best_path)
                logger(f"Deleted best checkpoint: {best_path}")

        if self.output_config.delete_last and 'last_model_path' in save_paths:
            last_path = Path(save_paths['last_model_path'])
            if last_path.exists():
                shutil.rmtree(last_path)
                logger(f"Deleted last checkpoint: {last_path}")

    def should_resume(self) -> bool:
        """Check if should resume from existing checkpoint."""
        if self.config.resume_mode != "resume":
            return False
        if self.config.resume_from not in ["best_model", "last_model"]:
            return False

        checkpoint_type = self.config.resume_from.replace("_model", "")
        ckpt_path = self.output_config.save_paths.get(f"{checkpoint_type}_model_path")
        return ckpt_path and Path(ckpt_path).exists()

    def resume_checkpoint(self, load_config: bool = False, weights_only: bool = False, **kwargs: Unpack[Checkpoints]) -> Tuple[Any, ...]:
        """Resume training from a saved checkpoint.

        Example
        -------
        >>> model, optimizer = tracker.resume_checkpoint(model=model, optimizer=optimizer)
        """
        checkpoint_type = self.config.resume_from.replace("_model", "")
        ckpt_path = self.output_config.save_paths.get(f"{checkpoint_type}_model_path")
        if not ckpt_path or not ckpt_path.exists():
            logger(f"{checkpoint_type} checkpoint not found, starting fresh")
            return kwargs.values()

        loaded_state = self.load_checkpoint(ckpt_path, load_config=load_config, **kwargs)

        if not weights_only:
            if training_state := loaded_state.get("state", None):
                training_state['start_epoch'] = training_state['current_epoch']
                training_state['start_step'] = training_state['current_step']
                self._state = TrainingState(**training_state)

            if es_state := loaded_state.get("early_stop", None):
                self.early_stopping.__setstate__(es_state)

            if ts := loaded_state.get("timer", None):
                self.timer.__setstate__(ts)

            if pbar_state := loaded_state.get("pbar", None):
                if self.pbar:
                    self.pbar.__setstate__(pbar_state)

            if rng_state := loaded_state.get("rng_state", None):
                ModelSerializer._restore_rng_state(rng_state)

        resumed_objects = [loaded_state.get(k, v) for k, v in kwargs.items()]

        if weights_only:
            logger(f"Loaded {checkpoint_type} model weights (state preserved)")
        else:
            epoch, step = self._state.start_epoch, self._state.start_step
            logger(f"Resumed from {checkpoint_type} checkpoint at [Epoch {epoch:03} / Step {step:03}]")
        if len(resumed_objects) == 1:
            return resumed_objects[0]
        return resumed_objects

    def add_evaluate_function(self, evaluate_fn: Callable, dataset=None) -> None:
        """Add evaluation function for test data assessment.

        Example
        -------
        >>> tracker.add_evaluate_function(evaluate_fn, test_dataset)
        >>> metrics = tracker.evaluate_data()
        """
        sig = inspect.signature(evaluate_fn)
        if "dataset" not in sig.parameters:
            raise ValueError("evaluate_fn must accept a 'dataset' parameter")
        self.test_evaluate_fn = evaluate_fn
        self._default_dataset = dataset

    def evaluate_data(self, dataset: Optional[object] = None, **kwargs) -> Dict[str, float]:
        """Evaluate model on dataset using registered evaluation function."""
        dataset_to_use = dataset or self._default_dataset
        if self.test_evaluate_fn is None or dataset_to_use is None:
            return {}

        original_status = self.status
        self.update_state(status=TrainingStatus.test)

        self.inference_timer.start()
        metrics = self.test_evaluate_fn(dataset=dataset_to_use, **kwargs)

        self.inference_timer.stop()
        self.update_state(status=original_status)
        return metrics

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary report."""
        self.timer.stop()
        time_info = dict(re.findall(
            pattern=r"(begin|elapsed)=([\d\-: :\.]+)",
            string=repr(self.timer)
        ))
        total_epochs = max(1, self.epoch)
        time_per_epoch = self.timer.total() / total_epochs if total_epochs > 0 else 0
        infer_per_epoch = self.inference_timer.total() / total_epochs if total_epochs > 0 else 0

        summary = SummaryReport(
            eval_config=C.resolve_config(self.eval_config),
            output_config=C.resolve_config(self.output_config),
            model_info=C.resolve_config(self.model_config),
            execute_info={
                "time_per_epoch": time_per_epoch,
                "infer_per_epoch": infer_per_epoch,
                "total_train_time": self.timer.total(),
                "total_infer_time": self.inference_timer.total(),
                **time_info
            },
            result_info={
                "last_epoch": self.epoch,
                "last_metrics": self.latest_metrics,
                "best_epoch": self.best_epoch,
                "best_step": self.best_step,
                "best_metrics": self.best_metrics,
            },
        )
        logger.info(f"Training summary generated: {summary.result_info}")
        if report_path := self.output_config.save_paths.get('report_path'):
            U.save_json(summary.model_dump(), report_path)
            logger(f"Summary saved to {report_path}")

        return summary.model_dump()

    # ─── Private Helpers ─────────────────────────────────────────────────

    def _get_primary_progress_value(self) -> int:
        return self.step if self.tracking_mode == 'steps' else self.epoch

    def _get_primary_progress_label(self) -> str:
        return 'step' if self.tracking_mode == 'steps' else 'epoch'

    def _format_progress(self, context: Literal['current', 'best', 'resume'] = 'current', include_both: bool = False) -> str:
        if context == 'best':
            epoch_val = self.best_epoch
            step_val = self.best_step
        else:
            epoch_val = self.epoch
            step_val = self.step

        if include_both:
            return f"at epoch {epoch_val} / step {step_val}"
        elif self.tracking_mode == 'steps':
            return f"at step {step_val}"
        else:
            return f"at epoch {epoch_val}"

    def __repr__(self):
        def truncate(d, n=100):
            s = str(d)
            return s if len(s) <= n else s[:n] + "..."

        ec = self.eval_config
        eval_line = f"primary_metric={ec.primary_metric}, patience={ec.patience}, buffer={ec.buffer}, direction={ec.min_delta}, min_delta={ec.min_delta}"
        oc = self.output_config
        save_model = (",".join(filter(None, ["best" if oc.save_best else "", "last" if oc.save_final else ""])) or "no")
        save_pred = "best" if oc.save_pred else "no"
        output_line = f"folder={oc.folder}, save_model={save_model}, save_pred={save_pred}"
        test_eval = self.test_evaluate_fn.__name__ if self.test_evaluate_fn else None
        return (
            f"{self.__class__.__name__}(\n"
            f"  config={truncate(self.config)}\n"
            f"  eval_config=({eval_line})\n"
            f"  output_config=({output_line})\n"
            f"  model_config={truncate(self.model_config)}\n"
            f"  state={self._state!r}\n"
            f"  timer={self.timer!r}\n"
            f"  inference_timer={self.inference_timer!r}\n"
            f"  pbar={self.pbar!r}\n"
            f"  early_stopping={self.early_stopping!r}\n"
            f"  test_evaluate_fn={test_eval}\n"
            f")"
        )

    def __rshift__(self, other: Union[str, int]):
        """Increment epoch or step using the right shift operator.

        Example
        -------
        >>> tracker >> 5           # Increment epoch by 5
        >>> tracker >> "3 epochs"  # Increment epoch by 3
        >>> tracker >> "10 steps"  # Increment step by 10
        """
        try:
            unit_type, value = U.parse_epoch_step_string(other, default_type=self.tracking_mode)
            if unit_type == 'epochs':
                self.update_state(epoch=value, is_relative=True)
            else:
                self.update_state(step=value, is_relative=True)
        except ValueError as e:
            raise ValueError(f"Invalid format: '{other}'. {str(e)}") from e


__all__ = [
    "TrainingState",
    "TuningState",
    "SummaryReport",
    "ModelSerializer",
    "Checkpoints",
    "Tracker",
]
