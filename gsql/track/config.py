"""
Configuration management for ML experiments.

Provides type-safe configs using springs/OmegaConf with YAML loading,
variable substitution, and hierarchical merging.
"""
from . import enums as T
from .enums import Direction, LogMode, SampleMode, PruneMode
import yaml
import json
import os
import springs as sp
from typing import Optional, Dict, List, Any, Type, TypeVar, Union, TypeAlias
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import is_dataclass, fields
from enum import Enum

from .log import get_logger
from . import util as U

#>>> OmegaConf custom resolvers <<<
try:
    OmegaConf.register_new_resolver("range", lambda *args: ["range", *args])
except ValueError:
    pass  # Already registered
try:
    OmegaConf.register_new_resolver("choice", lambda *args: ["choice", *args])
except ValueError:
    pass
try:
    OmegaConf.register_new_resolver("int", lambda x: ["int", x])
except ValueError:
    pass
try:
    OmegaConf.register_new_resolver("tag", lambda name, val: [name, val])
except ValueError:
    pass
try:
    OmegaConf.register_new_resolver("interval", lambda start, end: [start, end])
except ValueError:
    pass
except ValueError: pass  # Already registered

#>>> YAML custom tags <<<
def _important_constructor(loader, node):
    if isinstance(node, yaml.ScalarNode):
        value = U.parse_numeric(loader.construct_scalar(node))
        return {"_value": value, "_important": True}
    value = loader.construct_mapping(node, deep=True)
    value["_terminal"] = True
    return value
yaml.SafeLoader.add_constructor('!important', _important_constructor)

def _tune_constructor(loader, node):
    if isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
        return ["!tune"] + list(value)
    else:
        value = loader.construct_scalar(node)
        return ["!tune", value]

yaml.SafeLoader.add_constructor('!tune', _tune_constructor)
yaml.SafeLoader.add_constructor('!tune:choice', _tune_constructor)
yaml.SafeLoader.add_constructor('!tune:range', _tune_constructor)
yaml.SafeLoader.add_constructor('!tune:log', _tune_constructor)

G = TypeVar('G')
InputConfig: TypeAlias = Union[DictConfig, ListConfig]
DeviceConfig: TypeAlias = Union[list, int]

#>>> Config dataclasses <<<
@sp.dataclass
class LogConfig:
    """Logging configuration with flexible config dict.

    Example
    -------
    >>> cfg = LogConfig(mode=LogMode.wandb, name='my_project')
    """
    mode: LogMode = sp.field( default=T.LogMode.logger, help='which logger to use')
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

@sp.dataclass
class DataConfig:
    """Data processor configuration.

    Example
    -------
    >>> cfg = DataConfig(data_class='wrench_data', init_args={'name': 'yelp'})
    """
    init_args: Optional[dict] = sp.field(default_factory=dict)
    process_args: Optional[dict] = sp.field(default_factory=dict)
    data_class: Optional[str] = None

@sp.dataclass
class TaskConfig:
    """Task/dataset configuration for tuning and benchmarking.

    Example
    -------
    >>> task = TaskConfig(name='yelp')
    """
    name: str = sp.field(help="Task identifier")
    data: Dict[str, Any] = sp.field(
        default_factory=dict,
        help="Data specification (data_class, init_args, process_args)"
    )
    metrics: List[str] = sp.field(
        default_factory=lambda: ['acc', 'f1'],
        help="[Deprecated] Unused — display is controlled by eval.display_metric. Will be removed in a future version."
    )
    config: Optional[Dict[str, Any]] = sp.field(
        default=None,
        help="Task-level config overrides"
    )

    def __getattribute__(self, name):
        if name == 'metrics':
            import warnings
            warnings.warn(
                "TaskConfig.metrics is deprecated and unused. "
                "Display is controlled by eval.display_metric instead. "
                "This field will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getattribute__(name)

@sp.dataclass
class SamplerConfig:
    """Sampler configuration for hyperparameter tuning frameworks (Optuna/NNI).

    Example
    -------
    >>> cfg = SamplerConfig(sampler=SampleMode.tpe, pruner=PruneMode.median)
    >>> optuna_cfg = cfg.to_optuna()
    """
    sampler: SampleMode = sp.field(default=T.SampleMode.tpe)
    pruner: PruneMode = sp.field(default=T.PruneMode.nop)
    sampler_kws: Dict[str, Any] = sp.field(default_factory=dict)
    pruner_kws: Dict[str, Any] = sp.field(default_factory=dict)

    def to_optuna(self, direction: Direction = "maximize"):
        import optuna

        sampler_map = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "grid": optuna.samplers.GridSampler,
            "cmaes": optuna.samplers.CmaEsSampler,
        }
        pruner_map = {
            "median": optuna.pruners.MedianPruner,
            "nop": optuna.pruners.NopPruner,
            "patience": optuna.pruners.PatientPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
        }
        sampler_cls = sampler_map.get(self.sampler.lower())
        pruner_cls = pruner_map.get(self.pruner.lower())
        sampler_kwargs = self.sampler_kws.copy()
        return {
            "direction": direction,
            "sampler": sampler_cls(**sampler_kwargs) if sampler_cls else None,
            "pruner": pruner_cls(**self.pruner_kws) if pruner_cls else None,
        }


@sp.dataclass
class EvalConfig:
    """Evaluation and early stopping configuration.

    Note
    ----
    The interpretation of `patience` depends on TrainConfig.tracking_mode:
    - tracking_mode='epochs': values represent number of epochs
    - tracking_mode='steps': values represent number of steps

    Example
    -------
    >>> cfg = EvalConfig(primary_metric='val.loss', patience=10, display_metric='acc')
    """
    primary_metric: str = sp.field(
        default="val.loss", help="Primary metric for model selection"
    )
    patience: Optional[int] = sp.field(
        default=None, help="Number of tracking units (epochs/steps) to wait before early stopping"
    )
    buffer: int = sp.field(
        default=0, help="Number of initial metric logs to buffer before counting bad updates for early stopping"
    )
    direction: Direction = sp.field(
        default=T.Direction.minimize, help="Direction for metric improvement"
    )
    min_delta: float = sp.field(
        default=0.0, help="Minimum change to qualify as improvement"
    )
    display_metric: Any = sp.field(
        default="acc", help="Metric(s) to display in results CSV. String or list of strings (e.g., 'acc' or ['test.acc', 'test.f1'])"
    )
    
@sp.dataclass
class ModelConfig:
    """Base model configuration - to be extended by specific baselines.

    Example
    -------
    >>> @sp.dataclass
    >>> class MyConfig(ModelConfig):
    ...     lr: float = 0.01
    >>> cfg = MyConfig(name='my_model', metric='f1')
    """
    name: str = sp.field(help='Name of the algorithm, affecting output folder')
    metric: str = sp.field(
        default="acc", help="Evaluation metric (e.g., 'acc', 'f1', or callable)"
    )


@sp.dataclass
class OutputConfig:
    """Output configuration for saving models, predictions, and analysis.

    Example
    -------
    >>> cfg = OutputConfig(folder='./outputs', save_model=True, save_pred=True)
    >>> cfg.save_paths['best_model_path']
    """
    folder: str = sp.field(help="Output directory path")
    save_model: bool = sp.field(default=False, help="Save trained model to disk")
    save_model_only: bool = sp.field(
        default=False, help="Save trained model only without other stuff"
    )
    save_pred: bool = sp.field(default=False, help="Save model predictions to disk")
    save_final: bool = sp.field(
        default=False, help="Save final model/pred after training"
    )
    save_best: bool = sp.field(
        default=True, help="Save best model/pred based on validation metric"
    )
    delete_best: bool = sp.field(
        default=True, help="Delete best model checkpoint after training"
    )
    delete_last: bool = sp.field(
        default=True, help="Delete last model checkpoint after training"
    )
    save_config: bool = sp.field(default=False, help="Save training configuration")
    save_analysis: bool = sp.field(
        default=False, help="Save extra failure cases analysis"
    )
    save_paths: dict = sp.field(
        default_factory=dict,
        help="Paths for saving models, predictions, config, and analysis",
    )

    def __post_init__(self):
        """Create output folder and return all configured paths"""
        (folder := Path(self.folder)).mkdir(exist_ok=True, parents=True)
        self.save_paths = {"report_path": folder / "report.json"}

        if self.save_config:
            self.save_paths["config_path"] = folder / "config.yaml"
        if self.save_analysis:
            self.save_paths["analysis_path"] = folder / "analysis.json"
        if self.save_model and self.save_best:
            self.save_paths["best_model_path"] = folder / "best_model"
        if self.save_model and self.save_final:
            self.save_paths["last_model_path"] = folder / "last_model"
        if self.save_pred and self.save_best:
            self.save_paths["best_pred_path"] = folder / "best_pred.pkl"
        if self.save_pred and self.save_final:
            self.save_paths["last_pred_path"] = folder / "last_pred.pkl"

@sp.dataclass
class SlackConfig:
    """Slack notification configuration.

    Example
    -------
    >>> cfg = SlackConfig(agent='test1', enabled=True)
    """
    enabled: bool = sp.field(
        default=False, metadata={"help": "Enable Slack notifications"}
    )
    webhook_url: Optional[str] = sp.field(
        default=None, metadata={"help": "Slack webhook URL for notifications"}
    )
    agent: Optional[str] = sp.field(
        default=None, metadata={"help": "Which slack application to send messages"}
    )

    def __post_init__(self):
        self.webhook_url = os.environ.get(f"GSQL_SLACK_WEBHOOK_{(self.agent or '').upper()}")
        if self.webhook_url is None:
            self.enabled = False

    def to_config(self):
        return {
            'enabled': self.enabled,
            'webhook_url': self.webhook_url
        }

@sp.dataclass
class StartConfig:
    """Base configuration for experiment startup behavior.

    Example
    -------
    >>> cfg = StartConfig(resume_mode='fresh', seed=42, pre_import_modules=['src.contrib'])
    """
    resume_mode: "ResumeMode" = sp.field(
        default="resume",
        help="Resume mode: 'fresh' (delete all), 'resume' (skip completed), 'append' (always run, keep existing)"
    )
    seed: int = sp.field(default=42, help="Random seed for reproducibility")
    pre_import_modules: Optional[List[str]] = sp.field(
        default=None,
        help="Modules to import at startup (e.g., ['src.contrib'] to register data classes)"
    )


class ResumeMode(str, Enum):
    """Resume mode for experiment execution.

    Example
    -------
    >>> ResumeMode.fresh     # Delete all and start from scratch
    >>> ResumeMode.resume    # Skip completed runs
    >>> ResumeMode.append    # Always run without deleting
    """
    fresh = "fresh"
    resume = "resume"
    append = "append"


def resolve_resume_mode(value) -> 'ResumeMode':
    """Convert a string or ResumeMode to ResumeMode enum.

    Shared by bench_manager and tune_manager.

    Example
    -------
    >>> resolve_resume_mode("fresh")
    <ResumeMode.fresh: 'fresh'>
    """
    return value if isinstance(value, ResumeMode) else ResumeMode(value)


@sp.dataclass
class PbarConfig:
    """Progress bar configuration.

    Note
    ----
    The interpretation of `total` depends on TrainConfig.tracking_mode:
    - tracking_mode='epochs': total represents number of epochs
    - tracking_mode='steps': total represents number of steps

    Example
    -------
    >>> cfg = PbarConfig(enabled=True, total=100, steps_per_epoch=10)
    """
    enabled: bool = sp.field(
        default=False, help="Enable progress bar display during training"
    )
    total: Optional[int] = sp.field(
        default=None, help="Total number of tracking units (epochs or steps based on tracking_mode)"
    )
    steps_per_epoch: Optional[int] = sp.field(
        default=None, help="Number of steps per epoch (required for step-based tracking)"
    )
    epoch_metrics: list = sp.field(
        default_factory=list, help="List of metric names to display in progress bar"
    )
    persist: bool = sp.field(
        default=False, help="Keep progress bar displayed after completion"
    )
    
@sp.dataclass(kw_only=True)
class TrainConfig(StartConfig):
    """Configuration for model training.

    Example
    -------
    >>> cfg = TrainConfig(model=model_config, data=data_config, output=output_config)
    """
    model: ModelConfig
    data: DataConfig = sp.field(help="Dataset configuration")
    output: OutputConfig = sp.field(help="Output configuration")
    log: Optional[LogConfig] = sp.field( default_factory=LogConfig, help="Logging configuration")
    eval: EvalConfig = sp.field( default_factory=EvalConfig, help="Help evaluate model during training")
    pbar: PbarConfig = sp.field( default_factory=PbarConfig, help="Progress bar related features")
    resume_from: ResumeMode = ( sp.field(default=T.ResumeMode.best_model, help="Which model/pred to load from"))
    exp_name: str = sp.field( default="E01", metadata={"help": "Name to save experiment result under the same algorithm/data"})
    tracking_mode: Optional[T.TrackingMode] = sp.field(
        default=None,
        help="Tracking granularity: 'epochs' or 'steps'. Auto-inferred if None"
    )

    def __post_init__(self):
        """Infer and validate tracking mode consistency"""
        if self.tracking_mode is None:
            assert self.eval
            self.tracking_mode = self._infer_tracking_mode()
        else:
            self.tracking_mode = T.TrackingMode.get(self.tracking_mode)
        self._validate_tracking_mode()

    def _infer_tracking_mode(self) -> str:
        """Infer tracking mode from config with clear precedence.

        Priority:
        1. If pbar.total is set with steps_per_epoch → 'steps'
        2. Default → 'epochs'
        """
        if self.pbar.total is not None and self.pbar.steps_per_epoch is not None:
            return T.TrackingMode.steps.value
        return T.TrackingMode.epochs.value

    def _validate_tracking_mode(self):
        """Validate that configs are consistent with tracking mode"""
        issues = []
        if issues:
            raise ValueError(
                "Tracking mode configuration conflicts detected:\n" +
                "\n".join(f"  - {issue}" for issue in issues)
            )


#>>> Config utility functions <<<
def save_config(config: sp.dataclass, path):
    U.ensure_dir(path)
    with open(path, "w") as f:
        f.write(sp.to_yaml(config))

def create_config(config: Dict | List) -> InputConfig:
    """Create OmegaConf config from dict or list."""
    return OmegaConf.create(config)

def resolve_config(config: InputConfig, resolve=True) -> InputConfig:
    """Resolve OmegaConf configuration to a standard Python container.

    Example
    -------
    >>> cfg = create_config({"a": 1, "b": "${a}"})
    >>> resolve_config(cfg)
    {'a': 1, 'b': 1}
    """
    if not isinstance(config, InputConfig):
        config = create_config(config)
    return OmegaConf.to_container(config, resolve=resolve, enum_to_str=True)

def load_config(config_path):
    """Load config from YAML file."""
    unsolved_config = sp.from_file(config_path)
    config = resolve_config(unsolved_config)
    return create_config(config)

def _parse_type(field_type):
    """Return base type from Optional or Union type."""
    import typing
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        args = typing.get_args(field_type)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
    return field_type

def _parse_config(data: dict, cls: Type[G]) -> G:
    """Recursively instantiate a dataclass from a nested dict."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        value = data[field.name]
        field_type = _parse_type(field.type)
        if field_type and is_dataclass(field_type) and isinstance(value, dict):
            kwargs[field.name] = _parse_config(value, field_type)
        else:
            kwargs[field.name] = value
    return cls(**kwargs)

def set_readonly(cfg: InputConfig, flag: bool = False):
    """Recursively set OmegaConf config readonly flag."""
    OmegaConf.set_readonly(cfg, flag)
    if isinstance(cfg, DictConfig):
        for v in cfg.values():
            isinstance(v, InputConfig) and set_readonly(v, flag)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            isinstance(v, InputConfig) and set_readonly(v, flag)


def parse_config(
    config: Any,
    config_class: Optional[type] = None,
    overrides: Optional[Dict] = None,
) -> Any:
    """Parse a dict or OmegaConf object into a structured config, applying overrides.

    Example
    -------
    >>> cfg = create_config({'a': 1})
    >>> parse_config(cfg, overrides={'a': 2}).a
    """
    if not isinstance(config, InputConfig):
        config = create_config(config)
    if overrides:
        overrides_flat = U.flat_dict(overrides)
        config = sp.merge(config, sp.from_dict(overrides_flat))
    target_class = config_class or OmegaConf.get_type(config)
    if isinstance(config, target_class):
        return config
    config_dict = resolve_config(config)
    structured = _parse_config(config_dict, target_class)
    return OmegaConf.structured(structured)


#>>> Experiment lifecycle <<<
def start_experiment(config: DictConfig, output_folder: Path) -> bool:
    """Start an experiment with the given configuration."""
    import time
    import shutil

    logger = get_logger()
    logger(f"🏁 [>>>] {output_folder} | {datetime.now()}")
    U.stop_when_error()

    resume_mode = config.resume_mode
    if isinstance(resume_mode, str):
        resume_mode = ResumeMode(resume_mode)

    if (out_dir := Path(output_folder)).exists():
        if resume_mode == ResumeMode.fresh:
            logger("⚠️  Fresh mode: Removing the existing output")
            time.sleep(1.0)
            shutil.rmtree(out_dir)
            out_dir.mkdir()
            return True
        elif out_dir.joinpath("DONE").exists() and resume_mode == ResumeMode.append:
            logger("✅ Already done!")
            exit(0)
        else:
            logger(f"➡️ {resume_mode.value.upper()} mode: Continuing with the existing output")
            return True
    else:
        logger("♻️ Creating the output")
        out_dir.mkdir(parents=True, exist_ok=True)
        return True


def finish_experiment(output_folder: Path) -> None:
    """Mark an experiment as finished."""
    logger = get_logger()
    Path(output_folder).joinpath("DONE").touch()
    logger("-" * 80)
    logger(f"🎉 [<<<] {output_folder} | {datetime.now()}")
    logger.finish()


#>>> Hierarchical config merging with terminal markers <<<#
def merge_hierarchical_config(
    model_config: Optional[Dict],
    task_config: Optional[Dict],
    global_config: Optional[Dict]
) -> Dict:
    """Deep-merge configs with last-write-wins priority: model > task > global.

    Merge order: global first, then task overwrites, then model overwrites last.
    For nested dicts, keys are merged recursively — a higher-priority config only
    overwrites the specific keys it defines, not the entire sub-dict.

    Supports _terminal markers to prevent deep merging at specific keys.

    Args:
        model_config: Model-specific config (highest priority, merged last)
        task_config: Task-specific config (medium priority)
        global_config: Global config (lowest priority, merged first)

    Returns:
        Merged configuration dictionary

    Example:
        >>> global_cfg = {'lr': 0.001, 'optimizer': {'name': 'adam', 'betas': [0.9, 0.999]}}
        >>> task_cfg = {'lr': 0.01}
        >>> model_cfg = {'optimizer': {'name': 'sgd'}}
        >>> merge_hierarchical_config(model_cfg, task_cfg, global_cfg)
        {'lr': 0.01, 'optimizer': {'name': 'sgd', 'betas': [0.9, 0.999]}}
        # lr=0.01 from task (overwrites global's 0.001)
        # optimizer.name='sgd' from model (overwrites global's 'adam')
        # optimizer.betas from global (no override from task or model)
    """
    terminal_keys, result = set(), {}
    for cfg in [model_config, task_config, global_config]:
        cfg and terminal_keys.update(_find_terminal_markers(cfg))
    for cfg in [global_config, task_config, model_config]:
        cfg and _merge_with_terminals(result, resolve_config(cfg), terminal_keys)
    return result


def _find_terminal_markers(dct: Dict, prefix: str = "") -> List[str]:
    """Find all keys marked with _terminal to prevent deep merging."""
    dct, markers = dct or {}, []
    for k, v in dct.items():
        key_path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "_terminal" in v:
            markers.append(key_path)
        elif isinstance(v, dict):
            markers.extend(_find_terminal_markers(v, key_path))
    return markers


def _merge_with_terminals(dct: Dict, override: Dict, terminal_keys: set):
    """Merge override into dct, respecting terminal markers."""
    for k, v in override.items():
        if k in terminal_keys or (isinstance(v, dict) and "_terminal" in v):
            clean_v = {k2: v2 for k2, v2 in v.items() if k2 != "_terminal"} if isinstance(v, dict) else v
            dct[k] = clean_v
        elif k in dct and isinstance(dct[k], dict) and isinstance(v, dict):
            _merge_with_terminals(dct[k], v, terminal_keys)
        else:
            dct[k] = v
    return dct


#>>> Generate debug command for ipdb debugging <<<#
def generate_debug_cmd(function_path: str, config: Dict[str, Any]) -> str:
    """Create ready-to-use ipdb command with hashed config for debugging.

    Args:
        function_path: Function path (e.g., 'examples.train_model')
        config: Configuration dictionary

    Returns:
        Command string for ipdb debugging

    Example:
        >>> cmd = generate_debug_cmd('examples.train_model', {'lr': 0.001})
        >>> print(cmd)
        PYTHONPATH="." python -m ipdb examples/train_model.py -c /tmp/debug_configs/a1b2c3d4/config.yaml
    """
    import hashlib
    import springs as sp

    parts = function_path.split('.')
    module_path = '/'.join(parts[:-1]) + '.py'
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    temp_dir = Path("/tmp") / "debug_configs" / config_hash
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "config.yaml"
    if not temp_path.exists():
        temp_path.write_text(sp.to_yaml(create_config(config)))
    return f'PYTHONPATH="." python -m ipdb {module_path} -c {temp_path}'


#>>> Resolve per-task eval overrides (task.config.eval > merged config defaults) <<<#
def _normalize_display_metric(value) -> list:
    """Normalize display_metric to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [str(value)]


def resolve_task_eval(
    task: 'TaskConfig',
    default_metric: str,
    default_direction: str,
    default_display_metric=None,
) -> dict:
    """Resolve eval settings for a task, reading task.config.eval directly.

    Task-level eval overrides (primary_metric, direction, display_metric) take
    priority over model/global defaults. This is necessary because different tasks
    use different metrics (e.g. acc for multi-class, f1 for binary), and the
    standard merge_hierarchical_config gives model priority over task.

    Args:
        task: TaskConfig with optional config.eval overrides
        default_metric: Fallback primary_metric (from model or global config)
        default_direction: Fallback direction
        default_display_metric: Fallback display_metric (str or list, defaults to default_metric)

    Returns:
        Dict with keys: primary_metric, direction, display_metric (list of str)
    """
    metric = default_metric
    direction = default_direction
    display_metric = _normalize_display_metric(default_display_metric) or [default_metric]
    if task.config:
        tc = task.config if isinstance(task.config, dict) else dict(task.config)
        eval_cfg = tc.get('eval', {})
        if eval_cfg.get('primary_metric'):
            metric = eval_cfg['primary_metric']
        if eval_cfg.get('direction'):
            direction = eval_cfg['direction']
        if eval_cfg.get('display_metric'):
            display_metric = _normalize_display_metric(eval_cfg['display_metric'])
    return {
        'primary_metric': metric,
        'direction': direction,
        'display_metric': display_metric,
    }


__all__ = [
    "LogConfig",
    "DataConfig",
    "TaskConfig",
    "SamplerConfig",
    "EvalConfig",
    "OutputConfig",
    "SlackConfig",
    "ModelConfig",
    "StartConfig",
    "PbarConfig",
    "TrainConfig",
    "ResumeMode",
    "resolve_resume_mode",
    "save_config",
    "load_config",
    "set_readonly",
    "create_config",
    "parse_config",
    "resolve_config",
    "start_experiment",
    "finish_experiment",
    "merge_hierarchical_config",
    "generate_debug_cmd",
    "resolve_task_eval",
]