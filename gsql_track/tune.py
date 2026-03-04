"""
Hyperparameter tuning framework with Optuna integration.

Supports backend-agnostic search space definitions, seamless integration
with existing training code, and progress monitoring.
"""
import re
import time
import json
import shutil
import tempfile
import warnings
import operator
import numpy as np
import springs as sp
import pandas as pd

from pathlib import Path
from typing import Any, Optional, Union, Dict, TypeAlias, List
from omegaconf import DictConfig, ListConfig

import optuna
import optuna.trial
import requests

from . import config as C
from . import db as DB
from . import dispatch as DM
from . import plan as PM
from . import util as U
from .gsql_track import GsqlTrack
from .enums import Direction
from .tracker import TuningState
from .log import Logger

TaskConfig = C.TaskConfig
ProgressWriter = U.ProgressWriter

try:
    import torch
except ImportError:
    torch = None

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

SpaceDict: TypeAlias = Dict[str, Union["SpaceDict", List[str]]]

logger: Logger = None


# ─── Configuration classes ──────────────────────────────────────────────────

@sp.dataclass
class ModelTuneSpec:
    """Model specification with tunable parameters marked with !tune."""
    name: str = sp.field(help="Model identifier")
    function: str = sp.field(help="Training function path")
    train_config: Dict[str, Any] = sp.field(default_factory=dict, help="Config with !tune markers")
    n_seeds: int = sp.field(default=1, help="Seeds per model+task")
    n_trials: Optional[int] = sp.field(default=None, help="Override global n_trials for this model")
    timeout: Optional[int] = sp.field(default=None, help="Override global timeout (seconds) for this model")
    data_config: Optional[Dict[str, Any]] = sp.field(default=None)


@sp.dataclass(kw_only=True)
class TuneConfig(C.StartConfig):
    """Configuration for hyperparameter tuning experiments."""
    models: List[ModelTuneSpec] = sp.field(help="Models to tune")
    tasks: List[TaskConfig] = sp.field(help="Tasks to tune on")
    output: Path = sp.field(help="Output directory for tuning results")
    n_trials: int = sp.field(default=50, help="Number of hyperparameter trials to run")
    timeout: int = sp.field(default=6000, help="Timeout in seconds for tuning experiment")
    metric: str = sp.field(default="accuracy", help="Metric to optimize during tuning")
    direction: Direction = sp.field(default=Direction.maximize, help="Direction to optimize during tuning")
    sampler: Optional[Dict[str, Any]] = sp.field(default=None, help="Optuna sampler configuration")
    slacker: Optional[Dict[str, Any]] = sp.field(default=None, help="Slack notification config")
    n_workers: int = sp.field(default=1, help="Parallel workers (1=sequential)")
    device_ids: Optional[List[int]] = sp.field(default=None, help="GPU device IDs for parallel execution")
    global_config: Optional[Dict[str, Any]] = sp.field(default=None, help="Global config overrides")
    show_progress: bool = sp.field(default=True, help="Show progress messages")
    auto_confirm: bool = sp.field(default=False, help="Auto-confirm jobs without interactive prompt")

    def __post_init__(self):
        self.output = Path(self.output)
        self.models = [ModelTuneSpec(**m) if isinstance(m, dict) else m for m in self.models]
        self.tasks = [TaskConfig(**t) if isinstance(t, dict) else t for t in self.tasks]
        self.n_trials_strict = False
        if isinstance(self.n_trials, dict) and self.n_trials.get('_important'):
            self.n_trials_strict = True
            self.n_trials = self.n_trials['_value']


# ─── SQLite database utilities ──────────────────────────────────────────────

_get_db_connection = DB.get_db_connection
_init_tune_db = DB.tune_init_db
_get_job_id = DB.tune_get_job_id
_save_job_config = DB.tune_save_job_config
_start_job = DB.tune_start_job
_finish_job = DB.tune_finish_job
_fail_job = DB.tune_fail_job
_update_best_config = DB.tune_update_best_config
_update_job_stats = DB.tune_update_job_stats
_save_config = DB.tune_save_config
_get_config = DB.tune_get_config
get_failed_jobs = DB.tune_get_failed_jobs
get_best_config = DB.tune_get_best_config

_fmt_duration = U.fmt_duration


# ─── Helpers ────────────────────────────────────────────────────────────────

def _resolve_job_metric(task: 'TaskConfig', global_metric: str, global_direction: str) -> tuple:
    resolved = C.resolve_task_eval(task, global_metric, global_direction)
    return resolved['primary_metric'], resolved['direction']


def _save_best_result_from_study(study_path: Path, output_path: Path, model_name: str, task_name: str, metric_name: str = "metric"):
    if not study_path.exists():
        return
    try:
        study = optuna.load_study(study_name=f"{model_name}_{task_name}", storage=f"sqlite:///{study_path}")
        best = study.best_trial
        result_data = {
            'trial_number': best.number,
            'value': best.value,
            'metric_name': metric_name,
            'params': best.params,
            'duration': str(best.duration) if best.duration else None,
            'duration_human': best.user_attrs.get('duration_human'),
            'debug_cmd': best.user_attrs.get('debug_cmd'),
            'result': best.user_attrs.get('result'),
        }
        (best_results_dir := output_path / "best_results").mkdir(parents=True, exist_ok=True)
        (best_results_dir / f"{model_name}_{task_name}_best.json").write_text(
            json.dumps(result_data, cls=U.JSONEncoder, indent=2)
        )
    except Exception:
        pass


def _save_best_config_from_study(study_path: Path, output_path: Path, model_name: str, task_name: str):
    if not study_path.exists():
        return None
    try:
        study = optuna.load_study(study_name=f"{model_name}_{task_name}", storage=f"sqlite:///{study_path}")
        best_trial = study.best_trial
        if not (debug_cmd := best_trial.user_attrs.get('debug_cmd')):
            return None
        if not (match := re.search(r'-c\s+(\S+\.yaml)', debug_cmd)):
            return None
        if not (source_yaml := Path(match.group(1))).exists():
            return None
        (best_configs_dir := output_path / "best_configs").mkdir(parents=True, exist_ok=True)
        shutil.copy(source_yaml, best_configs_dir / f"{model_name}_{task_name}_best.yaml")
    except Exception:
        return None


def _update_state(db_path: Path, key: str, value: Any):
    DB.update_state(db_path, "tuning_state", key, value)

def _get_state(db_path: Path, key: str, default: Any = None) -> Any:
    return DB.get_state(db_path, "tuning_state", key, default)


# ─── Callbacks ──────────────────────────────────────────────────────────────

class SlackCallback:
    def __init__(self, webhook_url: str, function: str, metric: str, direction: Direction, exp_folder: str):
        self.webhook_url = webhook_url
        self.start_message = self._build_message(function, metric, direction, exp_folder)
        if direction == "maximize":
            self.best_value = float("-inf")
            self.op = operator.gt
        else:
            self.best_value = float("inf")
            self.op = operator.lt
        self._has_started = False

    @staticmethod
    def _build_message(function: str, metric: str, direction: Direction, exp_folder: str) -> str:
        emoji = "🚀" if direction == "maximize" else "📉"
        return {
            "text": f"{emoji} *Experiment Update*\n"
            f"> *Function:* `{function}`\n"
            f"> *Metric:* `{metric}` ({direction})\n"
            f"> *Experiment Folder:* `{exp_folder}`"
        }

    def __call__(self, study, trial):
        if not self._has_started:
            requests.post(self.webhook_url, json=self.start_message)
            self._has_started = True
        if self.op(study.best_value, self.best_value):
            self.best_value = study.best_value
            message = {
                "text": f"🎯 New best trial #{trial.number}!\n"
                f"Value: {trial.value:.4f}\n"
                f"Params: {trial.params}"
            }
            requests.post(self.webhook_url, json=message)


class ProgressCallback:
    """Optuna callback for real-time progress updates during study.optimize()."""
    def __init__(self, tracker: 'TuneProgressTracker', job_name: str, total_trials: int):
        self.tracker = tracker
        self.job_name = job_name
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        best_value = study.best_value if study.best_trial else None
        self.tracker.update(
            job_name=self.job_name,
            trial_num=completed,
            total_trials=self.total_trials,
            current_best=best_value
        )


# ─── Progress Tracking ─────────────────────────────────────────────────────

class TuneProgressTracker(U.BaseProgressTracker):
    """Track tuning progress with readable log output."""

    def __init__(self, total_jobs: int, writer: ProgressWriter, db_path: Path = None, n_trials: int = None):
        super().__init__(total=total_jobs, writer=writer)
        self.total_jobs = total_jobs
        self.total_trials = n_trials
        self.db_path = db_path

    def update(self, job_name: str, trial_num: int, total_trials: int, current_best: float = None):
        with self._lock:
            elapsed = time.time() - self.start_time
            static_part = f"{self.completed}|{job_name}|{trial_num}|{current_best}"
            if static_part != self.last_static_part:
                self.last_static_part = static_part
                eta = self.calculate_eta(elapsed)
                task_part = f"⏳ {job_name} (trial {trial_num}/{total_trials})"
                if current_best is not None:
                    task_part += f" best={current_best:.4f}"
                progress_msg = f"\r[✓ {self.completed}/{self.total_jobs} | {task_part}] {self.fmt_time(elapsed)}{eta}"
                self.writer.write_progress(progress_msg, overwrite=True)

    def complete_job(self, job_name: str, best_value: float, n_trials: int, stats: Dict = None, study_path: Path = None):
        with self._lock:
            self.completed += 1
            if stats is None and study_path and study_path.exists():
                stats = self._get_job_trial_stats(study_path, job_name)
            if stats:
                msg = (f"✓ Completed {job_name} ({self.completed}/{self.total_jobs}, "
                       f"mean={stats['mean']:.4f}±{stats['std']:.4f}, n={stats['n']}, "
                       f"best={best_value:.4f})")
            else:
                msg = f"✓ Completed {job_name} ({self.completed}/{self.total_jobs}, n={n_trials}, best={best_value:.4f})"
            self.writer.write_event(msg)
            self.last_static_part = ""

    def fail_job(self, job_name: str, error: str):
        with self._lock:
            self.completed += 1
            self.writer.write_event(f"✗ Failed {job_name} ({self.completed}/{self.total_jobs}): {error}")
            self.last_static_part = ""

    def _get_job_trial_stats(self, study_path: Path, job_name: str) -> Optional[Dict[str, Any]]:
        try:
            if len(parts := job_name.split(" on ")) != 2:
                return None
            model_name, task_name = parts
            study_name = f"{model_name}_{task_name}"
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                return None
            values = [t.value for t in completed_trials]
            metric_name = "metric"
            if self.db_path and self.db_path.exists():
                with _get_db_connection(self.db_path) as conn:
                    row = conn.execute(
                        "SELECT metric FROM tuning_jobs WHERE model_name=? AND task_name=?",
                        (model_name, task_name)
                    ).fetchone()
                    if row:
                        metric_name = row['metric']
            return {'mean': np.mean(values), 'std': np.std(values), 'metric_name': metric_name, 'n': len(values)}
        except Exception:
            return None

    def _fail_job(self, job_name: str, error: str):
        self.fail_job(job_name, error)

    def finalize(self):
        elapsed = time.time() - self.start_time
        self.writer.write_event(f"\n✓ Tuning Complete: {self.completed} jobs in {self.fmt_time(elapsed)}")


# ─── TuneJob & Tuner Classes ───────────────────────────────────────────────

class TuneJob:
    """Represents a single model+task tuning job."""
    def __init__(self, model: ModelTuneSpec, task: TaskConfig, config: TuneConfig):
        self.model = model
        self.task = task
        self.config = config
        self.search_space, self.base_config = self._extract_search_space()
        task_cfg = task.config or {}
        task_n_trials = task_cfg.get('n_trials')
        task_timeout = task_cfg.get('timeout')
        self.n_trials = task_n_trials if task_n_trials is not None else (model.n_trials if model.n_trials is not None else config.n_trials)
        self.n_trials_strict = getattr(config, 'n_trials_strict', False)
        self.timeout = None if self.n_trials_strict else (task_timeout if task_timeout is not None else (model.timeout if model.timeout is not None else config.timeout))

    def _extract_search_space(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        space = {}
        base = {}
        def traverse(config, space_dict, base_dict, prefix=""):
            for k, v in config.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, DictConfig)):
                    space_dict[k] = {}
                    traverse(v, space_dict[k], base_dict, key)
                    if not space_dict[k]:
                        del space_dict[k]
                elif isinstance(v, (list, ListConfig)) and v and isinstance(v[0], str) and (v[0].startswith('!tune') or v[0] in ('choice', 'log', 'int', 'float', 'range')):
                    marker = v[0]
                    params = v[1:]
                    if marker == '!tune':
                        space_dict[k] = params
                    elif marker in ('!tune:choice', '!tune:range', '!tune:log'):
                        space_dict[k] = [marker.split(':')[1]] + params
                    elif marker in ('choice', 'log', 'int', 'float', 'range'):
                        space_dict[k] = list(v)
                    else:
                        space_dict[k] = params
                else:
                    base_dict[key] = v
        traverse(self.model.train_config, space, base)
        return space, base


class JobResult:
    def __init__(self, model: str, task: str, best_value: float, best_params: Dict, n_trials: int, metric: str = ""):
        self.model = model
        self.task = task
        self.best_value = best_value
        self.best_params = best_params
        self.n_trials = n_trials
        self.metric = metric

def _fmt_val(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


class TuneResults:
    def __init__(self, results: List[JobResult], db_path: Path):
        self.results = results
        self.db_path = db_path

    def get_best_config(self, model: str, task: str) -> Optional[Dict]:
        return get_best_config(self.db_path, model, task)

    def _build_raw_dataframe(self) -> pd.DataFrame:
        rows = []
        studies_dir = self.db_path.parent / "studies" if self.db_path else None
        for r in self.results:
            study_path = studies_dir / f"{r.model}_{r.task}.db" if studies_dir else None
            if not study_path or not study_path.exists():
                continue
            try:
                study = optuna.load_study(study_name=f"{r.model}_{r.task}", storage=f"sqlite:///{study_path}")
            except Exception:
                continue
            for t in study.trials:
                if t.state != optuna.trial.TrialState.COMPLETE:
                    continue
                rows.append({
                    "model": r.model, "task": r.task, "trial": t.number + 1,
                    "trial_id": t.number, "primary_metric": r.metric,
                    "primary_value": t.value, "time_spent": t.user_attrs.get("duration_human", ""),
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["model", "task", "trial", "trial_id", "primary_metric", "primary_value", "time_spent"]
        )

    def _build_aggregated_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            a = self._analyze_study(r.model, r.task)
            best_result = a.get("best_result", {})
            row = {"model": r.model, "task": r.task}
            for k, v in r.best_params.items():
                row[f"param.{k}"] = U.fmt_float(v) if isinstance(v, (int, float)) else v
            display_base = a.get("display_base") or a.get("primary_base") or ""
            if display_base:
                for k, v in best_result.items():
                    if isinstance(v, (int, float)) and k.endswith(display_base):
                        row[k] = U.fmt_float(v)
            if self.db_path and Path(self.db_path).exists():
                try:
                    with _get_db_connection(self.db_path) as conn:
                        job_row = conn.execute(
                            "SELECT completed_trials, n_trials, elapsed_human FROM tuning_jobs WHERE model_name=? AND task_name=?",
                            (r.model, r.task)
                        ).fetchone()
                        if job_row:
                            row["completed"] = job_row["completed_trials"]
                            row["n_trials"] = job_row["n_trials"]
                            row["elapsed"] = job_row["elapsed_human"] or ""
                    timeout = _get_config(self.db_path, "timeout")
                    if timeout:
                        row["timeout"] = timeout
                except Exception:
                    pass
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def to_csv(self, path: Path, aggregated: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._build_aggregated_dataframe() if aggregated else self._build_raw_dataframe()
        df.to_csv(path, index=False)

    @staticmethod
    def _base_metric(metric: str) -> str:
        for prefix in ('val.', 'test.'):
            if metric.startswith(prefix):
                return metric[len(prefix):]
        return metric

    def _analyze_study(self, model_name: str, task_name: str) -> Dict:
        info = {
            'primary_metric': None, 'display_metric': None,
            'primary_base': None, 'display_base': None,
            'best_result': {}, 'stats': {}
        }
        if not self.db_path or not Path(self.db_path).exists():
            return info
        study_path = None
        try:
            with _get_db_connection(self.db_path) as conn:
                row = conn.execute(
                    "SELECT metric, study_path FROM tuning_jobs WHERE model_name=? AND task_name=?",
                    (model_name, task_name)
                ).fetchone()
                if row:
                    info['primary_metric'] = row['metric']
                    info['primary_base'] = self._base_metric(row['metric'])
                    study_path = self.db_path.parent / row['study_path'] if row['study_path'] else None
                config_row = conn.execute(
                    "SELECT config_json FROM tuning_job_configs WHERE model_name=? AND task_name=?",
                    (model_name, task_name)
                ).fetchone()
                if config_row:
                    config = json.loads(config_row['config_json'])
                    dm = config.get('eval', {}).get('display_metric')
                    if dm:
                        first_dm = dm[0] if isinstance(dm, list) else dm
                        info['display_metric'] = first_dm
                        info['display_base'] = self._base_metric(first_dm)
        except Exception:
            return info
        if not study_path or not study_path.exists():
            return info
        try:
            study = optuna.load_study(study_name=f"{model_name}_{task_name}", storage=f"sqlite:///{study_path}")
            try:
                info['best_result'] = study.best_trial.user_attrs.get('result', {})
            except ValueError:
                pass
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if trials and info['primary_base']:
                info['stats']['primary'] = self._calc_metric_stats(trials, info['primary_base'])
            if trials and info.get('display_base') and info['display_base'] != info.get('primary_base'):
                info['stats']['display'] = self._calc_metric_stats(trials, info['display_base'])
        except Exception:
            pass
        return info

    @staticmethod
    def _calc_metric_stats(trials, metric_base: str) -> Dict:
        val_key, test_key = f"val.{metric_base}", f"test.{metric_base}"
        all_val, all_test, paired_val, paired_test = [], [], [], []
        for t in trials:
            result = t.user_attrs.get('result', {})
            v, te = result.get(val_key), result.get(test_key)
            if isinstance(v, (int, float)):
                all_val.append(v)
            if isinstance(te, (int, float)):
                all_test.append(te)
            if isinstance(v, (int, float)) and isinstance(te, (int, float)):
                paired_val.append(v)
                paired_test.append(te)
        s = {}
        if all_val:
            s.update(val_mean=np.mean(all_val), val_std=np.std(all_val),
                     val_min=min(all_val), val_max=max(all_val), val_n=len(all_val))
        if all_test:
            s.update(test_mean=np.mean(all_test), test_std=np.std(all_test),
                     test_min=min(all_test), test_max=max(all_test), test_n=len(all_test))
        if len(paired_val) >= 2:
            s['correlation'] = float(np.corrcoef(paired_val, paired_test)[0, 1])
        return s

    @staticmethod
    def _format_stats_row(model: str, task: str, metric: str, s: Dict) -> str:
        n = s.get('val_n', s.get('test_n', '-'))
        val_ms = f"{s['val_mean']:.4f}±{s['val_std']:.4f}" if 'val_mean' in s else "-"
        val_rng = f"[{s['val_min']:.4f}, {s['val_max']:.4f}]" if 'val_min' in s else "-"
        test_ms = f"{s['test_mean']:.4f}±{s['test_std']:.4f}" if 'test_mean' in s else "-"
        test_rng = f"[{s['test_min']:.4f}, {s['test_max']:.4f}]" if 'test_min' in s else "-"
        corr = f"{s['correlation']:.4f}" if s.get('correlation') is not None else "-"
        return f"| {model} | {task} | {metric} | {n} | {val_ms} | {val_rng} | {test_ms} | {test_rng} | {corr} |"

    def to_markdown(self, path: Path = None) -> str:
        lines = ["# Tuning Results\n"]
        if not self.results:
            content = "\n".join(lines)
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(content)
            return content

        analyses = {}
        for r in self.results:
            analyses[(r.model, r.task)] = self._analyze_study(r.model, r.task)

        show_display = any(
            a.get('display_base') and a['display_base'] != a.get('primary_base')
            for a in analyses.values()
        )

        lines.append("## Best Results by Model/Task\n")
        if show_display:
            lines.append("| Model | Task | Metric | Best | Display | val | test | Trials |")
            lines.append("|-------|------|--------|------|---------|-----|------|--------|")
        else:
            lines.append("| Model | Task | Metric | Best Value | Trials |")
            lines.append("|-------|------|--------|------------|--------|")

        for r in self.results:
            a = analyses[(r.model, r.task)]
            if show_display:
                db = a.get('display_base', '')
                pb = a.get('primary_base', '')
                if db and db != pb:
                    br = a.get('best_result', {})
                    val_disp = _fmt_val(br.get(f"val.{db}"))
                    test_disp = _fmt_val(br.get(f"test.{db}"))
                    lines.append(
                        f"| {r.model} | {r.task} | {r.metric} | {r.best_value:.4f} "
                        f"| {db} | {val_disp} | {test_disp} | {r.n_trials} |"
                    )
                else:
                    lines.append(
                        f"| {r.model} | {r.task} | {r.metric} | {r.best_value:.4f} "
                        f"| - | - | - | {r.n_trials} |"
                    )
            else:
                lines.append(
                    f"| {r.model} | {r.task} | {r.metric} | {r.best_value:.4f} | {r.n_trials} |"
                )

        stats_rows = [(r, analyses[(r.model, r.task)]) for r in self.results
                      if analyses[(r.model, r.task)]['stats']]
        if stats_rows:
            lines.append("\n## Trial Statistics\n")
            lines.append("| Model | Task | Metric | n | val (mean±std) | val range | test (mean±std) | test range | r(val,test) |")
            lines.append("|-------|------|--------|---|----------------|-----------|-----------------|------------|-------------|")
            for r, a in stats_rows:
                ps = a['stats'].get('primary')
                if ps:
                    lines.append(self._format_stats_row(r.model, r.task, a['primary_base'], ps))
                ds = a['stats'].get('display')
                if ds:
                    lines.append(self._format_stats_row("", "", a['display_base'], ds))

        if self.results[0].best_params:
            lines.append("\n## Best Hyperparameters\n")
            for key in self.results[0].best_params:
                lines.append(f"\n### {key}")
                lines.append("| Model | Task | Value |")
                lines.append("|-------|------|-------|")
                for r in self.results:
                    val = r.best_params.get(key, "N/A")
                    lines.append(f"| {r.model} | {r.task} | {val} |")

        try:
            timing_rows = self._build_timing_table()
            if timing_rows:
                lines.append("\n## Job Timing\n")
                lines.append("| Model | Task | Start | End | Elapsed |")
                lines.append("|-------|------|-------|-----|---------|")
                for tr in timing_rows:
                    lines.append(f"| {tr['model']} | {tr['task']} | {tr['start']} | {tr['end']} | {tr['elapsed']} |")
        except Exception:
            pass

        content = "\n".join(lines)
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return content

    def _build_timing_table(self) -> list:
        rows = []
        if not self.db_path or not Path(self.db_path).exists():
            return rows
        with _get_db_connection(self.db_path) as conn:
            results = conn.execute("""
                SELECT model_name, task_name, start_time, end_time, elapsed_human
                FROM tuning_jobs WHERE status = 'completed'
                ORDER BY model_name, task_name
            """).fetchall()
            for r in results:
                start = U.Timer.format_time(r['start_time']) if r['start_time'] else "-"
                end = U.Timer.format_time(r['end_time']) if r['end_time'] else "-"
                elapsed = r['elapsed_human'] or "-"
                rows.append({"model": r['model_name'], "task": r['task_name'], "start": start, "end": end, "elapsed": elapsed})
        return rows

    def save_results(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.to_csv(output_dir / "raw_results.csv", aggregated=False)
        self.to_csv(output_dir / "aggregated_results.csv", aggregated=True)
        self.to_markdown(output_dir / "results.md")

    def print_summary_table(self, writer: ProgressWriter = None):
        if not self.db_path or not self.db_path.exists():
            return
        try:
            with _get_db_connection(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT model_name, task_name, best_value, trials_stats
                    FROM tuning_jobs
                    WHERE status = 'completed' AND best_value IS NOT NULL
                    ORDER BY model_name, task_name
                """).fetchall()
                if not rows:
                    return
                model_width = max(len('model_name'), max(len(r['model_name']) for r in rows))
                task_width = max(len('task_name'), max(len(r['task_name']) for r in rows))
                value_width = max(len('best_value'), 10)
                stats_width = max(len('trials_stats'), max(len(r['trials_stats'] or '') for r in rows))
                header = f"  {'model_name':<{model_width}} | {'task_name':<{task_width}} | {('best_value'):>{value_width}} | {('trials_stats'):<{stats_width}}"
                separator = f"  {'-' * model_width}-+-{'-' * task_width}-+-{'-' * value_width}-+-{'-' * stats_width}"
                output = ["\n" + separator, header, separator]
                for r in rows:
                    stats_str = r['trials_stats'] or 'N/A'
                    row = f"  {r['model_name']:<{model_width}} | {r['task_name']:<{task_width}} | {r['best_value']:>{value_width}.4f} | {stats_str:<{stats_width}}"
                    output.append(row)
                output.append(separator)
                if writer:
                    for line in output:
                        writer.write_event(line)
                else:
                    print("\n".join(output))
        except Exception:
            pass


# ─── Tuner ──────────────────────────────────────────────────────────────────

class Tuner:
    """Unified tuning engine using per-job Optuna studies."""

    def __init__(self, config: TuneConfig):
        self.config = config
        self.db_path = config.output / "tuning.db"
        self.run_json_path = config.output / "run.json"
        self.writer = ProgressWriter(silent=not config.show_progress)
        self.trial_output_base = Path(tempfile.gettempdir()) / "gsql_tune"
        self.trial_output_base.mkdir(parents=True, exist_ok=True)
        self.jobs = self._create_jobs()
        self.tracker = None

    def _create_jobs(self) -> List[TuneJob]:
        return [TuneJob(model, task, self.config) for model in self.config.models for task in self.config.tasks]

    def _init_database(self):
        self.config.output.mkdir(parents=True, exist_ok=True)
        _init_tune_db(self.db_path)

    def _handle_fresh_mode(self, active_keys: set):
        if self._get_resume_mode() != C.ResumeMode.fresh:
            return
        studies_folder = self.config.output / "studies"
        with _get_db_connection(self.db_path) as conn:
            for key in active_keys:
                model_name, task_name = key.split("_", 1)
                conn.execute(
                    "UPDATE tuning_jobs SET status = 'pending', start_time = NULL, end_time = NULL, elapsed_human = NULL, completed_trials = 0 WHERE model_name = ? AND task_name = ?",
                    (model_name, task_name)
                )
                study_path = studies_folder / f"{model_name}_{task_name}.db"
                if study_path.exists():
                    study_path.unlink()
        self.writer.write_event(f"🔄 Fresh mode: Reset {len(active_keys)} active job(s)")

    def _setup_pending_jobs(self):
        _init_tune_db(self.db_path)
        with _get_db_connection(self.db_path) as conn:
            for job in self.jobs:
                study_name = f"{job.model.name}_{job.task.name}"
                study_path = f"studies/{study_name}.db"
                job_n_trials = job.model.n_trials if job.model.n_trials is not None else self.config.n_trials
                job_metric, job_direction = _resolve_job_metric(job.task, self.config.metric, self.config.direction)
                conn.execute("""INSERT OR IGNORE INTO tuning_jobs
                    (model_name, task_name, study_path, n_trials, metric, direction, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
                    (job.model.name, job.task.name, study_path, job_n_trials, job_metric, job_direction))
        _save_config(self.db_path, 'n_trials', self.config.n_trials)
        _save_config(self.db_path, 'timeout', self.config.timeout)
        _save_config(self.db_path, 'metric', self.config.metric)
        _save_config(self.db_path, 'direction', self.config.direction)
        _save_config(self.db_path, 'n_workers', self.config.n_workers)
        _save_config(self.db_path, 'device_ids', self.config.device_ids)
        _save_config(self.db_path, 'sampler', self.config.sampler)
        _save_config(self.db_path, 'global_config', self.config.global_config)
        _save_config(self.db_path, 'resume_mode', self.config.resume_mode)

    def _get_resume_mode(self) -> C.ResumeMode:
        return C.resolve_resume_mode(self.config.resume_mode)

    def _check_studies_consistency(self):
        studies_folder = self.config.output / "studies"
        if self._get_resume_mode() == C.ResumeMode.resume:
            if not self.db_path.exists() and studies_folder.exists():
                self.writer.write_event("⚠️  Database missing but studies folder exists - resetting studies")
                shutil.rmtree(studies_folder)
            elif self.db_path.exists() and not studies_folder.exists():
                self.writer.write_event("ℹ️  Studies folder missing, will be recreated during tuning")

    def _create_objective(self, job: TuneJob):
        job_metric, _ = _resolve_job_metric(job.task, self.config.metric, self.config.direction)

        def objective(trial: optuna.Trial) -> float:
            sampled = sample_config(trial, job.search_space, [])
            params = U.unnest_dict(sampled) if isinstance(sampled, dict) else {}
            trial_output_dir = self.trial_output_base / job.model.name / job.task.name / f"trial_{trial.number}"
            train_config = self._build_config(job, params, trial_output_dir)
            debug_cmd = C.generate_debug_cmd(job.model.function, train_config)
            trial.set_user_attr('debug_cmd', debug_cmd)
            trial.set_user_attr('metric_name', job_metric)
            try:
                trial_start = time.time()
                train_fn = U.import_function(job.model.function)
                result = train_fn(train_config)
                value = self._extract_metric(result, job_metric)
                trial.set_user_attr('duration_human', _fmt_duration(time.time() - trial_start))
                if isinstance(result, dict):
                    trial.set_user_attr('result', {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))})
                U.cleanup_resources()
                return value
            except Exception as e:
                with _get_db_connection(self.db_path) as conn:
                    conn.execute(
                        "UPDATE tuning_jobs SET latest_debug_cmd=? WHERE model_name=? AND task_name=?",
                        (debug_cmd, job.model.name, job.task.name)
                    )
                U.cleanup_resources()
                raise
        return objective

    def _build_config(self, job: TuneJob, params: Dict, trial_output_dir: Path = None) -> Dict:
        model_config = job.base_config.copy()
        model_config.update(params)
        model_config = U.flat_dict(model_config)
        hierarchical = C.merge_hierarchical_config(model_config, job.task.config, self.config.global_config)
        hierarchical["data"] = C.merge_hierarchical_config(job.model.data_config, job.task.data, hierarchical.get("data", {}))
        if trial_output_dir:
            if "output" not in hierarchical:
                hierarchical["output"] = {}
            hierarchical["output"]["folder"] = str(trial_output_dir)
        job_metric, job_direction = _resolve_job_metric(job.task, self.config.metric, self.config.direction)
        runtime_overrides = {
            "log": {"mode": "logger", "config": {"log_level": "CRITICAL"}},
            "output": {k: False for k in ["save_model", "save_pred", "save_analysis", "_save_config", "save_final", "save_best"]},
            "eval": {"primary_metric": job_metric, "direction": job_direction}
        }
        U.merge_dict(hierarchical, runtime_overrides)
        return hierarchical

    def _extract_metric(self, result: Any, metric: str) -> float:
        if isinstance(result, dict):
            if metric in result:
                return result[metric]
            for key in [f"test.{metric}", f"val.{metric}"]:
                if key in result:
                    value = result[key]
                    return value if not isinstance(value, dict) else value.get(metric, value)
        return result if isinstance(result, (int, float)) else 0.0

    def run_single_job(self, job: TuneJob) -> JobResult:
        job_name = f"{job.model.name} on {job.task.name}"
        study_name = f"{job.model.name}_{job.task.name}"
        with _get_db_connection(self.db_path) as conn:
            result = conn.execute(
                "SELECT study_path FROM tuning_jobs WHERE model_name=? AND task_name=?",
                (job.model.name, job.task.name)
            ).fetchone()
            study_path_rel = result[0] if result else None
        if study_path_rel:
            storage_path = self.config.output / study_path_rel
        else:
            storage_path = self.config.output / "studies" / f"{study_name}.db"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        sampler_kwargs = self._create_sampler_kwargs()
        if storage_path.exists():
            try:
                study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path}", sampler=sampler_kwargs["sampler"])
            except KeyError:
                study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{storage_path}", direction=sampler_kwargs["direction"], sampler=sampler_kwargs["sampler"])
        else:
            study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{storage_path}", direction=sampler_kwargs["direction"], sampler=sampler_kwargs["sampler"])
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if (remaining_trials := job.n_trials - completed_trials) <= 0:
            self.writer.write_event(f"⏩ {job_name}: All {job.n_trials} trials complete")
            return self._load_job_result_from_study(study, job)
        base_config = self._build_config(job, {})
        _save_job_config(self.db_path, job.model.name, job.task.name, base_config)
        _start_job(self.db_path, job.model.name, job.task.name)
        objective = self._create_objective(job)
        callbacks = []
        if self.tracker is not None:
            callback = ProgressCallback(self.tracker, f"{job.model.name} on {job.task.name}", job.n_trials)
            callbacks.append(callback)
        gsql = GsqlTrack(f"tune/{self.config.output.name}")
        gsql_run = gsql.start_run(f"{job.model.name}/{job.task.name}", source="tune")
        gsql_run.log_params({"model": job.model.name, "task": job.task.name, "n_trials": job.n_trials})
        try:
            study.optimize(
                objective, n_trials=remaining_trials, timeout=job.timeout,
                show_progress_bar=False, gc_after_trial=True, callbacks=callbacks,
            )
            try:
                best_trial = study.best_trial
                debug_cmd = best_trial.user_attrs.get('debug_cmd')
                _update_best_config(self.db_path, job.model.name, job.task.name, best_trial.number, best_trial.value, best_trial.params, debug_cmd)
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed_trials:
                    values = [t.value for t in completed_trials]
                    stats_str = f"{np.mean(values):.4f}±{np.std(values):.4f}"
                    _update_job_stats(self.db_path, job.model.name, job.task.name, stats_str)
                _save_best_config_from_study(storage_path, self.config.output, job.model.name, job.task.name)
                job_metric, _ = _resolve_job_metric(job.task, self.config.metric, self.config.direction)
                _save_best_result_from_study(storage_path, self.config.output, job.model.name, job.task.name, metric_name=job_metric)
            except ValueError:
                pass
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            _finish_job(self.db_path, job.model.name, job.task.name, completed_trials=n_completed)
            try:
                gsql_run.log(0, best_value=study.best_trial.value, n_completed=n_completed)
            except ValueError:
                gsql_run.log(0, n_completed=n_completed)
            gsql_run.finish()
            gsql.close()
            return self._load_job_result_from_study(study, job)
        except Exception as e:
            _fail_job(self.db_path, job.model.name, job.task.name, str(e))
            gsql_run.fail()
            gsql.close()
            raise

    def _is_job_complete(self, job: TuneJob) -> bool:
        with _get_db_connection(self.db_path) as conn:
            row = conn.execute("SELECT status FROM tuning_jobs WHERE model_name=? AND task_name=?",
                             (job.model.name, job.task.name)).fetchone()
            return row and row['status'] == 'completed'

    def _load_job_result(self, job: TuneJob) -> JobResult:
        job_metric, _ = _resolve_job_metric(job.task, self.config.metric, self.config.direction)
        with _get_db_connection(self.db_path) as conn:
            row = conn.execute("""
                SELECT best_value, best_params, completed_trials, metric FROM tuning_jobs
                WHERE model_name=? AND task_name=?
            """, (job.model.name, job.task.name)).fetchone()
            if row and row['best_value'] is not None:
                return JobResult(
                    model=job.model.name, task=job.task.name,
                    best_value=row['best_value'],
                    best_params=json.loads(row['best_params']) if row['best_params'] else {},
                    n_trials=row['completed_trials'] or 0,
                    metric=row['metric'] or job_metric,
                )
        return JobResult(job.model.name, job.task.name, 0.0, {}, 0, metric=job_metric)

    def _load_job_result_from_study(self, study: optuna.Study, job: TuneJob) -> JobResult:
        job_metric, _ = _resolve_job_metric(job.task, self.config.metric, self.config.direction)
        try:
            best_trial = study.best_trial
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            return JobResult(model=job.model.name, task=job.task.name, best_value=best_trial.value, best_params=best_trial.params, n_trials=n_completed, metric=job_metric)
        except ValueError:
            return JobResult(job.model.name, job.task.name, 0.0, {}, 0, metric=job_metric)

    def _create_sampler_kwargs(self) -> Dict:
        sampler_config = self.config.sampler or {}
        sampler_type = sampler_config.get('sampler', 'tpe')
        direction = self.config.direction
        sampler_map = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "grid": optuna.samplers.GridSampler,
            "cmaes": optuna.samplers.CmaEsSampler,
        }
        sampler_cls = sampler_map.get(sampler_type.lower())
        sampler_kwargs = sampler_config.get('sampler_kws', {})
        sampler_kwargs_filtered = {k: v for k, v in sampler_kwargs.items() if k != 'seed'}
        return {
            "direction": direction,
            "sampler": sampler_cls(**sampler_kwargs_filtered) if sampler_cls else None,
        }

    def _confirm_jobs(self) -> Optional[List[TuneJob]]:
        job_pairs = [(j.model.name, j.task.name) for j in self.jobs]
        job_info = {}
        for j in self.jobs:
            key = PM.make_key(j.model.name, j.task.name)
            n_trials = j.model.n_trials or self.config.n_trials
            job_info[key] = f"{n_trials} trials"
        plan = PM.build_run_plan(
            job_pairs, self.run_json_path,
            n_workers=self.config.n_workers, device_ids=self.config.device_ids,
            resume_mode=self.config.resume_mode.value if hasattr(self.config.resume_mode, 'value') else str(self.config.resume_mode),
            manager_type="tune", job_info=job_info,
        )
        result = PM.confirm_and_save(plan, self.run_json_path, self.writer, auto_confirm=self.config.auto_confirm)
        if result is None:
            return None
        self.config.n_workers = result.n_workers
        self.config.device_ids = result.device_ids
        self.config.resume_mode = result.resume_mode
        if result.n_trials is not None:
            self.config.n_trials = result.n_trials
            for j in self.jobs:
                j.n_trials = result.n_trials
        active_keys = PM.get_active_job_keys(result)
        return [j for j in self.jobs if PM.make_key(j.model.name, j.task.name) in active_keys]

    def _load_cached_results_for_skipped(self, skipped_keys: set) -> List[JobResult]:
        cached = []
        for result in self._load_results_from_db():
            key = PM.make_key(result.model, result.task)
            if key in skipped_keys:
                cached.append(result)
        return cached

    def run_sequential(self) -> TuneResults:
        self._check_studies_consistency()
        self._setup_pending_jobs()
        active_jobs = self._confirm_jobs()
        if active_jobs is None:
            return TuneResults([], self.db_path)
        all_keys = {PM.make_key(j.model.name, j.task.name) for j in self.jobs}
        active_keys = {PM.make_key(j.model.name, j.task.name) for j in active_jobs}
        skipped_keys = all_keys - active_keys
        self._handle_fresh_mode(active_keys)
        start_time = time.time()
        n_jobs = len(active_jobs)
        self.writer.write_event(f"Starting sequential tuning: {n_jobs} jobs")
        self.tracker = TuneProgressTracker(n_jobs, self.writer, self.db_path, self.config.n_trials)
        results = []
        with U.no_tqdm_pbar(disable=True):
            for job in active_jobs:
                task_desc = f"{job.model.name} on {job.task.name}"
                self.tracker.update(task_desc, 0, self.config.n_trials, current_best=None)
                try:
                    result = self.run_single_job(job)
                    results.append(result)
                    study_name = f"{job.model.name}_{job.task.name}"
                    study_path = self.config.output / "studies" / f"{study_name}.db"
                    stats = self.tracker._get_job_trial_stats(study_path, task_desc)
                    self.tracker.complete_job(task_desc, result.best_value, result.n_trials, stats=stats)
                except Exception as e:
                    self.tracker._fail_job(task_desc, str(e))
                    raise
        results.extend(self._load_cached_results_for_skipped(skipped_keys))
        elapsed_time = time.time() - start_time
        DB.save_run_epilogue(self.db_path, 'tuning_state', elapsed_time)
        return TuneResults(results, self.db_path)

    def run_parallel(self) -> TuneResults:
        self._check_studies_consistency()
        self._setup_pending_jobs()
        active_jobs = self._confirm_jobs()
        if active_jobs is None:
            return TuneResults([], self.db_path)
        active_keys = {PM.make_key(j.model.name, j.task.name) for j in active_jobs}
        self._handle_fresh_mode(active_keys)
        start_time = time.time()
        mode, n_workers, device_assignment = DM.determine_execution_mode(self.config.device_ids, self.config.n_workers)
        self.tracker = TuneProgressTracker(len(active_jobs), self.writer, self.db_path, self.config.n_trials)
        config_dict = {
            'models': [sp.to_dict(m) for m in self.config.models],
            'tasks': [sp.to_dict(t) for t in self.config.tasks],
            'output': str(self.config.output),
            'n_trials': self.config.n_trials,
            'timeout': self.config.timeout,
            'metric': self.config.metric,
            'direction': self.config.direction,
            'sampler': self.config.sampler,
            'global_config': self.config.global_config,
            'resume_mode': self.config.resume_mode,
        }
        worker_configs = [
            {'rank': rank, 'db_path': str(self.db_path), 'config_dict': config_dict,
             'device_id': device_assignment[rank], 'mode': mode}
            for rank in range(n_workers)
        ]
        self.config.output.mkdir(parents=True, exist_ok=True)
        procs, worker_log_files = DM.launch_workers(
            n_workers, device_assignment, mode,
            "gsql_track.tune", worker_configs,
            log_dir=self.config.output,
        )
        self._monitor_workers(procs, active_keys)
        for log_path, log_file in worker_log_files:
            log_file.close()
        DM.merge_worker_logs([p for p, _ in worker_log_files], self.config.output / "run.log")
        elapsed_time = time.time() - start_time
        DB.save_run_epilogue(self.db_path, 'tuning_state', elapsed_time)
        results = self._load_results_from_db()
        return TuneResults(results, self.db_path)

    def _monitor_workers(self, procs, active_keys: set = None):
        print("\nMonitor progress: tail -f /tmp/tune_worker_*.txt\n")
        output_queue = DM.collect_worker_outputs(procs)
        last_completed, last_trial_info = 0, {}
        if active_keys:
            pairs = [k.split("_", 1) for k in active_keys]
            pair_filter = " OR ".join(["(model_name = ? AND task_name = ?)"] * len(pairs))
            pair_params = [v for p in pairs for v in p]
            where_active = f" AND ({pair_filter})"
        else:
            where_active = ""
            pair_params = []
        while any(p.poll() is None for p in procs):
            with _get_db_connection(self.db_path) as conn:
                completed = conn.execute(f"SELECT COUNT(*) FROM tuning_jobs WHERE status = 'completed'{where_active}", pair_params).fetchone()[0]
                running_row = conn.execute(
                    f"SELECT model_name, task_name, n_trials FROM tuning_jobs WHERE status = 'running'{where_active}", pair_params
                ).fetchone()
            current_task, best_value = "", None
            if running_row:
                model_name, task_name = running_row['model_name'], running_row['task_name']
                job_n_trials = running_row['n_trials']
                current_task = f"{model_name} on {task_name}"
                study_name = f"{model_name}_{task_name}"
                if (study_path := self.config.output / "studies" / f"{study_name}.db").exists():
                    try:
                        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_path}")
                        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                        try:
                            best_value = study.best_value
                        except ValueError:
                            best_value = None
                        trial_info_key = f"{model_name}_{task_name}"
                        current_trial_info = (completed_trials, best_value)
                        if current_trial_info != last_trial_info.get(trial_info_key):
                            last_trial_info[trial_info_key] = current_trial_info
                            elapsed = time.time() - self.tracker.start_time
                            task_part = f"⏳ {current_task} (trial {completed_trials}/{job_n_trials})"
                            if best_value is not None:
                                task_part += f" best={best_value:.4f}"
                            eta = self.tracker.calculate_eta(elapsed) if completed > 0 else ""
                            msg = f"\r[✓ {completed}/{self.tracker.total_jobs} | {task_part}] {self.tracker.fmt_time(elapsed)}{eta}"
                            self.writer.write_progress(msg, overwrite=True)
                    except Exception:
                        pass
            if completed > last_completed:
                self.writer.write_event(f"✓ Completed {completed}/{self.tracker.total_jobs} jobs")
                last_completed = completed
            DM.drain_worker_outputs(output_queue, self.writer)
            time.sleep(1)
        DM.drain_worker_outputs(output_queue, self.writer)

    def _load_results_from_db(self) -> List[JobResult]:
        results = []
        with _get_db_connection(self.db_path) as conn:
            rows = conn.execute("""
                SELECT model_name, task_name, best_value, best_params, completed_trials, metric
                FROM tuning_jobs
                WHERE status = 'completed' AND best_value IS NOT NULL
            """).fetchall()
            for row in rows:
                results.append(JobResult(
                    model=row['model_name'], task=row['task_name'],
                    best_value=row['best_value'],
                    best_params=json.loads(row['best_params']) if row['best_params'] else {},
                    n_trials=row['completed_trials'] or 0,
                    metric=row['metric'] or "",
                ))
        return results

    def run(self) -> TuneResults:
        return self.run_parallel() if self.config.n_workers > 1 else self.run_sequential()

    def save_results(self, results: TuneResults):
        output_dir = Path(self.config.output)
        results.save_results(output_dir)
        results.print_summary_table(self.writer)
        self.writer.write_event(f"\n✓ Results saved to {output_dir}")


# ─── Subprocess worker ──────────────────────────────────────────────────────

def _subprocess_worker_main():
    cfg = DM.parse_worker_args()
    _subprocess_worker(**cfg)


def _subprocess_worker(
    rank: int, db_path: str, config_dict: Dict, device_id: Optional[int], mode: str,
    jobs: List[tuple] = None,
):
    DM.setup_worker_process(mode, device_id, rank)
    if mode == "gpu" and torch is not None and torch.cuda.is_available():
        with open(f"/tmp/tune_worker_{rank}.txt", 'w') as f:
            f.write(f"Worker {rank}: Using GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})\n")
    config = TuneConfig(**config_dict)
    config.show_progress = False
    config.auto_confirm = True
    config.n_workers = 1
    runner = Tuner(config)

    def run_one(job):
        model_name, task_name = job
        j = next(j for j in runner.jobs if j.model.name == model_name and j.task.name == task_name)
        runner.run_single_job(j)

    with U.no_tqdm_pbar(disable=True):
        DM.run_worker_loop_queue(
            rank=rank, db_path=db_path,
            claim_fn=DB.tune_claim_next_job,
            run_one_fn=run_one,
            progress_prefix="tune_worker",
            fmt_job=lambda j: f"{j[0]}/{j[1]}",
        )


# ─── Sample config from hyperparameter space ───────────────────────────────

def sample_config(
    trial: optuna.trial.Trial,
    space: Union[bool, int, float, str, bytes, list, dict],
    label_parts: list,
) -> Any:
    """Sample configuration from hyperparameter space.

    Supported distributions:
        - ['range', low, high, step?]: Continuous uniform with optional step
        - ['choice', opt1, opt2, ...]: Categorical selection
        - ['int', low, high]: Integer uniform
        - ['int', 'log', low, high]: Log-scale integer
        - ['?dist', default, ...args]: Optional parameter with default
    """
    match space:
        case None:
            return None
        case bool() | int() | float() | str() | bytes():
            return space
        case dict() | DictConfig():
            return {k: sample_config(trial, v, label_parts + [k]) for k, v in space.items()}
        case []:
            return space
        case [str(dist), *args] if dist.startswith("?"):
            if not args:
                raise ValueError("Optional distribution requires default value")
            label = ".".join(map(str, label_parts))
            return (
                sample_config(trial, [dist[1:]] + args[1:], label_parts)
                if trial.suggest_categorical(f"?{label}", [False, True])
                else args[0]
            )
        case [("range" | "float" | "choice" | "int" | "log") as dist, *args]:
            label = ".".join(map(str, label_parts))
            match dist, args:
                case "log", [low, high]:
                    return trial.suggest_float(label, low, high, log=True)
                case "float", [low, high]:
                    return trial.suggest_float(label, low, high)
                case "range", [low, high]:
                    return trial.suggest_float(label, low, high)
                case "range", ["log", low, high]:
                    return trial.suggest_float(label, low, high, log=True)
                case "range", [low, high, step]:
                    return trial.suggest_float(label, low, high, step=step)
                case "range", _:
                    raise ValueError(f"range requires 2-3 args, got {len(args)}")
                case "choice", [*choices] if choices:
                    return trial.suggest_categorical(label, choices)
                case "choice", _:
                    raise ValueError("choice requires at least 1 option")
                case "int", [low, high]:
                    return trial.suggest_int(label, low, high)
                case "int", ["log", low, high]:
                    return trial.suggest_int(label, low, high, log=True)
                case "int", [low, high, step]:
                    return trial.suggest_int(label, low, high, step=step)
                case "int", _:
                    raise ValueError(f"int requires 2-3 args, got {len(args)}")
        case list():
            return [sample_config(trial, item, label_parts + [str(i)]) for i, item in enumerate(space)]
        case _:
            raise TypeError(f"Unsupported space type: {type(space)}")


def extract_tune_space(config: Dict[str, Any]) -> SpaceDict:
    """Extract key:value pairs with full paths where the value is a list starting with '_tune_'."""
    def set_nested_dict(d: Dict[str, Any], path: list[str], value: Any):
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    def extract_from_dict(d: Dict[str, Any], path: list[str] = []) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in d.items():
            current_path = path + [key]
            if isinstance(value, list) and value and re.match(r"^\??(range|choice|int|log)$", value[0]):
                set_nested_dict(result, current_path, value)
            elif isinstance(value, dict):
                nested = extract_from_dict(value, current_path)
                for k, v in nested.items():
                    result[k] = v
        return result
    return extract_from_dict(config)


def extract_tune_key(d: dict, parent_key: str = '') -> list[str]:
    result = []
    for key, value in d.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        match value:
            case list() if value and re.match(r"^\??(range|choice|int|log)$", str(value[0])):
                result.append(current_key)
            case dict():
                result.extend(extract_tune_key(value, current_key))
    return result


def sample_random_params(spaces: Dict[str, Any], seed: int) -> Dict[str, Any]:
    temp_storage = "sqlite:///:memory:"
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(study_name=f"baseline_{seed}", storage=temp_storage, sampler=sampler, direction="maximize")
    trial = study.ask()
    config = sample_config(trial, spaces, [])
    return config


# ─── Main entry point ───────────────────────────────────────────────────────

def run_tuning(config: TuneConfig) -> TuneResults:
    runner = Tuner(config)
    results = runner.run()
    runner.save_results(results)
    return results


# Backwards compatibility alias
tune_hyperparameters = run_tuning


__all__ = [
    "run_tuning",
    "tune_hyperparameters",
    "TuneConfig",
    "TaskConfig",
    "ModelTuneSpec",
    "Tuner",
    "TuneResults",
    "JobResult",
    "TuneJob",
    "SlackCallback",
    "ProgressCallback",
    "TuneProgressTracker",
    "extract_tune_space",
    "sample_random_params",
    "sample_config",
    "get_best_config",
    "get_failed_jobs",
]
