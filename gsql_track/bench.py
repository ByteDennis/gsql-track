"""Benchmark execution engine with SQLite persistence and parallel workers.

Example
-------
>>> runner = BenchmarkRunner(config)
>>> results = runner.run_parallel()
"""
import copy
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import springs as sp

from . import config as C
from . import db as DB
from . import dispatch as DM
from . import plan as PM
from . import util as U
from .gsql_track import GsqlTrack

ResumeMode = C.ResumeMode
TaskConfig = C.TaskConfig
ProgressWriter = U.ProgressWriter
BaseProgressTracker = U.BaseProgressTracker
_get_db_connection = DB.get_db_connection
_fmt_duration = U.fmt_duration

try:
    import torch
except ImportError:
    torch = None


# ─── DB functions (delegated to db) ─────────────────────────────────────────

_init_benchmark_db = DB.bench_init_db
_log_progress = DB.bench_log_progress
_start_run = DB.bench_start_run
_finish_run = DB.bench_finish_run
_load_results = DB.bench_load_results
_get_pending_runs = DB.bench_get_pending_runs


# ─── State management ───────────────────────────────────────────────────────

def _update_state(db_path: Path, key: str, value: Any):
    DB.update_state(db_path, "benchmark_state", key, value)

def _get_state(db_path: Path, key: str, default=None):
    return DB.get_state(db_path, "benchmark_state", key, default)


# ─── Config loading ─────────────────────────────────────────────────────────

def load_best_config(
    path_pattern: str,
    model_name: str = None,
    task_name: str = None,
    ignore_keys: List[str] = ['output', 'log', 'pbar', 'seed'],
    base_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Load best hyperparameters from a YAML config file.

    Path pattern supports {model_name} and {task_name} placeholders.

    Example
    -------
    >>> config = load_best_config('results/tune/best_configs/{model_name}_{task_name}_best.yaml', 'BERT', 'AGNews')
    """
    import yaml
    path_str = str(path_pattern).format(model_name=model_name or '', task_name=task_name or '')
    config_path = Path(path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Best config not found: {config_path}")

    with open(config_path) as f:
        tuned_config = yaml.safe_load(f)

    keys_to_ignore = ignore_keys if ignore_keys is not None else ['output', 'log', 'pbar', 'seed']
    filtered_config = {k: v for k, v in tuned_config.items() if k not in keys_to_ignore}
    if base_config is not None:
        result = copy.deepcopy(base_config)
        U.merge_dict(result, filtered_config)
        return result
    return filtered_config


# ─── Progress tracking ──────────────────────────────────────────────────────

class BenchmarkProgressTracker(BaseProgressTracker):
    """Real-time benchmark progress tracker.

    Example
    -------
    >>> tracker = BenchmarkProgressTracker(total=100)
    >>> tracker.update("completed", success=True, current_task="MV on yelp")
    """

    def update(self, status: str, success: bool = True, current_task: str = ""):
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            self._display_progress(current_task)

    def _display_progress(self, current_task: str = ""):
        elapsed = time.time() - self.start_time
        eta = f"| ETA: {self.fmt_time(((self.total - self.completed - self.failed) / ((self.completed + self.failed) / elapsed)) if self.completed + self.failed > 0 else 0)}" if self.completed + self.failed > 0 else ""
        status_str = f"⏳ {current_task}" if current_task else ""
        progress_msg = f"\r[✓ {self.completed}/{self.total} | ✗ {self.failed}"
        if status_str:
            progress_msg += f" | {status_str}"
        static_part = progress_msg + "]"
        if static_part != self.last_static_part:
            self.last_static_part = static_part
            self.writer.write_progress(progress_msg + f"] {self.fmt_time(elapsed)}{eta}" + " " * 10, overwrite=True)

    def finalize(self):
        elapsed = time.time() - self.start_time
        self.writer.write_event(f"\n✓ Benchmark Complete: {self.completed} succeeded, {self.failed} failed out of {self.total} total runs in {self.fmt_time(elapsed)}")
        return elapsed


# ─── Benchmark configuration ────────────────────────────────────────────────

@sp.dataclass
class ModelSpec:
    """Model specification for benchmarking.

    Example
    -------
    >>> spec = ModelSpec(name='mv', function='src.algs.majority_vote.train', n_seeds=3)
    """
    name: str = sp.field(help="Model identifier")
    function: str = sp.field(help="Training function path")
    train_config: Dict[str, Any] = sp.field(default_factory=dict, help="Training config overrides")
    n_seeds: int = sp.field(default=3, help="Number of seeds")
    data_config: Optional[Dict[str, Any]] = sp.field(default=None, help="Data config overrides")
    best_config_path: Optional[str] = sp.field(default=None, help="Path pattern to best config YAML file (supports {model_name}/{task_name} placeholders)")


@sp.dataclass(kw_only=True)
class BenchmarkConfig(C.StartConfig):
    """Main benchmark configuration.

    Example
    -------
    >>> config = BenchmarkConfig(models=[spec], tasks=[task], output='./bench', n_workers=4)
    """
    models: List[ModelSpec] = sp.field(help="Models to benchmark")
    tasks: List[TaskConfig] = sp.field(help="Tasks to evaluate")
    output: Path = sp.field(help="Output directory")
    n_workers: int = sp.field(default=1, help="Parallel workers")
    device_ids: Optional[List[int]] = sp.field(default=None, help="GPU device IDs for parallel execution")
    global_config: Optional[Dict[str, Any]] = sp.field(default=None, help="Global config overrides")
    pre_import_modules: Optional[List[str]] = sp.field(default=None, help="Modules to import at startup")
    show_progress: bool = sp.field(default=True, help="Show progress messages")
    auto_confirm: bool = sp.field(default=False, help="Auto-confirm jobs without interactive prompt")

    def __post_init__(self):
        self.output = Path(self.output)
        self.models = [ModelSpec(**m) if isinstance(m, dict) else m for m in (self.models or [])]
        self.tasks = [TaskConfig(**t) if isinstance(t, dict) else t for t in (self.tasks or [])]


# ─── Result containers ──────────────────────────────────────────────────────

class BenchmarkResult:
    """Container for single benchmark result."""
    def __init__(self, model: str, task: str, seed: int, best_metrics: Dict[str, float],
                 time_spent: float = None):
        self.model, self.task, self.seed = model, task, seed
        self.best_metrics = best_metrics
        self.time_spent = time_spent
        self.metrics = best_metrics

    def to_dict(self):
        result = {
            'model': self.model,
            'task': self.task,
            'seed': self.seed,
            **self.best_metrics
        }
        if self.time_spent is not None:
            result['time_spent'] = self.time_spent
        return result


class BenchmarkResults:
    """Aggregated benchmark results with export utilities."""

    def __init__(self, results: List[BenchmarkResult], db_path: Path = None,
                 display_metrics: List[str] = None, display_metric_map: Dict = None):
        self.results, self.db_path = results, db_path
        self.display_metric_map = display_metric_map or {}
        display_metrics = display_metrics or []
        if not results:
            self.df = pd.DataFrame(columns=['model', 'task', 'seed'])
            self.display_metrics = set()
            return
        df = pd.DataFrame([r.to_dict() for r in results])
        if display_metrics:
            metric_cols = [c for c in df.columns if any(c.endswith(dm) for dm in display_metrics) or c in ['model', 'task', 'seed', 'time_spent']]
            self.df = df[metric_cols]
        else:
            self.df = df
            metric_cols = df.columns.tolist()
        self.display_metrics = {c for c in metric_cols if c not in ['model', 'task', 'seed', 'time_spent']}

    def __repr__(self):
        agg = self.aggregate()
        for dm in self.display_metrics:
            mean_col = f'{dm}_mean'
            std_col = f'{dm}_std'
            if mean_col in agg.columns and std_col in agg.columns:
                agg[dm] = agg.apply(lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}", axis=1)
                agg = agg.drop(columns=[mean_col, std_col])
        formatters = {dm: lambda x: f"{x:>20}" for dm in self.display_metrics if dm in agg.columns}
        return agg.to_string(index=False, formatters=formatters)

    def aggregate(self, metrics: List[str] = None, add_time: bool = False) -> pd.DataFrame:
        metrics = metrics or [c for c in self.df.columns if c not in ['model', 'task', 'seed', 'time_spent']]
        if not metrics:
            return self.df.groupby(['model', 'task']).first().reset_index()
        grouped = self.df.groupby(['model', 'task'])[metrics].agg(['mean', 'std'])
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped[[c for c in grouped.columns if c.endswith("_std")]] = grouped[[c for c in grouped.columns if c.endswith("_std")]].fillna(0)
        result = grouped.reset_index()
        if add_time and self.db_path:
            result['elapsed_time'] = _get_state(self.db_path, 'elapsed_time', default=0)
        return result

    def pivot_table(self, metric: str = 'acc') -> pd.DataFrame:
        agg = self.aggregate([metric])
        agg['formatted'] = agg.apply(lambda r: f"{r[f'{metric}_mean']:.2f} ± {r[f'{metric}_std']:.2f}", axis=1)
        return agg.pivot(index='model', columns='task', values='formatted')

    def add_mean_column(self, metric: str = 'acc') -> pd.DataFrame:
        pivot, agg = self.pivot_table(metric), self.aggregate([metric])
        mean_by_model = agg.groupby('model')[f'{metric}_mean'].mean()
        pivot['Mean'] = mean_by_model.apply(lambda x: f"{x:.2f}")
        return pivot

    def to_latex(self, metric: str = 'acc', caption: str = "", label: str = "", include_mean: bool = True) -> str:
        pivot = self.add_mean_column(metric) if include_mean else self.pivot_table(metric)
        return pivot.to_latex(caption=caption or f"Benchmark Results ({metric.upper()})",
                             label=label or f"tab:benchmark_{metric}", escape=False, float_format="%.2f")

    def to_markdown(self, metric: str = 'acc', include_mean: bool = True) -> str:
        return (self.add_mean_column(metric) if include_mean else self.pivot_table(metric)).to_markdown()

    def to_csv(self, path: Path, aggregated: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if aggregated:
            self._build_aggregated_csv(path)
        else:
            self._build_raw_csv(path)

    def _build_raw_csv(self, path: Path):
        rows = []
        for r in self.results:
            dms = self.display_metric_map.get((r.model, r.task), [])
            dm = dms[0] if dms else ""
            display_value = r.best_metrics.get(dm) if dm else None
            rows.append({
                "model": r.model, "task": r.task, "seed": r.seed,
                "display_metric": dm, "display_value": display_value,
                "time_spent": r.time_spent,
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def _build_aggregated_csv(self, path: Path):
        rows = []
        for (model, task), group_df in self.df.groupby(['model', 'task']):
            row = {"model": model, "task": task}
            row["n_seeds"] = len(group_df)
            for col in self.display_metrics:
                if col in group_df.columns:
                    values = group_df[col].dropna()
                    if len(values) > 0:
                        row[f"{col}_mean"] = U.fmt_float(values.mean())
                        row[f"{col}_std"] = U.fmt_float(values.std() if len(values) > 1 else 0.0)
            if 'time_spent' in group_df.columns:
                total_time = group_df['time_spent'].dropna().sum()
                row["elapsed"] = _fmt_duration(total_time) if total_time > 0 else ""
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)


# ─── Parallel worker subprocess ─────────────────────────────────────────────

def _subprocess_worker_main():
    """Entry point for subprocess worker."""
    cfg = DM.parse_worker_args()
    _subprocess_worker(**cfg)


def _subprocess_worker(
    rank: int, db_path: str, config_dict: Dict, device_id: Optional[int], mode: str,
    worker_id: int = 0, jobs: List[tuple] = None,
):
    """Worker process with GPU isolation (queue mode)."""
    logging.disable(logging.ERROR)
    DM.setup_worker_process(mode, device_id, rank)
    if mode == "gpu":
        with open(f"/tmp/bench_worker_{worker_id}.txt", 'w') as f:
            f.write(f"Worker {rank}: Using GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})\n")
    config = BenchmarkConfig(**config_dict)
    config.show_progress = False
    runner = BenchmarkRunner(config)

    def run_one(job):
        model_name, task_name, seed = job
        model_spec = next(m for m in config.models if m.name == model_name)
        task_config = next(t for t in config.tasks if t.name == task_name)
        runner.run_single(model_spec, task_config, seed)

    with U.no_tqdm_pbar(disable=True):
        DM.run_worker_loop_queue(
            rank=worker_id, db_path=db_path,
            claim_fn=DB.bench_claim_next_job,
            run_one_fn=run_one,
            progress_prefix="bench_worker",
            fmt_job=lambda j: f"{j[0]}/{j[1]}/seed_{j[2]}",
        )


# ─── Benchmark runner ───────────────────────────────────────────────────────

class BenchmarkRunner:
    """Benchmark execution engine with SQLite persistence."""

    def __init__(self, config: BenchmarkConfig):
        self.config, self.db_path = config, config.output / "benchmark.db"
        self.run_json_path = config.output / "run.json"
        self.writer = ProgressWriter(silent=not config.show_progress)
        config.output.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if config.pre_import_modules:
            U.pre_import_modules(config.pre_import_modules, silent=not config.show_progress)

    def _validate_config(self, train_config: Dict, model_name: str, task_name: str):
        if not train_config.get('eval', {}).get('primary_metric'):
            raise ValueError(f"Model '{model_name}' + Task '{task_name}': eval.primary_metric not found in final config")

    def _build_train_config(self, model_spec: ModelSpec, task_config: TaskConfig, seed: int) -> Dict:
        output_path = self.config.output / model_spec.name / task_config.name / f"seed_{seed}"
        if model_spec.best_config_path:
            try:
                model_config = load_best_config(
                    model_spec.best_config_path, model_spec.name, task_config.name,
                    ignore_keys=['output', 'log', 'pbar', 'seed'],
                    base_config=model_spec.train_config
                )
            except FileNotFoundError:
                model_config = model_spec.train_config
        else:
            model_config = model_spec.train_config
        base_config = {
            'seed': seed, 'output': {'folder': str(output_path), 'save_model': False, 'save_pred': False, 'save_config': False},
            'log': {'mode': 'gsql', 'config': {'experiment': f"bench/{self.config.output.name}", 'run_name': f"{model_spec.name}/{task_config.name}/seed_{seed}"}}, 'pbar': {'enabled': False}
        }
        hierarchical = C.merge_hierarchical_config(model_config, task_config.config, self.config.global_config)
        hierarchical["data"] = C.merge_hierarchical_config(model_spec.data_config, task_config.data, hierarchical.get("data", {}))
        default_eval = hierarchical.get('eval', {})
        resolved_eval = C.resolve_task_eval(
            task_config,
            default_metric=default_eval.get('primary_metric', 'val.acc'),
            default_direction=default_eval.get('direction', 'maximize'),
            default_display_metric=default_eval.get('display_metric'),
        )
        hierarchical.setdefault('eval', {}).update(resolved_eval)
        U.merge_dict(base_config, hierarchical)
        base_config['seed'] = seed
        if 'output' in base_config and 'folder' in base_config['output']:
            original_folder = Path(base_config['output']['folder'])
            base_config['output']['folder'] = str(original_folder / f"seed_{seed}")
        return base_config

    def _collect_display_metric_map(self) -> Dict[Tuple[str, str], list]:
        global_eval = (self.config.global_config or {}).get('eval', {})
        result = {}
        for model_spec in self.config.models:
            model_eval = model_spec.train_config.get('eval', {}) if model_spec.train_config else {}
            merged_eval = {**global_eval, **model_eval}
            default_metric = merged_eval.get('primary_metric', 'acc')
            default_direction = merged_eval.get('direction', 'maximize')
            default_display = merged_eval.get('display_metric')
            for task_config in self.config.tasks:
                resolved = C.resolve_task_eval(
                    task_config, default_metric, default_direction, default_display
                )
                result[(model_spec.name, task_config.name)] = resolved['display_metric']
        return result

    def _is_run_complete(self, model_spec: ModelSpec, task_config: TaskConfig, seed: int) -> bool:
        return (self.config.output / model_spec.name / task_config.name / f"seed_{seed}" / "DONE").exists()

    def _extract_metrics(self, result_metrics: Dict[str, Any], requested_metrics: List[str]) -> Dict[str, float]:
        extracted = {}
        for m in requested_metrics:
            for key in [m, f'test.{m}', f'val.{m}']:
                if key in result_metrics:
                    value = result_metrics[key]
                    extracted.update(U.unnest_dict({m: value}) if isinstance(value, dict) else {m: value})
                    break
            else:
                extracted[m] = np.nan
        return extracted

    def run_single(self, model_spec: ModelSpec, task_config: TaskConfig, seed: int) -> BenchmarkResult:
        if self._is_run_complete(model_spec, task_config, seed):
            if (resume_mode := self._get_resume_mode()) == ResumeMode.resume:
                self.writer.write_event(f"⏩ Skipping {model_spec.name} on {task_config.name} (seed={seed}) - already done")
                with _get_db_connection(self.db_path) as conn:
                    row = conn.execute("""SELECT res.best_metrics, res.time_spent
                                        FROM benchmark_runs r JOIN benchmark_results res ON r.id = res.run_id
                                        WHERE r.model_name = ? AND r.task_name = ? AND r.seed = ?""",
                                      (model_spec.name, task_config.name, seed)).fetchone()
                    if row:
                        return BenchmarkResult(model_spec.name, task_config.name, seed,
                                              json.loads(row['best_metrics']),
                                              row['time_spent'])
            elif resume_mode == ResumeMode.fresh:
                self.writer.write_event(f"🔄 Fresh mode: re-running {model_spec.name} on {task_config.name} (seed={seed})")
        train_config_dict = self._build_train_config(model_spec, task_config, seed)
        exe_cmd = C.generate_debug_cmd(model_spec.function, train_config_dict) if model_spec.function else None
        run_id = _start_run(self.db_path, model_spec.name, task_config.name, seed, exe_cmd)
        gsql = GsqlTrack(f"bench/{self.config.output.name}", db_path=str(self.config.output / "track.db"))
        gsql_run = gsql.start_run(f"{model_spec.name}/{task_config.name}/seed_{seed}", source="bench")
        curated = {"model": model_spec.name, "task": task_config.name, "seed": seed}
        init_args = (model_spec.train_config or {}).get("model", {}).get("init_args", {})
        for k in ("lr", "batch_size", "n_steps", "max_tokens", "fine_tune_layers"):
            if k in init_args:
                curated[k] = init_args[k]
        eval_cfg = train_config_dict.get("eval", {})
        if eval_cfg.get("primary_metric"):
            curated["primary_metric"] = eval_cfg["primary_metric"]
        gsql_run.log_params(curated)
        try:
            self._validate_config(train_config_dict, model_spec.name, task_config.name)
            _log_progress(self.db_path, run_id, "Config validated")
            train_fn = U.import_function(model_spec.function)
            _log_progress(self.db_path, run_id, "Starting training")
            result_metrics = train_fn(train_config_dict)
            best_metrics = result_metrics if isinstance(result_metrics, dict) else {}
            _log_progress(self.db_path, run_id, message="Finished training")
            U.cleanup_resources()
            _finish_run(self.db_path, run_id, result_metrics, train_config_dict)
            done_path = self.config.output / model_spec.name / task_config.name / f"seed_{seed}" / "DONE"
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.touch()
            numeric_metrics = {k: float(v) for k, v in best_metrics.items() if isinstance(v, (int, float)) and k != "step"}
            if numeric_metrics:
                gsql_run.log(0, **numeric_metrics)
            gsql_run.finish()
            gsql.close()
            return BenchmarkResult(model_spec.name, task_config.name, seed, best_metrics)
        except Exception as e:
            U.cleanup_resources()
            _finish_run(self.db_path, run_id, {}, error=str(e))
            gsql_run.fail()
            gsql.close()
            raise

    def _setup_pending_runs(self):
        _init_benchmark_db(self.db_path)
        with _get_db_connection(self.db_path) as conn:
            for model_spec in self.config.models:
                for task_config in self.config.tasks:
                    for seed in range(1, 1 + model_spec.n_seeds):
                        conn.execute("INSERT OR IGNORE INTO benchmark_runs (model_name, task_name, seed, status) VALUES (?, ?, ?, 'pending')",
                                   (model_spec.name, task_config.name, seed))

    def _generate_jobs(self):
        return [(m, t, s) for m in self.config.models for t in self.config.tasks for s in range(1, 1 + m.n_seeds)]

    def _handle_fresh_mode(self, active_keys: set):
        if self._get_resume_mode() != ResumeMode.fresh:
            return
        with _get_db_connection(self.db_path) as conn:
            for key in active_keys:
                model_name, task_name = key.split("_", 1)
                run_ids = conn.execute(
                    "SELECT id FROM benchmark_runs WHERE model_name = ? AND task_name = ?",
                    (model_name, task_name)
                ).fetchall()
                for row in run_ids:
                    conn.execute("DELETE FROM benchmark_results WHERE run_id = ?", (row['id'],))
                conn.execute(
                    "UPDATE benchmark_runs SET status = 'pending', start_time = NULL, end_time = NULL, error = NULL WHERE model_name = ? AND task_name = ?",
                    (model_name, task_name)
                )
                for model_spec in self.config.models:
                    if model_spec.name == model_name:
                        for task_config in self.config.tasks:
                            if task_config.name == task_name:
                                for seed in range(1, 1 + model_spec.n_seeds):
                                    done_path = self.config.output / model_name / task_name / f"seed_{seed}" / "DONE"
                                    if done_path.exists():
                                        done_path.unlink()
        self.writer.write_event(f"🔄 Fresh mode: Reset {len(active_keys)} active job(s)")

    def _get_resume_mode(self) -> ResumeMode:
        return C.resolve_resume_mode(self.config.resume_mode)

    def _reset_missing_done_files(self):
        with _get_db_connection(self.db_path) as conn:
            completed_rows = conn.execute("SELECT model_name, task_name, seed FROM benchmark_runs WHERE status = 'completed'").fetchall()
            for row in completed_rows:
                model_name, task_name, seed = row['model_name'], row['task_name'], row['seed']
                model_spec = next(m for m in self.config.models if m.name == model_name)
                task_config = next(t for t in self.config.tasks if t.name == task_name)
                if not self._is_run_complete(model_spec, task_config, seed):
                    conn.execute("UPDATE benchmark_runs SET status = 'pending' WHERE model_name = ? AND task_name = ? AND seed = ?",
                                (model_name, task_name, seed))

    def _confirm_jobs(self, jobs):
        seen = set()
        job_pairs = []
        for model_spec, task_config, seed in jobs:
            key = PM.make_key(model_spec.name, task_config.name)
            if key not in seen:
                seen.add(key)
                job_pairs.append((model_spec.name, task_config.name))
        job_info = {}
        for model_spec, task_config, seed in jobs:
            key = PM.make_key(model_spec.name, task_config.name)
            if key not in job_info:
                job_info[key] = f"{model_spec.n_seeds} seeds"
        plan = PM.build_run_plan(
            job_pairs,
            self.run_json_path,
            n_workers=self.config.n_workers,
            device_ids=self.config.device_ids,
            resume_mode=self.config.resume_mode.value if hasattr(self.config.resume_mode, 'value') else str(self.config.resume_mode),
            manager_type="bench",
            job_info=job_info,
        )
        result = PM.confirm_and_save(
            plan, self.run_json_path, self.writer,
            auto_confirm=self.config.auto_confirm,
        )
        if result is None:
            return None, set()
        self.config.n_workers = result.n_workers
        self.config.device_ids = result.device_ids
        self.config.resume_mode = result.resume_mode
        if result.n_seeds is not None:
            for model_spec in self.config.models:
                model_spec.n_seeds = result.n_seeds
            jobs = self._generate_jobs()
            self._setup_pending_runs()
        active_keys = PM.get_active_job_keys(result)
        skipped_keys = PM.get_skipped_job_keys(result)
        filtered = [(m, t, s) for m, t, s in jobs
                     if PM.make_key(m.name, t.name) in active_keys]
        return filtered, skipped_keys

    def _load_cached_results_for_skipped(self, skipped_keys: set) -> List[BenchmarkResult]:
        cached = []
        for r in _load_results(self.db_path):
            key = PM.make_key(r['model'], r['task'])
            if key in skipped_keys:
                cached.append(BenchmarkResult(
                    r['model'], r['task'], r['seed'],
                    r['best_metrics'], r['time_spent']
                ))
        return cached

    def _save_bench_configs(self, results: BenchmarkResults, output_dir: Path):
        import yaml
        bench_configs_dir = output_dir / "bench_configs"
        bench_configs_dir.mkdir(parents=True, exist_ok=True)
        for (model, task), group in results.df.groupby(['model', 'task']):
            with _get_db_connection(self.db_path) as conn:
                row = conn.execute("""
                    SELECT res.config FROM benchmark_runs r
                    JOIN benchmark_results res ON r.id = res.run_id
                    WHERE r.model_name = ? AND r.task_name = ? AND r.status = 'completed'
                    ORDER BY res.run_id LIMIT 1
                """, (model, task)).fetchone()
                if row and row['config']:
                    config = json.loads(row['config'])
                    for key in ['output', 'log', 'pbar', 'seed']:
                        config.pop(key, None)
                    config_path = bench_configs_dir / f"{model}_{task}_best.yaml"
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def run_sequential(self) -> BenchmarkResults:
        self._setup_pending_runs()
        jobs = self._generate_jobs()
        filtered, skipped_keys = self._confirm_jobs(jobs)
        if filtered is None:
            return BenchmarkResults([], db_path=self.db_path)
        jobs = filtered
        active_keys = {PM.make_key(m.name, t.name) for m, t, s in jobs}
        self._handle_fresh_mode(active_keys)
        n_seeds = self.config.models[0].n_seeds if self.config.models else 0
        self.writer.write_event(f"Starting sequential benchmark: {len(jobs)} runs")
        progress_file = "/tmp/bench_worker_0.txt"
        print(f"\nMonitor progress: tail -f {progress_file}\n")
        progress, results = BenchmarkProgressTracker(len(jobs), writer=self.writer), []
        with U.no_tqdm_pbar(disable=False):
            for i, (model_spec, task_config, seed) in enumerate(jobs):
                task_desc = f"{model_spec.name} on {task_config.name} (seed={seed})"
                with open(progress_file, 'w') as f:
                    f.write(f"[{i+1}/{len(jobs)}] {task_desc}\n")
                progress._display_progress(current_task=task_desc)
                try:
                    results.append(self.run_single(model_spec, task_config, seed))
                    progress.update("completed", success=True, current_task=task_desc)
                except Exception as e:
                    with open(progress_file, 'w') as f:
                        f.write(f"ERROR: {task_desc}: {e}\n")
                    progress.update("failed", success=False, current_task=task_desc)
                    self.writer.write_event(f"\n✗ {task_desc}: {e}")
                    raise
        with open(progress_file, 'w') as f:
            f.write(f"Done ({len(jobs)} jobs completed)\n")
        results.extend(self._load_cached_results_for_skipped(skipped_keys))
        elapsed_time = progress.finalize()
        DB.save_run_epilogue(self.db_path, 'benchmark_state', elapsed_time)
        dm_map = self._collect_display_metric_map()
        display_metrics = sorted({m for ms in dm_map.values() for m in ms})
        return BenchmarkResults(results, db_path=self.db_path, display_metrics=display_metrics, display_metric_map=dm_map)

    def run_parallel(self) -> BenchmarkResults:
        self._setup_pending_runs()
        jobs = self._generate_jobs()
        filtered, skipped_keys = self._confirm_jobs(jobs)
        if filtered is None:
            return BenchmarkResults([], db_path=self.db_path)
        jobs = filtered
        active_keys = {PM.make_key(m.name, t.name) for m, t, s in jobs}
        self._handle_fresh_mode(active_keys)
        mode, n_workers, device_assignment = DM.determine_execution_mode(self.config.device_ids, self.config.n_workers)
        self.writer.write_event(f"Starting {'GPU-isolated' if mode == 'gpu' else 'CPU parallel'} benchmark: {len(jobs)} runs on {n_workers} {'GPUs' if mode == 'gpu' else 'workers'}")
        config_dict = {
            'models': [sp.to_dict(m) for m in self.config.models],
            'tasks': [sp.to_dict(t) for t in self.config.tasks],
            'output': str(self.config.output),
            'n_workers': n_workers,
            'device_ids': self.config.device_ids,
            'global_config': self.config.global_config,
            'pre_import_modules': self.config.pre_import_modules,
            'resume_mode': self.config.resume_mode,
            'show_progress': False
        }
        worker_configs = [
            {'rank': rank, 'db_path': str(self.db_path), 'config_dict': config_dict,
             'device_id': device_assignment[rank], 'mode': mode, 'worker_id': rank}
            for rank in range(n_workers)
        ]
        procs, _ = DM.launch_workers(
            n_workers, device_assignment, mode,
            "gsql_track.bench", worker_configs, writer=self.writer,
        )
        print("\nMonitor progress: tail -f /tmp/bench_worker_*.txt\n")
        progress = BenchmarkProgressTracker(len(jobs), writer=self.writer)
        self._monitor_workers(procs, progress, active_keys)
        results = [BenchmarkResult(r['model'], r['task'], r['seed'], r['best_metrics'], r['time_spent'])
                  for r in _load_results(self.db_path)]
        elapsed_time = progress.finalize()
        DB.save_run_epilogue(self.db_path, 'benchmark_state', elapsed_time)
        dm_map = self._collect_display_metric_map()
        display_metrics = sorted({m for ms in dm_map.values() for m in ms})
        return BenchmarkResults(results, db_path=self.db_path, display_metrics=display_metrics, display_metric_map=dm_map)

    def _monitor_workers(self, procs, progress, active_keys: set = None):
        output_queue = DM.collect_worker_outputs(procs)
        last_completed, last_failed = 0, 0
        if active_keys:
            pairs = [k.split("_", 1) for k in active_keys]
            pair_filter = " OR ".join(["(model_name = ? AND task_name = ?)"] * len(pairs))
            pair_params = [v for p in pairs for v in p]
            where_active = f" AND ({pair_filter})"
        else:
            where_active = ""
            pair_params = []
        while any(p.poll() is None for p in procs):
            self._reset_missing_done_files()
            with _get_db_connection(self.db_path) as conn:
                completed = conn.execute(f"SELECT COUNT(*) FROM benchmark_runs WHERE status = 'completed'{where_active}", pair_params).fetchone()[0]
                failed = conn.execute(f"SELECT COUNT(*) FROM benchmark_runs WHERE status = 'failed'{where_active}", pair_params).fetchone()[0]
                pending = conn.execute(f"SELECT COUNT(*) FROM benchmark_runs WHERE status = 'pending'{where_active}", pair_params).fetchone()[0]
                running_row = conn.execute("""
                    SELECT r.model_name, r.task_name, r.seed, p.current_step, p.total_steps, p.message
                    FROM benchmark_runs r
                    LEFT JOIN (
                        SELECT run_id, current_step, total_steps, message,
                               ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY timestamp DESC) as rn
                        FROM benchmark_progress
                    ) p ON r.id = p.run_id AND p.rn = 1
                    WHERE r.status = 'running' LIMIT 1
                """).fetchone()
                current_task = ""
                if running_row:
                    task_name = f"{running_row['model_name']} on {running_row['task_name']} (seed={running_row['seed']})"
                    progress_info = running_row['message'] or ""
                    if running_row['current_step'] and running_row['total_steps']:
                        progress_info += f" {running_row['current_step']}/{running_row['total_steps']}"
                    if progress_info:
                        current_task = f"{task_name} [{progress_info}]"
                    else:
                        current_task = task_name
            status_changed = False
            if completed > last_completed:
                for _ in range(completed - last_completed):
                    progress.update("completed", success=True, current_task=current_task)
                last_completed = completed
                status_changed = True
            if failed > last_failed:
                for _ in range(failed - last_failed):
                    progress.update("failed", success=False, current_task=current_task)
                last_failed = failed
                status_changed = True
            if not status_changed:
                progress._display_progress(current_task=current_task)
            DM.drain_worker_outputs(output_queue, self.writer)
            time.sleep(0.1 if pending == 0 else 1)
        DM.drain_worker_outputs(output_queue, self.writer)
        with _get_db_connection(self.db_path) as conn:
            completed = conn.execute(f"SELECT COUNT(*) FROM benchmark_runs WHERE status = 'completed'{where_active}", pair_params).fetchone()[0]
            failed = conn.execute(f"SELECT COUNT(*) FROM benchmark_runs WHERE status = 'failed'{where_active}", pair_params).fetchone()[0]
        if completed > last_completed:
            for _ in range(completed - last_completed):
                progress.update("completed", success=True)
        if failed > last_failed:
            for _ in range(failed - last_failed):
                progress.update("failed", success=False)

    def run(self) -> BenchmarkResults:
        return self.run_parallel() if self.config.n_workers > 1 else self.run_sequential()

    def _save_best_results(self, results: BenchmarkResults, output_dir: Path):
        best_results_dir = output_dir / "best_results"
        best_results_dir.mkdir(parents=True, exist_ok=True)
        agg = results.aggregate()
        for _, row in agg.iterrows():
            model, task = row['model'], row['task']
            result_data = {
                'model': model, 'task': task,
                **{k: v for k, v in row.items() if k not in ('model', 'task')},
            }
            with _get_db_connection(self.db_path) as conn:
                db_row = conn.execute(
                    "SELECT debug FROM benchmark_runs WHERE model_name = ? AND task_name = ? LIMIT 1",
                    (model, task)
                ).fetchone()
                if db_row and db_row['debug']:
                    result_data['debug_cmd'] = db_row['debug']
            (best_results_dir / f"{model}_{task}_best.json").write_text(
                json.dumps(result_data, cls=U.JSONEncoder, indent=2)
            )

    def save_results(self, results: BenchmarkResults):
        if not results.results:
            self.writer.write_event("No results to save.")
            return
        output_dir = Path(self.config.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / 'raw_results.csv', aggregated=False)
        results.to_csv(output_dir / 'aggregated_results.csv', aggregated=True)
        self._save_best_results(results, output_dir)
        self._save_bench_configs(results, output_dir)
        markdown_content = []
        markdown_content.append("# Benchmark Results\n")
        for metric in results.display_metrics:
            metric_table = results.to_markdown(metric, include_mean=True)
            markdown_content.append(f"\n## {metric.upper()}\n\n")
            markdown_content.append(metric_table)
            U.print_title(f"Results for {metric.upper()}")
            self.writer.write_event(metric_table)
        try:
            timing_rows = self._build_timing_table()
            if timing_rows:
                markdown_content.append("\n## Job Timing\n")
                markdown_content.append("| Model | Task | Start | End | Elapsed |")
                markdown_content.append("|-------|------|-------|-----|---------|")
                for tr in timing_rows:
                    markdown_content.append(f"| {tr['model']} | {tr['task']} | {tr['start']} | {tr['end']} | {tr['elapsed']} |")
        except Exception:
            pass
        combined_markdown = "\n".join(markdown_content)
        (output_dir / 'results.md').write_text(combined_markdown)
        self.writer.write_event(f"\n✓ Results saved to {output_dir}")

    def _build_timing_table(self) -> List[Dict]:
        rows = []
        with _get_db_connection(self.db_path) as conn:
            results = conn.execute("""
                SELECT model_name, task_name, MIN(start_time) as start_time, MAX(end_time) as end_time
                FROM benchmark_runs WHERE status = 'completed'
                GROUP BY model_name, task_name ORDER BY model_name, task_name
            """).fetchall()
            for r in results:
                start = U.Timer.format_time(r['start_time']) if r['start_time'] else "-"
                end = U.Timer.format_time(r['end_time']) if r['end_time'] else "-"
                elapsed = _fmt_duration(r['end_time'] - r['start_time']) if r['start_time'] and r['end_time'] else "-"
                rows.append({"model": r['model_name'], "task": r['task_name'], "start": start, "end": end, "elapsed": elapsed})
        return rows


# ─── Main entry point ───────────────────────────────────────────────────────

def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    runner = BenchmarkRunner(config)
    import sys
    gsql = GsqlTrack(f"bench/{config.output.name}", db_path=str(config.output / "track.db"))
    gsql.log_experiment_metadata({
        "command": " ".join(sys.argv),
        "models": [m.name for m in config.models],
        "tasks": [t.name for t in config.tasks],
        "n_workers": config.n_workers,
    })
    gsql.close()
    results = runner.run()
    runner.save_results(results)
    return results


__all__ = [
    'BenchmarkConfig',
    'TaskConfig',
    'ModelSpec',
    'BenchmarkRunner',
    'BenchmarkProgressTracker',
    'BenchmarkResults',
    'BenchmarkResult',
    'run_benchmark',
    'load_best_config',
]
