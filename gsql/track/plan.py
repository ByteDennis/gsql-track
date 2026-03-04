"""Job plan management with run.json persistence and interactive confirmation.

Provides shared utilities for bench_manager and tune_manager to:
- Persist per-job skip preferences in run.json
- Display job plan tables before execution
- Interactively confirm/modify job plans (skip/unskip, devices, workers)

Example
-------
>>> plan = build_run_plan(job_pairs, run_json_path, n_workers=2, device_ids=[0,1], resume_mode="resume", manager_type="tune")
>>> plan = confirm_and_save(plan, run_json_path, writer)
>>> active_keys = get_active_job_keys(plan)
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import util as U

ProgressWriter = U.ProgressWriter


@dataclass
class JobPlanEntry:
    """Single job in a run plan."""
    key: str           # e.g. "BERT_AGNews"
    model: str         # e.g. "BERT"
    task: str          # e.g. "AGNews"
    skip: bool = False
    skip_reason: str = ""  # e.g. "run.json"
    info: str = ""     # e.g. "3 seeds" or "50 trials"


@dataclass
class RunPlan:
    """Complete run plan with job list and execution settings."""
    jobs: List[JobPlanEntry] = field(default_factory=list)
    n_workers: int = 1
    device_ids: Optional[List[int]] = None
    resume_mode: str = "resume"
    manager_type: str = "bench"  # "bench" or "tune"
    n_seeds: Optional[int] = None   # Override for bench: seeds per model×task
    n_trials: Optional[int] = None  # Override for tune: trials per model×task


def make_key(model: str, task: str) -> str:
    """Create a job key from model and task names."""
    return f"{model}_{task}"


#>>> run.json I/O <<<#

def load_run_json(path: Path) -> dict:
    """Load run.json from disk. Returns empty dict if not found."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_run_json(path: Path, plan: RunPlan):
    """Save run.json from a RunPlan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "jobs": {
            entry.key: {"skip": entry.skip}
            for entry in plan.jobs
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


#>>> Plan building <<<#

def build_run_plan(
    job_pairs: List[Tuple[str, str]],
    run_json_path: Path,
    n_workers: int = 1,
    device_ids: Optional[List[int]] = None,
    resume_mode: str = "resume",
    manager_type: str = "bench",
    job_info: Optional[Dict[str, str]] = None,
) -> RunPlan:
    """Build a RunPlan from model/task pairs, merging with existing run.json.

    New jobs get skip=False. Existing jobs keep their skip preference.
    Stale entries (not in job_pairs) are removed.

    Args:
        job_info: Optional dict mapping job key to info string (e.g. {"BERT_AGNews": "3 seeds"})
    """
    existing = load_run_json(run_json_path)
    existing_jobs = existing.get("jobs", {})
    job_info = job_info or {}

    entries = []
    for model, task in job_pairs:
        key = make_key(model, task)
        if key in existing_jobs:
            skip = existing_jobs[key].get("skip", False)
            reason = "run.json" if skip else ""
        else:
            skip = False
            reason = ""
        entries.append(JobPlanEntry(
            key=key, model=model, task=task,
            skip=skip, skip_reason=reason,
            info=job_info.get(key, ""),
        ))

    return RunPlan(
        jobs=entries,
        n_workers=n_workers,
        device_ids=device_ids,
        resume_mode=resume_mode,
        manager_type=manager_type,
    )


#>>> Display <<<#

def display_plan(plan: RunPlan, writer: ProgressWriter):
    """Print the job plan table."""
    header = f"\n{'=' * 60}\nJob Plan ({plan.manager_type})"
    devices_str = str(plan.device_ids) if plan.device_ids else "cpu"
    header += f"\n  Resume: {plan.resume_mode} | Workers: {plan.n_workers} | Devices: {devices_str}"
    header += f"\n{'=' * 60}\n"
    writer.write_event(header)

    # Column widths
    num_w = max(3, len(str(len(plan.jobs))))
    model_w = max(5, max((len(e.model) for e in plan.jobs), default=5))
    task_w = max(4, max((len(e.task) for e in plan.jobs), default=4))
    has_info = any(e.info for e in plan.jobs)
    info_w = max(4, max((len(e.info) for e in plan.jobs), default=4)) if has_info else 0

    if has_info:
        hdr = f"  {'#':>{num_w}}  {'Model':<{model_w}}  {'Task':<{task_w}}  {'Info':<{info_w}}  Status"
    else:
        hdr = f"  {'#':>{num_w}}  {'Model':<{model_w}}  {'Task':<{task_w}}  Status"
    sep = f"  {'-' * len(hdr)}"
    writer.write_event(hdr)
    writer.write_event(sep)

    for i, entry in enumerate(plan.jobs, 1):
        if entry.skip:
            status = f"skip ({entry.skip_reason})" if entry.skip_reason else "skip"
        else:
            status = "run"
        if has_info:
            writer.write_event(f"  {i:>{num_w}}  {entry.model:<{model_w}}  {entry.task:<{task_w}}  {entry.info:<{info_w}}  {status}")
        else:
            writer.write_event(f"  {i:>{num_w}}  {entry.model:<{model_w}}  {entry.task:<{task_w}}  {status}")

    n_run = sum(1 for e in plan.jobs if not e.skip)
    n_skip = sum(1 for e in plan.jobs if e.skip)
    writer.write_event(f"\n  {n_run} to run, {n_skip} to skip\n")


#>>> Interactive confirmation <<<#

def _parse_indices(text: str, max_idx: int) -> List[int]:
    """Parse comma-separated 1-based indices into 0-based list."""
    indices = []
    for part in text.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < max_idx:
                indices.append(idx)
    return indices


def interactive_confirm(plan: RunPlan, writer: ProgressWriter) -> Optional[RunPlan]:
    """Interactive confirmation loop. Returns modified plan or None if quit."""
    while True:
        try:
            response = input("Proceed? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            writer.write_event("\nAborted.")
            return None

        if response in ("", "y", "yes"):
            return plan
        elif response in ("n", "no"):
            # Enter command loop
            # Build help text based on manager type
            type_cmd = "seeds 5" if plan.manager_type == "bench" else "trials 100"
            help_text = f"\nCommands: skip 1,3 | unskip 2 | devices 0,2 | workers 3 | {type_cmd} | resume fresh | done | quit"
            writer.write_event(help_text)
            while True:
                try:
                    cmd = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    writer.write_event("\nAborted.")
                    return None

                if not cmd:
                    continue

                # Support multiple commands separated by |
                subcmds = [s.strip() for s in cmd.split("|")]
                redisplay = False
                early_exit = None

                for subcmd in subcmds:
                    if not subcmd:
                        continue
                    parts = subcmd.split(maxsplit=1)
                    action = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""

                    if action == "quit":
                        early_exit = None  # signal quit
                        break
                    elif action == "done":
                        early_exit = plan
                        break
                    elif action == "skip" and args:
                        for idx in _parse_indices(args, len(plan.jobs)):
                            plan.jobs[idx].skip = True
                            plan.jobs[idx].skip_reason = "user"
                        redisplay = True
                    elif action == "unskip" and args:
                        for idx in _parse_indices(args, len(plan.jobs)):
                            plan.jobs[idx].skip = False
                            plan.jobs[idx].skip_reason = ""
                        redisplay = True
                    elif action == "devices" and args:
                        try:
                            plan.device_ids = [int(x.strip()) for x in args.split(",")]
                            plan.n_workers = len(plan.device_ids)
                            writer.write_event(f"  Devices: {plan.device_ids}, Workers: {plan.n_workers}")
                        except ValueError:
                            writer.write_event("  Invalid device IDs")
                    elif action == "workers" and args:
                        try:
                            plan.n_workers = int(args.strip())
                            writer.write_event(f"  Workers: {plan.n_workers}")
                        except ValueError:
                            writer.write_event("  Invalid worker count")
                    elif action == "seeds" and args and plan.manager_type == "bench":
                        try:
                            n = int(args.strip())
                            if n < 1:
                                raise ValueError
                            plan.n_seeds = n
                            for entry in plan.jobs:
                                entry.info = f"{n} seeds"
                            writer.write_event(f"  Seeds: {n}")
                            redisplay = True
                        except ValueError:
                            writer.write_event("  Invalid seed count")
                    elif action == "trials" and args and plan.manager_type == "tune":
                        try:
                            n = int(args.strip())
                            if n < 1:
                                raise ValueError
                            plan.n_trials = n
                            for entry in plan.jobs:
                                entry.info = f"{n} trials"
                            writer.write_event(f"  Trials: {n}")
                            redisplay = True
                        except ValueError:
                            writer.write_event("  Invalid trial count")
                    elif action == "resume" and args:
                        mode = args.strip().lower()
                        if mode in ("resume", "fresh", "append"):
                            plan.resume_mode = mode
                            redisplay = True
                        else:
                            writer.write_event(f"  Invalid resume mode: {mode}. Use: resume, fresh, append")
                    else:
                        type_cmd = "seeds" if plan.manager_type == "bench" else "trials"
                        writer.write_event(f"  Unknown command: {action}. Use: skip/unskip/devices/workers/{type_cmd}/resume/done/quit")

                if action in ("quit", "done"):
                    return early_exit
                if redisplay:
                    display_plan(plan, writer)
                    type_cmd = "seeds 5" if plan.manager_type == "bench" else "trials 100"
                    writer.write_event(f"Commands: skip 1,3 | unskip 2 | devices 0,2 | workers 3 | {type_cmd} | resume fresh | done | quit")
        else:
            writer.write_event("  Please enter Y or n")


def confirm_and_save(
    plan: RunPlan,
    run_json_path: Path,
    writer: ProgressWriter,
    auto_confirm: bool = False,
) -> Optional[RunPlan]:
    """Top-level: display plan, optionally confirm, save run.json.

    Args:
        plan: The run plan to confirm
        run_json_path: Path to save run.json
        writer: Progress writer for output
        auto_confirm: If True, auto-confirm without interactive prompt (CI mode)

    Returns:
        Modified plan, or None if user quit
    """
    display_plan(plan, writer)

    if not auto_confirm and sys.stdin.isatty():
        result = interactive_confirm(plan, writer)
        if result is None:
            return None
        plan = result

    save_run_json(run_json_path, plan)
    writer.write_event("Saved run.json\n")
    return plan


#>>> Plan query helpers <<<#

def get_active_job_keys(plan: RunPlan) -> Set[str]:
    """Get keys of jobs that will run (not skipped)."""
    return {e.key for e in plan.jobs if not e.skip}


def get_skipped_job_keys(plan: RunPlan) -> Set[str]:
    """Get keys of jobs that are skipped."""
    return {e.key for e in plan.jobs if e.skip}


__all__ = [
    "JobPlanEntry",
    "RunPlan",
    "make_key",
    "load_run_json",
    "save_run_json",
    "build_run_plan",
    "display_plan",
    "interactive_confirm",
    "confirm_and_save",
    "get_active_job_keys",
    "get_skipped_job_keys",
]
