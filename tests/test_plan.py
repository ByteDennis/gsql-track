"""Tests for gsql_track.plan — RunPlan, run.json I/O, interactive confirm."""
import json
from unittest.mock import patch

from gsql_track.plan import (
    RunPlan,
    JobPlanEntry,
    build_run_plan,
    load_run_json,
    save_run_json,
    get_active_job_keys,
    get_skipped_job_keys,
    make_key,
    interactive_confirm,
    display_plan,
    confirm_and_save,
)
from gsql_track.util import ProgressWriter


# ── RunPlan build ──

def test_build_run_plan_fresh(tmp_path):
    run_json = tmp_path / "run.json"
    plan = build_run_plan(
        [("BERT", "SST2"), ("CNN", "MNIST")],
        run_json,
        n_workers=2,
        device_ids=[0, 1],
    )
    assert len(plan.jobs) == 2
    assert all(not j.skip for j in plan.jobs)
    assert plan.jobs[0].key == "BERT_SST2"


def test_build_run_plan_preserves_skip(tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(json.dumps({"jobs": {"BERT_SST2": {"skip": True}}}))
    plan = build_run_plan([("BERT", "SST2"), ("CNN", "MNIST")], run_json)
    assert plan.jobs[0].skip is True
    assert plan.jobs[0].skip_reason == "run.json"
    assert plan.jobs[1].skip is False


def test_active_and_skipped_keys():
    plan = RunPlan(jobs=[
        JobPlanEntry(key="A_B", model="A", task="B", skip=False),
        JobPlanEntry(key="C_D", model="C", task="D", skip=True),
    ])
    assert get_active_job_keys(plan) == {"A_B"}
    assert get_skipped_job_keys(plan) == {"C_D"}


def test_make_key():
    assert make_key("BERT", "AGNews") == "BERT_AGNews"


# ── run.json I/O ──

def test_save_load_roundtrip(tmp_path):
    run_json = tmp_path / "run.json"
    plan = RunPlan(jobs=[
        JobPlanEntry(key="X_Y", model="X", task="Y", skip=True),
        JobPlanEntry(key="A_B", model="A", task="B", skip=False),
    ])
    save_run_json(run_json, plan)
    data = load_run_json(run_json)
    assert data["jobs"]["X_Y"]["skip"] is True
    assert data["jobs"]["A_B"]["skip"] is False


def test_load_run_json_missing_file(tmp_path):
    result = load_run_json(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_run_json_corrupt(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid json")
    assert load_run_json(bad) == {}


# ── interactive_confirm (mocked input) ──

def test_interactive_confirm_accept():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", return_value="y"):
        result = interactive_confirm(plan, writer)
    assert result is plan


def test_interactive_confirm_quit():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "quit"]):
        result = interactive_confirm(plan, writer)
    assert result is None


def test_interactive_confirm_skip_then_done():
    plan = RunPlan(jobs=[
        JobPlanEntry(key="A_B", model="A", task="B"),
        JobPlanEntry(key="C_D", model="C", task="D"),
    ])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "skip 1", "done"]):
        result = interactive_confirm(plan, writer)
    assert result is not None
    assert result.jobs[0].skip is True
    assert result.jobs[1].skip is False


# ── Interactive commands ──

def test_interactive_unskip():
    plan = RunPlan(jobs=[
        JobPlanEntry(key="A_B", model="A", task="B", skip=True, skip_reason="run.json"),
    ])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "unskip 1", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.jobs[0].skip is False


def test_interactive_devices():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "devices 0,2", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.device_ids == [0, 2]
    assert result.n_workers == 2


def test_interactive_workers():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "workers 4", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_workers == 4


def test_interactive_seeds_bench():
    plan = RunPlan(
        jobs=[JobPlanEntry(key="A_B", model="A", task="B")],
        manager_type="bench",
    )
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "seeds 5", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_seeds == 5
    assert result.jobs[0].info == "5 seeds"


def test_interactive_resume_mode():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "resume fresh", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.resume_mode == "fresh"


def test_interactive_eof():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=EOFError):
        result = interactive_confirm(plan, writer)
    assert result is None


# ── display_plan ──

def test_display_plan_output(capsys):
    plan = RunPlan(
        jobs=[
            JobPlanEntry(key="A_B", model="A", task="B", info="3 seeds"),
            JobPlanEntry(key="C_D", model="C", task="D", skip=True, skip_reason="run.json"),
        ],
        manager_type="bench",
    )
    writer = ProgressWriter(silent=False)
    display_plan(plan, writer)
    out = capsys.readouterr().out
    assert "1 to run" in out
    assert "1 to skip" in out


# ── confirm_and_save auto-confirm ──

def test_confirm_and_save_auto(tmp_path):
    run_json = tmp_path / "run.json"
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    result = confirm_and_save(plan, run_json, writer, auto_confirm=True)
    assert result is plan
    assert run_json.exists()


# ── interactive trials (tune) ──

def test_interactive_trials_tune():
    plan = RunPlan(
        jobs=[JobPlanEntry(key="A_B", model="A", task="B")],
        manager_type="tune",
    )
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "trials 50", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_trials == 50
    assert result.jobs[0].info == "50 trials"


# ── pipe commands ──

def test_interactive_pipe_commands():
    plan = RunPlan(jobs=[
        JobPlanEntry(key="A_B", model="A", task="B"),
        JobPlanEntry(key="C_D", model="C", task="D"),
    ])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "skip 1 | workers 3", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.jobs[0].skip is True
    assert result.n_workers == 3


# ── invalid resume mode ──

def test_interactive_invalid_resume():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "resume badmode", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.resume_mode == "resume"  # unchanged


# ── unknown command ──

def test_interactive_unknown_command():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "foobar", "done"]):
        result = interactive_confirm(plan, writer)
    assert result is plan  # should still work


# ── invalid inputs ──

def test_interactive_invalid_devices():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "devices abc", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.device_ids is None  # unchanged


def test_interactive_invalid_workers():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "workers xyz", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_workers == 1  # unchanged


def test_interactive_invalid_seeds():
    plan = RunPlan(
        jobs=[JobPlanEntry(key="A_B", model="A", task="B")],
        manager_type="bench",
    )
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "seeds 0", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_seeds is None  # unchanged


def test_interactive_invalid_trials():
    plan = RunPlan(
        jobs=[JobPlanEntry(key="A_B", model="A", task="B")],
        manager_type="tune",
    )
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", "trials abc", "done"]):
        result = interactive_confirm(plan, writer)
    assert result.n_trials is None  # unchanged


# ── keyboard interrupt in command loop ──

def test_interactive_keyboard_interrupt_in_loop():
    plan = RunPlan(jobs=[JobPlanEntry(key="A_B", model="A", task="B")])
    writer = ProgressWriter(silent=True)
    with patch("builtins.input", side_effect=["n", KeyboardInterrupt]):
        result = interactive_confirm(plan, writer)
    assert result is None


# ── build_run_plan with job_info ──

def test_build_run_plan_with_info(tmp_path):
    plan = build_run_plan(
        [("BERT", "SST2")],
        tmp_path / "run.json",
        job_info={"BERT_SST2": "3 seeds"},
    )
    assert plan.jobs[0].info == "3 seeds"


# ── display_plan no info column ──

def test_display_plan_no_info(capsys):
    plan = RunPlan(jobs=[
        JobPlanEntry(key="A_B", model="A", task="B"),
    ])
    writer = ProgressWriter(silent=False)
    display_plan(plan, writer)
    out = capsys.readouterr().out
    assert "1 to run" in out
