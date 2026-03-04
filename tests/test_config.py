"""Tests for gsql_track.config — config loading, merging, dataclasses."""
import pytest
from omegaconf import OmegaConf

from gsql_track.config import (
    create_config,
    resolve_config,
    set_readonly,
    ResumeMode,
    resolve_resume_mode,
    merge_hierarchical_config,
    EvalConfig,
    SlackConfig,
    _normalize_display_metric,
    resolve_task_eval,
)


# ── create_config / resolve_config ──

def test_create_and_resolve():
    cfg = create_config({"a": 1, "b": {"c": 2}})
    resolved = resolve_config(cfg)
    assert resolved == {"a": 1, "b": {"c": 2}}


def test_resolve_from_raw_dict():
    resolved = resolve_config({"x": 10})
    assert resolved == {"x": 10}


# ── set_readonly ──

def test_set_readonly_prevents_mutation():
    cfg = create_config({"a": 1})
    set_readonly(cfg, True)
    with pytest.raises(Exception):
        cfg.a = 2
    set_readonly(cfg, False)
    cfg.a = 2
    assert cfg.a == 2


# ── ResumeMode ──

def test_resolve_resume_mode_string():
    assert resolve_resume_mode("fresh") == ResumeMode.fresh
    assert resolve_resume_mode(ResumeMode.resume) == ResumeMode.resume


def test_resume_mode_invalid():
    with pytest.raises(ValueError):
        resolve_resume_mode("invalid")


# ── merge_hierarchical_config ──

def test_merge_priority_order():
    """model > task > global"""
    g = {"lr": 0.001, "opt": {"name": "adam", "wd": 0.01}}
    t = {"lr": 0.01}
    m = {"opt": {"name": "sgd"}}
    result = merge_hierarchical_config(m, t, g)
    assert result["lr"] == 0.01          # task overrides global
    assert result["opt"]["name"] == "sgd"  # model overrides global
    assert result["opt"]["wd"] == 0.01     # global preserved


def test_merge_with_none_configs():
    result = merge_hierarchical_config(None, None, {"a": 1})
    assert result == {"a": 1}


def test_merge_terminal_marker():
    g = {"opt": {"name": "adam", "betas": [0.9]}}
    m = {"opt": {"name": "sgd", "_terminal": True}}
    result = merge_hierarchical_config(m, None, g)
    # Terminal means model's opt replaces entirely, no merge
    assert result["opt"] == {"name": "sgd"}


# ── EvalConfig ──

def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.primary_metric == "val.loss"
    assert cfg.patience is None
    assert cfg.direction.value == "minimize"


# ── SlackConfig ──

def test_slack_config_disabled_without_env(monkeypatch):
    monkeypatch.delenv("GSQL_SLACK_WEBHOOK_TEST", raising=False)
    cfg = SlackConfig(agent="test", enabled=True)
    assert cfg.enabled is False  # no webhook → disabled


def test_slack_config_enabled_with_env(monkeypatch):
    monkeypatch.setenv("GSQL_SLACK_WEBHOOK_BOT", "https://hooks.slack.com/xxx")
    cfg = SlackConfig(agent="bot", enabled=True)
    assert cfg.enabled is True
    assert cfg.webhook_url == "https://hooks.slack.com/xxx"


# ── _normalize_display_metric ──

@pytest.mark.parametrize("inp,expected", [
    (None, []),
    ("acc", ["acc"]),
    (["acc", "f1"], ["acc", "f1"]),
    (42, ["42"]),
])
def test_normalize_display_metric(inp, expected):
    assert _normalize_display_metric(inp) == expected


# ── resolve_task_eval ──

def test_resolve_task_eval_defaults():
    class FakeTask:
        config = None
    result = resolve_task_eval(FakeTask(), "val.loss", "minimize")
    assert result["primary_metric"] == "val.loss"
    assert result["direction"] == "minimize"
    assert result["display_metric"] == ["val.loss"]


def test_resolve_task_eval_overrides():
    class FakeTask:
        config = {"eval": {"primary_metric": "f1", "direction": "maximize"}}
    result = resolve_task_eval(FakeTask(), "val.loss", "minimize")
    assert result["primary_metric"] == "f1"
    assert result["direction"] == "maximize"


# ── load_config ──

def test_load_config_yaml(tmp_path):
    from gsql_track.config import load_config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("a: 1\nb:\n  c: 2\n")
    cfg = load_config(cfg_file)
    assert cfg.a == 1
    assert cfg.b.c == 2


# ── parse_config with overrides ──

def test_parse_config_with_overrides():
    from gsql_track.config import parse_config
    cfg = create_config({"a": 1, "b": 2})
    result = parse_config(cfg, overrides={"a": 99})
    resolved = resolve_config(result)
    assert resolved["a"] == 99
    assert resolved["b"] == 2


def test_parse_config_from_dict():
    from gsql_track.config import parse_config
    result = parse_config({"x": 10, "y": 20})
    resolved = resolve_config(result)
    assert resolved["x"] == 10


# ── OutputConfig ──

def test_output_config_creates_paths(tmp_path):
    from gsql_track.config import OutputConfig
    folder = str(tmp_path / "output")
    cfg = OutputConfig(
        folder=folder, save_model=True, save_pred=True,
        save_config=True, save_analysis=True, save_best=True, save_final=True,
    )
    assert "best_model_path" in cfg.save_paths
    assert "last_model_path" in cfg.save_paths
    assert "config_path" in cfg.save_paths
    assert "analysis_path" in cfg.save_paths
    assert "report_path" in cfg.save_paths


# ── generate_debug_cmd ──

def test_generate_debug_cmd():
    from gsql_track.config import generate_debug_cmd
    cmd = generate_debug_cmd("src.examples.train_model", {"lr": 0.001})
    assert "ipdb" in cmd
    assert "src/examples.py" in cmd
    assert "/tmp/debug_configs/" in cmd


# ── _parse_type ──

def test_parse_type_optional():
    from gsql_track.config import _parse_type
    from typing import Optional
    assert _parse_type(Optional[int]) is int
    assert _parse_type(int) is int


# ── set_readonly on nested ──

def test_set_readonly_nested():
    cfg = create_config({"a": {"b": 1}, "c": [1, 2]})
    set_readonly(cfg, True)
    with pytest.raises(Exception):
        cfg.a.b = 2
    set_readonly(cfg, False)
    cfg.a.b = 2
    assert cfg.a.b == 2


# ── YAML custom tags ──

def test_tune_yaml_tag():
    import yaml
    result = yaml.safe_load("val: !tune [0.001, 0.01, 0.1]")
    assert result["val"][0] == "!tune"
