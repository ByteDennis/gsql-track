"""Tests for gsql_track.util — Timer, EarlyStopping, dict utils, JSONEncoder, ProgressWriter."""
import io
import json
import random
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest
from gsql_track.util import (
    Timer,
    EarlyStopping,
    JSONEncoder,
    ProgressWriter,
    BaseProgressTracker,
    Tee,
    fmt_float,
    set_seed,
    unnest_dict,
    nest_dict,
    merge_dict,
    fmt_duration,
)


# ── Timer ──

def test_timer_start_stop():
    t = Timer()
    t.start()
    time.sleep(0.05)
    elapsed = t.stop()
    assert elapsed >= 0.04
    assert t.elapsed == elapsed


def test_timer_reset():
    t = Timer()
    t.start()
    time.sleep(0.02)
    t.stop()
    t.reset()
    assert t.elapsed == 0.0
    assert t.start_time is None


def test_timer_stop_without_start():
    t = Timer()
    elapsed = t.stop()
    assert elapsed == 0.0


def test_timer_format_time():
    ts = datetime(2024, 1, 15, 10, 30, 0).timestamp()
    result = Timer.format_time(ts)
    assert "2024" in result
    assert Timer.format_time(None) == "-"


# ── EarlyStopping ──

def test_early_stopping_maximize():
    es = EarlyStopping(patience=3, direction="maximize")
    es.update(0.5)  # new best
    assert es.best_value == 0.5
    assert es.counter == 0
    es.update(0.4)  # worse
    assert es.counter == 1
    es.update(0.3)  # worse
    es.update(0.2)  # worse
    assert es.should_stop()


def test_early_stopping_minimize():
    es = EarlyStopping(patience=2, direction="minimize")
    es.update(1.0)
    es.update(0.8)  # better
    assert es.counter == 0
    es.update(0.9)  # worse
    es.update(1.0)  # worse
    assert es.should_stop()


def test_early_stopping_reset_on_improvement():
    es = EarlyStopping(patience=3, direction="maximize")
    es.update(0.5)
    es.update(0.4)  # counter=1
    es.update(0.6)  # new best, counter=0
    assert es.counter == 0
    assert es.best_value == 0.6


def test_early_stopping_disabled():
    es = EarlyStopping(patience=0)
    es.update(0.5)
    es.update(0.4)
    assert not es.should_stop()


def test_early_stopping_should_end():
    es = EarlyStopping(n_epochs=5, n_steps=100)
    assert not es.should_end(step=50, epoch=3)
    assert es.should_end(step=50, epoch=5)
    assert es.should_end(step=100, epoch=3)


# ── Dict utilities ──

def test_unnest_dict():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    flat = unnest_dict(nested)
    assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_nest_dict():
    flat = {"a.b": 1, "a.c.d": 2, "e": 3}
    nested = nest_dict(flat)
    assert nested == {"a": {"b": 1, "c": {"d": 2}}, "e": 3}


def test_unnest_nest_roundtrip():
    original = {"x": {"y": 10, "z": {"w": 20}}, "top": 5}
    assert nest_dict(unnest_dict(original)) == original


def test_merge_dict():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = merge_dict(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


# ── JSONEncoder ──

def test_json_encoder_path():
    data = {"p": Path("/tmp/test")}
    result = json.dumps(data, cls=JSONEncoder)
    assert "/tmp/test" in result


def test_json_encoder_datetime():
    data = {"dt": datetime(2024, 6, 15, 12, 0)}
    result = json.dumps(data, cls=JSONEncoder)
    assert "2024-06-15" in result


def test_json_encoder_enum():
    class Color(Enum):
        RED = "red"

    data = {"c": Color.RED}
    result = json.dumps(data, cls=JSONEncoder)
    assert "red" in result


def test_json_encoder_set():
    data = {"s": {1, 2, 3}}
    result = json.loads(json.dumps(data, cls=JSONEncoder))
    assert sorted(result["s"]) == [1, 2, 3]


# ── set_seed ──

def test_set_seed_reproducibility():
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


# ── fmt_duration ──

@pytest.mark.parametrize("secs,expected", [
    (30, "30.0s"),
    (90, "1.5m"),
    (3700, "1h 1m"),
])
def test_fmt_duration(secs, expected):
    assert fmt_duration(secs) == expected


# ── fmt_float ──

def test_fmt_float():
    assert fmt_float(0.123456, 2) == "0.12"
    assert fmt_float("N/A") == "N/A"


# ── ProgressWriter ──

def test_progress_writer_silent(capsys):
    pw = ProgressWriter(silent=True)
    pw.write_progress("hello")
    pw.write_event("world")
    assert capsys.readouterr().out == ""


def test_progress_writer_output(capsys):
    pw = ProgressWriter(silent=False)
    pw.write_event("event msg")
    assert "event msg" in capsys.readouterr().out


# ── BaseProgressTracker ──

def test_progress_tracker_eta():
    pw = ProgressWriter(silent=True)
    tracker = BaseProgressTracker(total=10, writer=pw)
    tracker.completed = 5
    eta = tracker.calculate_eta(10.0)  # 10s for 5 items
    assert "ETA" in eta


def test_progress_tracker_eta_zero():
    pw = ProgressWriter(silent=True)
    tracker = BaseProgressTracker(total=10, writer=pw)
    assert tracker.calculate_eta(5.0) == ""  # no completed items


# ── Tee ──

def test_tee_writes_to_multiple():
    buf1 = io.StringIO()
    buf2 = io.StringIO()
    tee = Tee(buf1, buf2, strip_ansi=False)
    tee.write("hello\n")
    tee.flush()
    assert buf1.getvalue() == "hello\n"
    assert buf2.getvalue() == "hello\n"


# ── Tee ANSI stripping ──

def test_tee_strips_ansi():
    """Tee strips ANSI from non-stdout/stderr files."""
    buf = io.StringIO()
    # Tee checks f not in (sys.__stdout__, sys.__stderr__) for stripping
    tee = Tee(sys.__stdout__, buf, strip_ansi=True)
    tee.write("normal text")
    tee.flush()
    assert "normal text" in buf.getvalue()


def test_tee_skips_carriage_return():
    """Tee skips lines starting with \\r for non-terminal files."""
    buf = io.StringIO()
    tee = Tee(sys.__stdout__, buf, strip_ansi=True)
    tee.write("\rprogress 50%")
    tee.flush()
    assert buf.getvalue() == ""  # \r lines skipped for file output


def test_tee_isatty():
    buf = io.StringIO()
    tee = Tee(buf, strip_ansi=False)
    assert tee.isatty() is False


def test_tee_fileno_error():
    buf = io.StringIO()
    tee = Tee(buf, strip_ansi=False)
    with pytest.raises(AttributeError):
        tee.fileno()


# ── no_tqdm_pbar ──

def test_no_tqdm_pbar_disabled():
    from gsql_track.util import no_tqdm_pbar
    with no_tqdm_pbar(disable=True):
        pass  # should not raise


def test_no_tqdm_pbar_enabled():
    from gsql_track.util import no_tqdm_pbar
    with no_tqdm_pbar(disable=False):
        pass  # should not raise


# ── redirect_output_to_file ──

def test_redirect_output_to_file(tmp_path):
    from gsql_track.util import redirect_output_to_file
    log_file = tmp_path / "out.log"
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        tee = redirect_output_to_file(log_file)
        assert isinstance(tee, Tee)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ── import_function ──

def test_import_function():
    from gsql_track.util import import_function
    fn = import_function("json.dumps")
    assert fn is json.dumps


def test_import_function_error():
    from gsql_track.util import import_function
    with pytest.raises(ImportError):
        import_function("nonexistent.module.func")


# ── flat_dict ──

def test_flat_dict():
    from gsql_track.util import flat_dict
    result = flat_dict({"a.b": 1, "a.c": 2})
    assert result == {"a": {"b": 1, "c": 2}}


# ── print_title ──

def test_print_title(capsys):
    from gsql_track.util import print_title
    print_title("Hello")
    out = capsys.readouterr().out
    assert "Hello" in out
    assert "=" in out


def test_print_title_with_logger():
    from gsql_track.util import print_title
    lines = []
    print_title("Test", logger=lines.append)
    assert any("Test" in l for l in lines)


# ── cleanup_resources ──

def test_cleanup_resources():
    from gsql_track.util import cleanup_resources
    cleanup_resources()  # should not raise even without GPU


# ── pre_import_modules ──

def test_pre_import_modules(capsys):
    from gsql_track.util import pre_import_modules
    pre_import_modules(["json"], silent=False)
    out = capsys.readouterr().out
    assert "json" in out


def test_pre_import_modules_failure(capsys):
    from gsql_track.util import pre_import_modules
    pre_import_modules(["nonexistent_xyz"], silent=False)
    out = capsys.readouterr().out
    assert "Failed" in out
