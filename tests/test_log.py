"""Tests for gsql_track.log — logger lifecycle, GsqlLogger, registry."""
import pytest
from unittest.mock import patch

from gsql_track.log import (
    BaseLogger,
    GsqlLogger,
    get_logger,
    clear_logger_registry,
    _logger_registry,
    _set_default_logger,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset logger registry between tests."""
    clear_logger_registry()
    yield
    clear_logger_registry()


# ── GsqlLogger lifecycle ──

def test_gsql_logger_log_and_finish(tmp_path):
    db = str(tmp_path / "track.db")
    logger = GsqlLogger("test-run", config=GsqlLogger.Config(experiment="exp", db_path=db))
    logger.log({"loss": 0.5, "acc": 0.8}, step=1)
    logger.log_loss(0.3, step=2)
    logger.log_hyperparameters({"lr": 0.001, "nested": {"a": 1}})  # nested skipped
    logger.finish()
    # Verify DB has data
    import sqlite3
    conn = sqlite3.connect(db)
    runs = conn.execute("SELECT status FROM runs").fetchall()
    assert runs[0][0] == "FINISHED"
    metrics = conn.execute("SELECT key FROM metrics").fetchall()
    keys = {r[0] for r in metrics}
    assert "loss" in keys
    assert "acc" in keys
    conn.close()


def test_gsql_logger_context_manager(tmp_path):
    db = str(tmp_path / "track.db")
    with GsqlLogger("ctx-run", config=GsqlLogger.Config(experiment="exp", db_path=db)) as logger:
        logger.log({"x": 1.0}, step=0)
    import sqlite3
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT status FROM runs").fetchone()[0] == "FINISHED"
    conn.close()


# ── Logger registry ──

def test_get_logger_gsql(tmp_path):
    db = str(tmp_path / "track.db")
    logger = get_logger("gsql", name="reg-test", config=GsqlLogger.Config(experiment="e", db_path=db))
    assert isinstance(logger, GsqlLogger)
    # Same name returns cached instance
    assert get_logger(name="reg-test") is logger
    logger.finish()


def test_get_logger_default(tmp_path):
    db = str(tmp_path / "track.db")
    get_logger("gsql", name="default-test", config=GsqlLogger.Config(experiment="e", db_path=db), default_logger=True)
    default = get_logger()
    assert isinstance(default, GsqlLogger)
    default.finish()


def test_get_logger_no_default():
    with pytest.raises(ValueError, match="No default logger"):
        get_logger()


def test_get_logger_unknown_type():
    with pytest.raises(ValueError, match="Unknown logger type"):
        get_logger("nonexistent", name="x")


def test_clear_registry(tmp_path):
    db = str(tmp_path / "track.db")
    get_logger("gsql", name="temp", config=GsqlLogger.Config(experiment="e", db_path=db), default_logger=True)
    clear_logger_registry()
    with pytest.raises(ValueError):
        get_logger()


# ── BaseLogger config parsing ──

def test_base_logger_config_from_dict(tmp_path):
    db = str(tmp_path / "track.db")
    logger = GsqlLogger("dict-cfg", config={"experiment": "e", "db_path": db, "verbose": False})
    assert logger.config.verbose is False
    logger.finish()


def test_base_logger_callable(tmp_path, capsys):
    db = str(tmp_path / "track.db")
    logger = GsqlLogger("call-test", config=GsqlLogger.Config(experiment="e", db_path=db, verbose=True))
    logger("hello")  # should not raise
    logger.info("world")
    logger.finish()


# ── Logger (loguru-based) lifecycle ──

def test_logger_basic_lifecycle(tmp_path):
    from gsql_track.log import Logger
    log_path = tmp_path / "logs" / "test.log"
    logger = Logger("test", config=Logger.Config(log_path=str(log_path)))
    logger.log({"val_score": 0.8}, step=1)
    assert logger.best_metric == 0.8
    logger.log({"val_score": 0.7}, step=2)
    assert logger.best_metric == 0.8  # no improvement
    logger.log({"val_score": 0.9}, step=3)
    assert logger.best_metric == 0.9
    logger("hello from logger")
    logger.finish()


def test_logger_no_log_path():
    from gsql_track.log import Logger
    logger = Logger("test-no-path", config=Logger.Config(log_path=None))
    assert logger.log_dir is None
    logger.log({}, step=0)
    logger.finish()


def test_logger_hyperparameters(tmp_path):
    from gsql_track.log import Logger
    log_path = tmp_path / "logs" / "test.log"
    logger = Logger("test", config=Logger.Config(log_path=str(log_path)))
    logger.log_hyperparameters({"lr": 0.01, "bs": 64})
    hp_path = tmp_path / "logs" / "hyperparameters.json"
    assert hp_path.exists()
    import json
    data = json.loads(hp_path.read_text())
    assert data["lr"] == 0.01
    logger.finish()


def test_logger_checkpoint(tmp_path):
    from gsql_track.log import Logger
    log_path = tmp_path / "logs" / "test.log"
    logger = Logger("test", config=Logger.Config(log_path=str(log_path)))
    logger.create_checkpoint(step=5, state={"epoch": 5, "loss": 0.3})
    ckpt = tmp_path / "logs" / "model" / "checkpoint_step_5.pkl"
    assert ckpt.exists()
    logger.finish()


def test_logger_checkpoint_no_dir():
    from gsql_track.log import Logger
    logger = Logger("test", config=Logger.Config(log_path=None))
    logger.create_checkpoint(step=1, state={"x": 1})  # should not raise
    logger.finish()


# ── Logger context manager ──

def test_logger_context_manager():
    from gsql_track.log import Logger
    with Logger("ctx", config=Logger.Config(log_path=None)) as logger:
        logger.log({}, step=0)


# ── BaseLogger config error ──

def test_base_logger_bad_config_type():
    with pytest.raises(NotImplementedError):
        GsqlLogger("bad", config=42)


# ── get_logger name lookup without type ──

def test_get_logger_name_not_found():
    with pytest.raises(ValueError, match="not found"):
        get_logger(name="nonexistent")


# ── log_experiment decorator ──

def test_log_experiment_decorator_import():
    """Verify log_experiment is importable and callable."""
    from gsql_track.log import log_experiment, Logger
    # The decorator can be applied (but Config(config=...) is broken for most loggers)
    decorated = log_experiment(Logger)
    assert callable(decorated)
