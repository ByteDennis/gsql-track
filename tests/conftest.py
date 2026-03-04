import pytest
from gsql_track.gsql_track import GsqlTrack


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_track.db")


@pytest.fixture
def tracker(tmp_db):
    t = GsqlTrack("test-exp", db_path=tmp_db)
    yield t
    t.close()


@pytest.fixture
def sample_run(tracker):
    run = tracker.start_run("sample-run")
    run.log_params({"lr": 0.001, "bs": 64})
    return run
