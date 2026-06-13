"""Driver stage resume logic: which stages run vs. skip given checkpoints."""

import pytest

from ASCENT_ACP import run as driver
from ASCENT_ACP.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path):
    c = PipelineConfig(campaign="ACTIVATE", year="2021")
    c.paths.input_pkl = str(tmp_path / "merged_timeShifted.pkl")
    c.paths.meta_pickle = str(tmp_path / "merged_meta.pickle")
    c.paths.output_dir = str(tmp_path / "out")
    (tmp_path / "out").mkdir()
    return c


@pytest.fixture
def record(monkeypatch):
    """Replace the four stage functions with recorders; return the call list."""
    calls = []
    monkeypatch.setattr(driver, "stage_merge", lambda cfg, dates: calls.append("merge"))
    monkeypatch.setattr(driver, "stage_align", lambda cfg, plots: calls.append("align"))
    monkeypatch.setattr(driver, "stage_retrieve",
                        lambda cfg, dates, mw: calls.append("retrieve") or {})
    monkeypatch.setattr(driver, "stage_export", lambda cfg, state=None: calls.append("export"))
    return calls


def _touch(cfg, *stages):
    p = driver._paths(cfg)
    files = {"merge": [p["merged"], p["meta"]], "align": [p["shifted"]],
             "retrieve": [p["bundle"]], "export": [p["netcdf"]]}
    for s in stages:
        for f in files[s]:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")


def test_all_missing_runs_everything(cfg, record):
    driver.run(cfg)
    assert record == ["merge", "align", "retrieve", "export"]


def test_all_present_skips_everything(cfg, record):
    _touch(cfg, "merge", "align", "retrieve", "export")
    driver.run(cfg)
    assert record == []


def test_force_reruns_all(cfg, record):
    _touch(cfg, "merge", "align", "retrieve", "export")
    driver.run(cfg, force=True)
    assert record == ["merge", "align", "retrieve", "export"]


def test_missing_midstage_reruns_downstream(cfg, record):
    # merge + align present, retrieve missing -> retrieve and export run, not merge/align
    _touch(cfg, "merge", "align", "export")
    driver.run(cfg)
    assert record == ["retrieve", "export"]


def test_from_stage_forces_from_there(cfg, record):
    _touch(cfg, "merge", "align", "retrieve", "export")
    driver.run(cfg, from_stage="align")
    assert record == ["align", "retrieve", "export"]


def test_upstream_rerun_cascades(cfg, record):
    # only export missing, but from_stage=merge -> everything reruns
    _touch(cfg, "merge", "align", "retrieve")
    driver.run(cfg, from_stage="merge")
    assert record == ["merge", "align", "retrieve", "export"]
