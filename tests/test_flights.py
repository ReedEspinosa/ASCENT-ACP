"""Flight segmentation and the (flight x seconds-of-day) grid."""

import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP import flights
from ASCENT_ACP.config import PipelineConfig

from test_filtering import make_df


def test_single_flight_presence():
    cfg = PipelineConfig()
    df = make_df(n=120)
    g = flights.build(df, cfg)
    assert g.n_flights == 1
    assert g.source == "data-presence gaps"
    assert g.flight_id == ["20210513"]
    assert g.n_dropped == 0
    # seconds-of-day mapping: 12:00 UTC start
    assert g.row_sec[0] == 12 * 3600
    assert g.n_seconds == 12 * 3600 + 120


def test_two_same_day_flights_split_and_legs():
    cfg = PipelineConfig()
    a = make_df(n=120)
    b = make_df(n=90)
    b.index = b.index + pd.Timedelta(hours=3)
    df = pd.concat([a, b])
    g = flights.build(df, cfg)
    assert g.n_flights == 2
    assert g.flight_id == ["20210513_L1", "20210513_L2"]
    assert list(g.flight_number) == [1, 2]
    # both share the same takeoff-day midnight
    assert g.midnight_epoch_s[0] == g.midnight_epoch_s[1]
    # rows assigned to the right flight
    assert (g.row_flight[:120] == 0).all() and (g.row_flight[120:] == 1).all()


def test_midnight_crossing_extends_axis():
    cfg = PipelineConfig()
    df = make_df(n=7200)  # 2 h
    df.index = pd.date_range("2021-05-13 23:30", periods=7200, freq="1s", tz="UTC")
    g = flights.build(df, cfg)
    assert g.n_flights == 1
    assert g.date == ["2021-05-13"]  # takeoff day, not landing day
    assert g.landing_sod[0] > 86400
    assert g.n_seconds > 86400
    # last row maps beyond the 86400 boundary, still on the takeoff-day axis
    assert g.row_sec[-1] == 23 * 3600 + 30 * 60 + 7199  # 23:30 start + 2 h


def test_scatter_roundtrip():
    cfg = PipelineConfig()
    df = make_df(n=60)
    g = flights.build(df, cfg)
    vals = np.arange(60, dtype=float)
    arr = flights.scatter(g, vals)
    assert arr.shape == (1, g.n_seconds)
    assert np.allclose(arr[0, g.row_sec], vals)
    assert np.isnan(arr[0, : g.row_sec.min()]).all()


def test_icartt_span_source(tmp_path):
    # marker files define envelopes; rows outside them are dropped
    cfg = PipelineConfig()
    cfg.merge.icartt_dir = str(tmp_path)
    cfg.merge.flight_marker_instrument = "TEST-SUMMARY"
    # two same-day marker files (legs), like ACTIVATE L1/L2
    for leg, (t0, t1) in {"_L1": (43200, 43260), "_L2": (50400, 50460)}.items():
        lines = ["14, 1001", "PI", "ORG", "Test state instrument", "MISSION",
                 "1, 1", "2021, 05, 13, 2023, 01, 01", "1",
                 "Time_Start, sec", "1", "1", "-9999", "X, unit, STD, desc",
                 "DATA_INFO: none"]
        lines += [f"{t}.0, 1.0" for t in range(t0, t1)]
        (tmp_path / f"TEST-SUMMARY_HU25_20210513_R0{leg}.ict").write_text("\n".join(lines))

    idx = pd.date_range("2021-05-13 12:00", periods=7300, freq="1s", tz="UTC")
    df = pd.DataFrame({"v": 1.0}, index=idx)  # 12:00 -> 14:01:39, covers both spans
    g = flights.build(df, cfg)
    assert g.source == "ICARTT envelopes of TEST-SUMMARY"
    assert g.n_flights == 2
    assert g.flight_id == ["20210513_L1", "20210513_L2"]
    # rows between the padded envelopes are dropped
    assert g.n_dropped > 0
