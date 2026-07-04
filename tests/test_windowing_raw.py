"""60 s raw statistics: wind vector averaging across the 0/360 wrap."""

import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP import windowing_raw
from ASCENT_ACP.config import PipelineConfig

from test_filtering import P_NAV


def _wind_df(directions, speed=10.0):
    idx = pd.date_range("2021-05-13 12:00", periods=len(directions), freq="1s", tz="UTC")
    return pd.DataFrame({
        P_NAV + "Wind_Speed": speed,
        P_NAV + "Wind_Direction": directions,
        P_NAV + "Latitude": 37.0,
    }, index=idx)


def test_direction_mean_across_north_wrap():
    # naive mean of 350..10 deg is ~180; vector mean must be ~0/360
    dirs = np.concatenate([np.full(30, 355.0), np.full(30, 5.0)])
    out = windowing_raw.aggregate_raw(_wind_df(dirs), PipelineConfig())
    d = out[P_NAV + "Wind_Direction_mean"].iloc[0]
    assert d < 1.0 or d > 359.0
    # spread is small, Yamartino std ~5 deg
    assert out[P_NAV + "Wind_Direction_std"].iloc[0] == pytest.approx(5.0, abs=1.0)


def test_vector_vs_scalar_speed():
    # opposing winds cancel: vector speed ~0, scalar mean stays 10
    dirs = np.concatenate([np.full(30, 0.0), np.full(30, 180.0)])
    out = windowing_raw.aggregate_raw(_wind_df(dirs), PipelineConfig())
    assert out[P_NAV + "Wind_Speed_mean"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert out[P_NAV + "Wind_Speed_scalar_mean"].iloc[0] == pytest.approx(10.0)


def test_steady_wind_unchanged():
    dirs = np.full(60, 270.0)
    out = windowing_raw.aggregate_raw(_wind_df(dirs), PipelineConfig())
    assert out[P_NAV + "Wind_Direction_mean"].iloc[0] == pytest.approx(270.0)
    assert out[P_NAV + "Wind_Speed_mean"].iloc[0] == pytest.approx(10.0)
    assert out[P_NAV + "Wind_Direction_std"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_other_columns_still_plain_mean():
    dirs = np.full(60, 90.0)
    df = _wind_df(dirs)
    out = windowing_raw.aggregate_raw(df, PipelineConfig())
    assert out[P_NAV + "Latitude_mean"].iloc[0] == pytest.approx(37.0)
    assert out["n_points"].iloc[0] == 60
