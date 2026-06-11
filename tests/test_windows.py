import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP.config import PipelineConfig, PSDConfig
from ASCENT_ACP import filtering, sizebins, windows

from test_filtering import make_df, P_OPT, P_CDP  # noqa: F401


def add_psd_columns(df, value=100.0):
    for i in range(1, 31):
        df[f"Submicron_X_SMPS_Bin{i:02d}"] = value
    for i in range(1, 27):
        df[f"Particle_X_LAS_Bin{i:02d}"] = value
    return df


def run(df, cfg=None):
    cfg = cfg or PipelineConfig()
    grid = sizebins.build_grid(df, cfg.psd)
    opt = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, opt, cfg)
    return windows.aggregate(df, opt, masks, grid, cfg), grid


def test_clean_window_passes():
    df = add_psd_columns(make_df(n=120))
    out, grid = run(df)
    assert len(out) == 2
    assert (out["window_qc_flag"] == 0).all()
    assert (out["n_valid"] == 60).all()
    assert np.allclose(out["Sc450_dry_mean"], 50.0)
    assert np.allclose(out[windows.psd_col_name(grid.dpg_um[0])], 100.0)
    # center-time index and bounds
    assert out["time_end"].iloc[0] - out["time_start"].iloc[0] == pd.Timedelta("60s")
    assert (out.index - out["time_start"] == pd.Timedelta("30s")).all()


def test_empty_windows_dropped():
    df1 = add_psd_columns(make_df(n=120))
    df2 = add_psd_columns(make_df(n=120))
    df2.index = df2.index + pd.Timedelta(hours=6)  # flight gap
    out, _ = run(pd.concat([df1, df2]))
    assert len(out) == 4  # 2 windows per segment, none for the 6 h gap


def test_too_few_points_flagged():
    df = add_psd_columns(make_df(n=120))
    n_cdp = np.zeros(120)
    n_cdp[0:55] = 50.0  # cloud kills most of window 1 (plus 5 s padding)
    df[P_CDP + "N_CDP"] = n_cdp
    out, _ = run(df)
    assert out["window_qc_flag"].iloc[0] & windows.FLAG_TOO_FEW_POINTS
    assert out["n_cloudy"].iloc[0] == 60
    assert out["window_qc_flag"].iloc[1] == 0


def test_unstable_ae_flagged():
    df = add_psd_columns(make_df(n=60))
    rng = np.random.default_rng(0)
    df[P_OPT + "AEscat_450to700nm"] = 1.0 + rng.normal(0, 0.8, 60)  # very noisy
    out, _ = run(df)
    assert out["window_qc_flag"].iloc[0] & windows.FLAG_AE_UNSTABLE


def test_missing_absorption_flagged():
    df = add_psd_columns(make_df(n=60))
    df[P_OPT + "Abs532_total"] = np.nan
    out, _ = run(df)
    assert out["window_qc_flag"].iloc[0] & windows.FLAG_MISSING_OPTICS


def test_sparse_psd_bin_naned():
    cfg = PipelineConfig()
    df = add_psd_columns(make_df(n=60))
    col = "Particle_X_LAS_Bin05"
    vals = np.full(60, 100.0)
    vals[5:] = np.nan  # only 5 valid samples < min_valid_points_per_bin=10
    df[col] = vals
    out, grid = run(df, cfg)
    i = grid.columns.index(col)
    assert np.isnan(out[windows.psd_col_name(grid.dpg_um[i])]).all()
