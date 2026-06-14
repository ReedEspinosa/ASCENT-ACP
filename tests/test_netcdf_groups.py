"""Grouped netCDF v2 export: structure, native-cadence, fill/flag, families."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP import filtering, netcdf_export, results, sizebins, windows

from test_filtering import make_df, P_OPT, P_CDP  # noqa: F401
from test_windows import add_psd_columns


def build_results(n=180, cfg=None):
    """Realistic windowed results (no ISARA) from a synthetic merged frame."""
    cfg = cfg or PipelineConfig()
    df = add_psd_columns(make_df(n=n))
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    res = results.assemble(wdf, pd.DataFrame(), grid, cfg)
    return df, masks, res, grid, cfg


def make_tree(n=180):
    df, masks, res, grid, cfg = build_results(n=n)
    dt = netcdf_export.build_datatree(df, masks, res, grid, cfg, meta=None)
    return dt, df, res, cfg


def test_group_tree_present():
    dt, df, res, cfg = make_tree()
    groups = set(dt.groups)
    for g in ["/observations", "/windowed", "/windowed/retrievals", "/windowed/raw",
              "/observations/optical", "/windowed/raw/optical"]:
        assert g in groups
    # default config has no shift-diagnostics CSV -> no clock_alignment group
    assert "/clock_alignment" not in groups


def test_native_cadence_detected():
    dt, df, res, cfg = make_tree()
    obs = dt["/observations"].dataset
    assert obs.attrs["native_sampling_seconds"] == pytest.approx(1.0)
    assert obs.sizes["time_obs"] == len(df)


def test_windowed_raw_shares_time():
    dt, df, res, cfg = make_tree()
    rawopt = dt["/windowed/raw/optical"]
    assert "time" in rawopt.coords  # inherited from /windowed
    assert rawopt.sizes["time"] == len(res)


def test_retrieval_qc_flag_logic():
    df, masks, res, grid, cfg = build_results()
    # passing windows with no ISARA -> attempted-but-failed (2); failing -> 1
    res["attempt_flag_CRI_unitless"] = np.where(res["window_qc_flag"] == 0, 2, 0)
    results.add_retrieval_qc_flag(res)
    passed = res["window_qc_flag"] == 0
    assert (res.loc[passed, "retrieval_qc_flag"] == results.RETRIEVAL_OK).all()
    assert (res.loc[~passed, "retrieval_qc_flag"] == results.RETRIEVAL_SKIPPED_QA).all()


def test_fill_where_qa_fails(tmp_path):
    df, masks, res, grid, cfg = build_results()
    # force one window to fail QA, give the rest a successful CRI value
    res["attempt_flag_CRI_unitless"] = np.where(res["window_qc_flag"] == 0, 2, 0)
    res["dry_RRI_unitless"] = np.where(res["window_qc_flag"] == 0, 1.5, np.nan)
    results.add_retrieval_qc_flag(res)
    dt = netcdf_export.build_datatree(df, masks, res, grid, cfg, meta=None)
    path = netcdf_export.write(dt, tmp_path / "v2.nc", cfg)
    o = xr.open_datatree(path)
    r = o["/windowed/retrievals"].to_dataset()
    w = o["/windowed"].to_dataset()
    bad = w.window_qc_flag.values != 0
    assert np.all(np.isnan(r.refractive_index_real.values[bad]))
    assert np.array_equal(r.retrieval_qc_flag.values == 1, bad)


def test_roundtrip_and_units(tmp_path):
    dt, df, res, cfg = make_tree()
    path = netcdf_export.write(dt, tmp_path / "v2.nc", cfg)
    o = xr.open_datatree(path)
    assert o.attrs["Conventions"] == "CF-1.8"
    # scattering stored in m-1 (converted from Mm-1)
    sca = o["/windowed/retrievals"].to_dataset().scattering_dry_measured
    assert sca.attrs["units"] == "m-1"
    assert np.nanmax(sca.values) < 1e-3  # ~50 Mm-1 -> 5e-5 m-1


def test_clock_alignment_group_shift_decision(tmp_path):
    # apply_clock_alignment records applied shifts as decision "SHIFT"
    csv = tmp_path / "diag.csv"
    csv.write_text(
        "date,shift_group,n_valid,optimal_shift_s,peak_r,monotonic_halfwidth_s,decision,reason\n"
        "2021-01-29,Optical,5000,10.0,0.8,12,SHIFT,\n"
        "2021-01-29,AMS,0,,,,SKIP,n_valid=0<300\n"
    )
    cfg = PipelineConfig()
    cfg.paths.shift_diagnostics_csv = str(csv)
    ds = netcdf_export._clock_alignment_ds(cfg)
    import numpy as np
    gi = list(ds.shift_group.values).index("Optical")
    aj = list(ds.shift_group.values).index("AMS")
    assert ds.applied_shift_s.values[0, gi] == 10.0
    assert ds.decision_code.values[0, gi] == 1
    assert ds.applied_shift_s.values[0, aj] == 0.0  # SKIP -> 0
    assert ds.decision_code.values[0, aj] == 0


def test_row_qc_flag_bitmask():
    dt, df, res, cfg = make_tree()
    obs = dt["/observations"].dataset
    assert "row_qc_flag" in obs
    assert obs.row_qc_flag.dtype == np.int16
    assert "flag_masks" in obs.row_qc_flag.attrs
