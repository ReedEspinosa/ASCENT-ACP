"""Grouped netCDF v3 export: (flight,time) layout, compaction, broadcast."""

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP import filtering, flights, netcdf_export, results, sizebins, windows

from test_filtering import make_df, P_OPT, P_NAV  # noqa: F401
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


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    df, masks, res, grid, cfg = build_results()
    path = tmp_path_factory.mktemp("nc") / "v3.nc"
    netcdf_export.export(df, masks, res, grid, cfg, meta=None, path=path)
    return xr.open_datatree(path), df, res, grid, cfg


def test_group_tree_present(exported):
    o, df, res, grid, cfg = exported
    groups = set(o.groups)
    for g in ["/observations", "/windowed", "/windowed/retrievals",
              "/observations/optical"]:
        assert g in groups
    # raw variables live only in /observations; no 60 s repeat of them
    assert "/windowed/raw" not in groups
    # default config has no shift-diagnostics CSV -> no clock_alignment group
    assert "/clock_alignment" not in groups


def test_flight_time_layout(exported):
    o, df, res, grid, cfg = exported
    root = o.dataset
    assert root.sizes["flight"] == 1
    assert root.flight.values[0] == 1
    # time coordinate is seconds since takeoff-day midnight
    t0 = root.takeoff_time.values[0]
    assert t0 == pytest.approx(12 * 3600, abs=5)  # make_df starts 12:00 UTC
    # CF units make xarray decode midnight_epoch to the actual timestamp
    assert root.midnight_epoch.values[0] == np.datetime64("2021-05-13T00:00:00")
    assert root.sizes["time"] >= t0 + 180 - 1


def test_native_cadence_detected(exported):
    o, *_ = exported
    assert o["/observations"].attrs["native_sampling_seconds"] == pytest.approx(1.0)


def test_observation_values_land_on_grid(exported):
    o, df, res, grid, cfg = exported
    g = flights.build(df, cfg)
    got = o["/observations/optical"].to_dataset()["Sc550_submicron"].values[0]
    exp = df[P_OPT + "Sc550_submicron"].to_numpy(float)
    assert np.allclose(got[g.row_sec], exp, equal_nan=True, atol=1e-4)
    # off-flight seconds are fill
    assert np.isnan(got[: int(o.dataset.takeoff_time.values[0]) - 130]).all()


def test_psd_compacted_with_radius(exported):
    o, df, res, grid, cfg = exported
    # synthetic SMPS/LAS columns have unknown titles -> land in 'other'
    ds = o["/observations/other"].to_dataset()
    assert "dndlogd_smps" in ds and "dndlogd_las" in ds
    assert ds.dndlogd_smps.dims == ("flight", "time", "size_smps")
    # radius = diameter/2, ascending, from the packaged bin CSVs
    r = ds.radius_smps.values
    assert np.all(np.diff(r) > 0)
    assert np.allclose(ds.diameter_smps.values, 2 * r)
    assert np.all(ds.radius_lower_smps.values < r)
    assert np.all(ds.radius_upper_smps.values > r)
    # no leftover per-bin scalar variables anywhere
    assert not any("Bin" in v for v in ds.data_vars)


def test_windowed_broadcast_on_fine_grid(exported):
    o, df, res, grid, cfg = exported
    g = flights.build(df, cfg)
    w = o["/windowed"].to_dataset()
    assert w.window_qc_flag.dims == ("flight", "time")
    nv = w.n_valid.values[0][g.row_sec]
    centers = pd.DatetimeIndex(df.index.floor("60s") + pd.Timedelta(seconds=30))
    exp = res["n_valid"].reindex(centers).to_numpy(float)
    assert np.allclose(nv, exp, equal_nan=True)
    # a retrieval variable is constant across the seconds of one window
    ssa = o["/windowed/retrievals"].to_dataset()["ssa_measured"].values[0, :, 0]
    seg = ssa[g.row_sec[:60]]
    seg = seg[np.isfinite(seg)]
    if seg.size > 1:
        assert np.nanstd(seg) == pytest.approx(0.0, abs=1e-6)


def test_retrieval_qc_flag_logic():
    df, masks, res, grid, cfg = build_results()
    res["attempt_flag_CRI_unitless"] = np.where(res["window_qc_flag"] == 0, 2, 0)
    results.add_retrieval_qc_flag(res)
    passed = res["window_qc_flag"] == 0
    assert (res.loc[passed, "retrieval_qc_flag"] == results.RETRIEVAL_OK).all()
    assert (res.loc[~passed, "retrieval_qc_flag"] == results.RETRIEVAL_SKIPPED_QA).all()


def test_fill_where_qa_fails(tmp_path):
    df, masks, res, grid, cfg = build_results()
    res["attempt_flag_CRI_unitless"] = np.where(res["window_qc_flag"] == 0, 2, 0)
    res["dry_RRI_unitless"] = np.where(res["window_qc_flag"] == 0, 1.5, np.nan)
    results.add_retrieval_qc_flag(res)
    path = netcdf_export.export(df, masks, res, grid, cfg, meta=None,
                                path=tmp_path / "v3.nc")
    o = xr.open_datatree(path)
    r = o["/windowed/retrievals"].to_dataset()
    w = o["/windowed"].to_dataset()
    covered = ~np.isnan(w.window_qc_flag.values)
    bad = covered & (w.window_qc_flag.values != 0)
    assert np.all(np.isnan(r.refractive_index_real.values[bad]))
    assert np.array_equal(r.retrieval_qc_flag.values[covered] == 1, bad[covered])


def test_roundtrip_and_units(exported):
    o, *_ = exported
    assert o.attrs["Conventions"] == "CF-1.8"
    sca = o["/windowed/retrievals"].to_dataset().scattering_dry_measured
    assert sca.attrs["units"] == "m-1"
    assert sca.attrs["measurement_conditions"] == "STP"
    assert np.nanmax(sca.values) < 1e-3  # ~50 Mm-1 -> 5e-5 m-1


def test_row_qc_flag_bitmask(exported):
    o, *_ = exported
    obs = o["/observations"].to_dataset()
    assert "row_qc_flag" in obs
    assert "flag_masks" in obs.row_qc_flag.attrs


def test_measurement_conditions_attr_present(exported):
    o, *_ = exported
    ds = o["/observations/state_nav"].to_dataset()
    assert ds["Latitude"].attrs["measurement_conditions"] == "not_applicable"
    # no headers available in tests -> optical falls back to 'unspecified'
    opt = o["/observations/optical"].to_dataset()
    assert opt["Sc550_submicron"].attrs["measurement_conditions"] == "unspecified"


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
    gi = list(ds.shift_group.values).index("Optical")
    aj = list(ds.shift_group.values).index("AMS")
    assert ds.applied_shift_s.values[0, gi] == 10.0
    assert ds.decision_code.values[0, gi] == 1
    assert ds.applied_shift_s.values[0, aj] == 0.0  # SKIP -> 0
    assert ds.decision_code.values[0, aj] == 0
