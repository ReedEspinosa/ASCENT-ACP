import sys
import types

import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP import filtering, sizebins, windows, isara_bridge

from test_filtering import make_df
from test_windows import add_psd_columns


@pytest.fixture
def fake_isara(monkeypatch, tmp_path):
    """Install a stub ISARA module that records calls instead of running MOPSMAP."""
    calls = []

    def fake_retr_psd(**kwargs):
        calls.append(kwargs)
        return {
            "dry_RRI_unitless": 1.53,
            "dry_IRI_unitless": 0.002,
            "kappa_unitless": 0.45,
            "attempt_flag_CRI_unitless": 2,
            "attempt_flag_kappa_unitless": 2,
        }

    mod = types.ModuleType("ISARA")
    mod.Retr_PSD = fake_retr_psd
    monkeypatch.setitem(sys.modules, "ISARA", mod)
    monkeypatch.setattr(isara_bridge, "_ISARA", None)
    monkeypatch.setattr(isara_bridge, "_LUTS", {})
    return calls


def make_windows(cfg):
    df = add_psd_columns(make_df(n=120))
    grid = sizebins.build_grid(df, cfg.psd)
    opt = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, opt, cfg)
    return windows.aggregate(df, opt, masks, grid, cfg), grid


def make_cfg(tmp_path):
    cfg = PipelineConfig()
    cfg.isara.n_workers = 1
    cfg.isara.use_lut = False  # LUT plumbing tested separately
    cfg.paths.scratch_dir = str(tmp_path)
    return cfg


def test_kwargs_units_and_shapes(fake_isara, tmp_path):
    cfg = make_cfg(tmp_path)
    wdf, grid = make_windows(cfg)
    res = isara_bridge.run_all_windows(wdf, grid, cfg, progress=False)

    assert len(res) == 2 and len(fake_isara) == 2
    kw = fake_isara[0]
    # radii are bin-center DIAMETERS / 2
    assert np.allclose(kw["radii_um"] * 2, grid.dpg_um)
    # Mm-1 -> m-1: Sc450 = 50 Mm-1 -> 50e-6 m-1 (RH=30 < 40, no drying applied)
    assert np.isclose(kw["dry_sca_coef"][0], 50.0e-6)
    assert np.isclose(kw["dry_abs_coef"][0], 2.0e-6)
    # wet channel is gamma-humidified upward from 35 Mm-1 at 550
    assert kw["wet_sca_coef"][0] > 35.0e-6
    assert kw["dry_wvl"] == {"sca": [450, 550, 700], "abs": [470, 532, 660]}
    assert kw["RH_wet"] == 80.0
    assert (res["dry_RRI_unitless"] == 1.53).all()


def test_nan_psd_bins_passed_through(fake_isara, tmp_path):
    cfg = make_cfg(tmp_path)
    wdf, grid = make_windows(cfg)
    col = windows.psd_col_name(grid.dpg_um[3])
    wdf[col] = np.nan
    isara_bridge.run_all_windows(wdf, grid, cfg, progress=False)
    assert np.isnan(fake_isara[0]["dndlogdp_cm3"][3])
    assert np.isfinite(fake_isara[0]["dndlogdp_cm3"][4])


def test_flagged_windows_skipped(fake_isara, tmp_path):
    cfg = make_cfg(tmp_path)
    wdf, grid = make_windows(cfg)
    wdf.loc[wdf.index[0], "window_qc_flag"] = windows.FLAG_AE_UNSTABLE
    res = isara_bridge.run_all_windows(wdf, grid, cfg, progress=False)
    assert len(res) == 1 and len(fake_isara) == 1


def test_lut_built_per_pattern_and_passed(monkeypatch, tmp_path):
    """With use_lut on, common bin patterns get a LUT routed into Retr_PSD."""
    received = []

    def fake_retr_psd(lut=None, **kwargs):
        received.append(lut)
        return {"attempt_flag_CRI_unitless": 2, "attempt_flag_kappa_unitless": 0}

    builds = []

    class FakeLUT:
        def __init__(self, **state):
            self.state = state
            for k, v in state.items():
                setattr(self, k, v)

    def fake_build(wvl, cri_grid, dpg_um, *a, **k):
        builds.append(np.asarray(dpg_um))
        n = (len(cri_grid), len(wvl), len(dpg_um))
        return FakeLUT(
            wvl_nm=np.asarray(wvl), cri_grid=np.asarray(cri_grid),
            dpg_um=np.asarray(dpg_um), K_ext=np.zeros(n), K_sca=np.zeros(n),
            size_equ="cs", nonabs_fraction=0.0, shape="sphere", rho=1.0,
            num_theta=2,
        )

    isara_mod = types.ModuleType("ISARA")
    isara_mod.Retr_PSD = fake_retr_psd
    isara_mod.default_CRI_grid = lambda: np.array([[1.52, 0.001], [1.53, 0.002]])
    lut_mod = types.ModuleType("optics_lut")
    lut_mod.build = fake_build
    lut_mod.OpticsLUT = FakeLUT
    monkeypatch.setitem(sys.modules, "ISARA", isara_mod)
    monkeypatch.setitem(sys.modules, "optics_lut", lut_mod)
    monkeypatch.setattr(isara_bridge, "_ISARA", None)
    monkeypatch.setattr(isara_bridge, "_LUTS", {})

    cfg = make_cfg(tmp_path)
    cfg.isara.use_lut = True
    cfg.isara.lut_min_pattern_count = 1
    cfg.paths.output_dir = str(tmp_path)
    wdf, grid = make_windows(cfg)
    # knock one bin out of the SECOND window only -> two patterns, two LUTs
    col = windows.psd_col_name(grid.dpg_um[3])
    wdf.loc[wdf.index[1], col] = np.nan

    res = isara_bridge.run_all_windows(wdf, grid, cfg, progress=False)
    assert len(res) == 2
    assert len(builds) == 2  # full pattern + one-bin-missing pattern
    assert all(isinstance(lut, FakeLUT) for lut in received)
    sizes = sorted(lut.state["dpg_um"].size for lut in received)
    assert sizes == [len(grid) - 1, len(grid)]


def test_value_error_becomes_flag_zero(monkeypatch, tmp_path):
    def angry(**kwargs):
        raise ValueError("At least 2 valid PSD bins are required.")

    mod = types.ModuleType("ISARA")
    mod.Retr_PSD = angry
    monkeypatch.setitem(sys.modules, "ISARA", mod)
    monkeypatch.setattr(isara_bridge, "_ISARA", None)

    cfg = make_cfg(tmp_path)
    wdf, grid = make_windows(cfg)
    res = isara_bridge.run_all_windows(wdf, grid, cfg, progress=False)
    assert (res["attempt_flag_CRI_unitless"] == 0).all()
    assert res["retrieval_error"].str.contains("2 valid").all()
