import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP.config import PSDConfig
from ASCENT_ACP import sizebins


@pytest.fixture
def fake_df():
    cols = [f"Submicron_Particle_X_SMPS_Bin{i:02d}" for i in range(1, 31)]
    cols += [f"Particle_Size_X_LAS_Bin{i:02d}" for i in range(1, 27)]
    return pd.DataFrame(np.zeros((3, len(cols))), columns=cols)


def test_full_grid_monotonic_and_complete(fake_df):
    cfg = PSDConfig(psd_max_um=5.0)
    grid = sizebins.build_grid(fake_df, cfg)
    assert len(grid) == 56  # 30 SMPS + 26 LAS, nothing cut at 5 um
    assert np.all(np.diff(grid.dpg_um) > 0)
    assert np.all(grid.dpl_um < grid.dpg_um) and np.all(grid.dpg_um < grid.dpu_um)
    assert list(grid.instrument[:30]) == ["SMPS"] * 30


def test_submicron_truncation(fake_df):
    grid = sizebins.build_grid(fake_df, PSDConfig(psd_max_um=1.0))
    assert grid.dpg_um.max() <= 1.0
    # LAS centers run ...891, 1000, 1259 nm: the 1000 nm bin survives a 1.0 um cut
    assert np.isclose(grid.dpg_um.max(), 1.0)
    assert len(grid) == 30 + 21


def test_inlet_cutoff_overrides_larger_max(fake_df):
    g5 = sizebins.build_grid(fake_df, PSDConfig(psd_max_um=99.0, inlet_cutoff_um=5.0))
    assert g5.dpg_um.max() <= 5.0 and len(g5) == 56


def test_smps_min_dp_trim(fake_df):
    grid = sizebins.build_grid(fake_df, PSDConfig(psd_max_um=1.0, smps_min_dp_um=0.01))
    assert grid.dpg_um.min() >= 0.01
    assert grid.columns[0].endswith("SMPS_Bin11")  # 10.0 nm is the first kept center


def test_no_smps_las_overlap_in_shipped_tables():
    cfg = PSDConfig()
    smps = sizebins.load_bins(cfg.smps_bins_csv)
    las = sizebins.load_bins(cfg.las_bins_csv)
    assert smps["dpu"][-1] <= las["dpl"][0]
