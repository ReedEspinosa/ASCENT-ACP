import numpy as np
import pandas as pd
import pytest

from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP import filtering

P_OPT = "In-situ_optical_aerosol_measurements_from_the_NASA_HU-25_"
P_MIC = "In-situ_microphysical_aerosol_measurements_from_the_NASA_HU-25_"
P_NAV = "In-situ_state_and_aircraft_measurements_from_the_NASA_HU-25_"
P_CDP = "Cloud_Size_Distributions_From_the_DMT_Cloud_Droplet_Probe_"
P_FCDP = "Cloud_Size_Distributions_from_the_SPEC_Cloud_Droplet_Probe_FCDP_"


def make_cfg():
    cfg = PipelineConfig()
    # defaults are the 2021 suffixes; keep FCDP on
    return cfg


def make_df(n=30, **overrides):
    idx = pd.date_range("2021-05-13 12:00", periods=n, freq="1s", tz="UTC")
    data = {
        P_OPT + "Sc450_submicron": 50.0,
        P_OPT + "Sc550_submicron": 35.0,
        P_OPT + "Sc700_submicron": 20.0,
        P_OPT + "RH_Sc_submicron": 30.0,
        P_OPT + "Abs470_total": 2.0,
        P_OPT + "Abs532_total": 1.5,
        P_OPT + "Abs660_total": 1.0,
        P_OPT + "SSA_450nm": 0.95,
        P_OPT + "SSA_550nm": 0.95,
        P_OPT + "SSA_700nm": 0.95,
        P_OPT + "gamma550": 0.3,
        P_OPT + "fRH550_RH20to80": 1.5,
        P_OPT + "AEscat_450to700nm": 1.9,
        P_MIC + "InletFlag_LARGE": 0.0,
        P_CDP + "N_CDP": 0.0,
        P_CDP + "LWC_CDP": 0.0,
        P_FCDP + "N_FCDP": 0.0,
        P_FCDP + "LWC_FCDP": 0.0,
        P_NAV + "Latitude": 37.0,
        P_NAV + "Longitude": -75.0,
        P_NAV + "GPS_altitude": 500.0,
    }
    df = pd.DataFrame(data, index=idx)
    for col, vals in overrides.items():
        df[col] = vals
    return df


def test_gamma_adjust_roundtrip():
    sc40 = filtering.gamma_adjust_scattering(35.0, 0.3, 55.0, 40.0)
    back = filtering.gamma_adjust_scattering(sc40, 0.3, 40.0, 55.0)
    assert np.isclose(back, 35.0)
    # humidifying must increase scattering (gamma > 0)
    assert filtering.gamma_adjust_scattering(35.0, 0.3, 40.0, 80.0) > 35.0
    # drying must decrease it
    assert sc40 < 35.0


def test_derive_optical_dry_passthrough_below_ref_rh():
    cfg = make_cfg()
    df = make_df()
    opt = filtering.derive_optical_columns(df, cfg)
    # RH=30 <= 40: no drying applied
    assert np.allclose(opt["Sc450_dry"], 50.0)
    # wet at 80% from 30%: amplified
    assert (opt["Sc550_wet"] > 35.0).all()


def test_derive_optical_dries_when_humid():
    cfg = make_cfg()
    df = make_df(**{P_OPT + "RH_Sc_submicron": 60.0})
    opt = filtering.derive_optical_columns(df, cfg)
    assert (opt["Sc450_dry"] < 50.0).all()
    expected = 50.0 / np.exp(0.3 * np.log((100 - 40.0) / (100 - 60.0)))
    assert np.allclose(opt["Sc450_dry"], expected)


def test_cloud_mask_padding():
    cfg = make_cfg()
    n_cdp = np.zeros(30)
    n_cdp[15] = 50.0  # single cloudy second
    df = make_df(**{P_CDP + "N_CDP": n_cdp})
    cloudy = filtering.cloud_mask(df, cfg)
    assert cloudy.iloc[10:21].all()  # +/- 5 s padding
    assert not cloudy.iloc[:10].any() and not cloudy.iloc[21:].any()


def test_row_qc_masks():
    cfg = make_cfg()
    sc450 = np.full(30, 50.0)
    sc450[0] = 5.0  # low signal
    sc450[1] = np.nan  # missing -> unusable
    ssa = np.full(30, 0.95)
    ssa[2] = 0.5  # absorbing artifact
    ssa[3] = np.nan  # missing SSA passes
    inlet = np.zeros(30)
    inlet[4] = 1.0  # CVI
    df = make_df(
        **{
            P_OPT + "Sc450_submicron": sc450,
            P_OPT + "SSA_550nm": ssa,
            P_MIC + "InletFlag_LARGE": inlet,
        }
    )
    opt = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, opt, cfg)
    assert masks["low_signal"].iloc[0] and masks["low_signal"].iloc[1]
    assert masks["low_ssa"].iloc[2] and not masks["low_ssa"].iloc[3]
    assert masks["inlet_bad"].iloc[4]
    assert not masks["valid"].iloc[[0, 1, 2, 4]].any()
    assert masks["valid"].iloc[3]  # NaN SSA alone does not invalidate the row
    assert masks["valid"].iloc[5:].all()
