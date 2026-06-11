"""Row-level (1 Hz) quality control and RH adjustment of LARGE optical data.

Method follows Kacenelenbogen et al. (2022), ACP 22, 3713, Appendix A1.1:
cloud screening with wing-mounted probes, a minimum-signal filter on dry
scattering at 450 nm, an SSA sanity filter, plus the ACTIVATE inlet flag.

RH adjustments invert the gamma relation given in the LARGE OPTICAL ICARTT
header: ``SC_calcRH = SC_measRH / exp(GAMMA * ln((100-calcRH)/(100-measRH)))``.
"""

import numpy as np
import pandas as pd

from . import varmap


def gamma_adjust_scattering(sc, gamma, rh_from, rh_to):
    """Adjust scattering measured at ``rh_from`` (%) to ``rh_to`` (%)."""
    return sc / np.exp(gamma * np.log((100.0 - rh_to) / (100.0 - rh_from)))


def derive_optical_columns(df, cfg):
    """Return a working DataFrame of RH-standardized optical variables.

    Columns: ``Sc{wvl}_dry`` (at <= dry_ref_rh), ``Sc550_wet`` (at wet_rh),
    ``Abs{wvl}``, ``RH_Sc``, ``gamma``, ``AE``, ``SSA``, ``lat/lon/alt``.
    Done per 1 Hz row, before any averaging, so intra-window RH variability
    is handled exactly.
    """
    ch, flt = cfg.channels, cfg.filters
    out = pd.DataFrame(index=df.index)
    rh = df[varmap.resolve(df, ch.rh_sc_suffix)]
    gamma = df[varmap.resolve(df, ch.gamma_suffix)]
    out["RH_Sc"] = rh
    out["gamma"] = gamma

    for wvl, suffix in ch.sca_suffixes.items():
        sc = df[varmap.resolve(df, suffix)]
        needs_drying = rh > flt.dry_ref_rh
        dried = gamma_adjust_scattering(sc, gamma, rh, flt.dry_ref_rh)
        out[f"Sc{wvl}_dry"] = sc.where(~needs_drying, dried)
    # Humidified scattering for the kappa retrieval (gamma-synthesized; the
    # merged dataset has no directly measured high-RH nephelometer channel)
    wet_wvl = str(cfg.channels.wet_wvl_sca[0])
    sc_for_wet = df[varmap.resolve(df, ch.sca_suffixes[wet_wvl])]
    out[f"Sc{wet_wvl}_wet"] = gamma_adjust_scattering(sc_for_wet, gamma, rh, flt.wet_rh)

    for wvl, suffix in ch.abs_suffixes.items():
        out[f"Abs{wvl}"] = df[varmap.resolve(df, suffix)]
    for wvl, suffix in ch.ssa_suffixes.items():
        out[f"SSA{wvl}"] = df[varmap.resolve(df, suffix)]
    out["AE"] = df[varmap.resolve(df, ch.ae_suffix)]
    out["fRH"] = df[varmap.resolve(df, ch.frh_suffix)]
    out["lat"] = df[varmap.resolve(df, ch.lat_suffix)]
    out["lon"] = df[varmap.resolve(df, ch.lon_suffix)]
    out["alt"] = df[varmap.resolve(df, ch.alt_suffix)]
    return out


def cloud_mask(df, cfg):
    """Boolean Series: True where in (or within cloud_pad_s of) cloud.

    A row is cloudy when any available probe exceeds the droplet-number or
    LWC threshold; missing probe data does not flag a row by itself.
    """
    ch, flt = cfg.channels, cfg.filters
    pairs = [(ch.n_cdp_suffix, flt.cloud_n_max_cm3), (ch.lwc_cdp_suffix, flt.cloud_lwc_max_gm3)]
    if flt.use_fcdp:
        pairs += [(ch.n_fcdp_suffix, flt.cloud_n_max_cm3), (ch.lwc_fcdp_suffix, flt.cloud_lwc_max_gm3)]
    cloudy = pd.Series(False, index=df.index)
    for suffix, thresh in pairs:
        col = varmap.resolve(df, suffix, required=False)
        if col is not None:
            cloudy |= df[col] > thresh
    if flt.cloud_pad_s > 0:
        w = 2 * flt.cloud_pad_s + 1
        cloudy = cloudy.rolling(w, center=True, min_periods=1).max().astype(bool)
    return cloudy


def row_qc(df, optical, cfg):
    """Named boolean masks (True = problem) plus the combined ``valid`` mask."""
    ch, flt = cfg.channels, cfg.filters
    masks = pd.DataFrame(index=df.index)
    masks["cloudy"] = cloud_mask(df, cfg)
    if flt.require_inlet_flag_zero:
        inlet = df[varmap.resolve(df, ch.inlet_flag_suffix)]
        masks["inlet_bad"] = (inlet != 0) | inlet.isna()  # unknown inlet = bad
    else:
        masks["inlet_bad"] = False
    # NaN dry scattering also fails: the row is unusable for retrieval
    masks["low_signal"] = ~(optical["Sc450_dry"] > flt.min_dry_sc450_Mm)
    ssa = optical[f"SSA{flt.ssa_filter_wvl}"]
    masks["low_ssa"] = ssa <= flt.min_ssa  # NaN passes (SSA needs Abs > 1 Mm-1)
    masks["valid"] = ~masks[["cloudy", "inlet_bad", "low_signal", "low_ssa"]].any(axis=1)
    return masks
