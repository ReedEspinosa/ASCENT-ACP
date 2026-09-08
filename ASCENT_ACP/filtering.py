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
    ``Sc550_amb``/``RH_amb`` (at ambient RH when available and below
    ambient_rh_max), ``Abs{wvl}``, ``RH_Sc``, ``gamma``, ``AE``, ``SSA``,
    ``lat/lon/alt``. Done per 1 Hz row, before any averaging, so intra-window
    RH variability is handled exactly.
    """
    ch, flt = cfg.channels, cfg.filters
    out = pd.DataFrame(index=df.index)
    if ch.rh_sc_suffix:
        rh = df[varmap.resolve(df, ch.rh_sc_suffix)]
    else:
        # campaign archives no nephelometer sample RH; assume the configured
        # constant (documented caveat: the wet/ambient gamma synthesis then
        # carries the RH-assumption error)
        rh = pd.Series(ch.rh_sc_assumed_percent, index=df.index)
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
    # Ambient-RH state: the configured primary RH column, filled where
    # missing from the fallback chain (direct RH columns in priority order,
    # then RH derived from an H2O mixing ratio + static T/P). Rows above
    # ambient_rh_max (or with no source at all) get NaN, never a capped value.
    rh_amb_raw = ambient_rh_chain(df, ch)
    if rh_amb_raw is not None:
        rh_amb = rh_amb_raw.where(
            (rh_amb_raw > 0) & (rh_amb_raw <= flt.ambient_rh_max))
        out["RH_amb"] = rh_amb
        out[f"Sc{wet_wvl}_amb"] = gamma_adjust_scattering(sc_for_wet, gamma, rh, rh_amb)

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


def ambient_rh_chain(df, ch):
    """Ambient RH (%) coalesced over the configured source chain, or None.

    Sources, in priority order: ``rh_ambient_suffix``, each entry of
    ``rh_ambient_fallback_suffixes``, then RH derived from the H2O mixing
    ratio (ppmv) with static temperature (degC) and pressure (hPa) via
    e = ppmv*1e-6*P and the Alduchov & Eskridge (1996) saturation vapor
    pressure over liquid water. Missing columns are skipped silently.
    """
    rh = None
    for sfx in [ch.rh_ambient_suffix, *ch.rh_ambient_fallback_suffixes]:
        if not sfx:
            continue
        col = varmap.resolve(df, sfx, required=False)
        if col is None:
            continue
        rh = df[col] if rh is None else rh.fillna(df[col])
    if ch.rh_ambient_h2o_ppmv_suffix:
        cols = [varmap.resolve(df, s, required=False)
                for s in (ch.rh_ambient_h2o_ppmv_suffix,
                          ch.rh_ambient_temp_c_suffix,
                          ch.rh_ambient_press_hpa_suffix)]
        if all(c is not None for c in cols):
            e_hpa = df[cols[0]] * 1e-6 * df[cols[2]]
            t_c = df[cols[1]]
            es_hpa = 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))
            derived = 100.0 * e_hpa / es_hpa
            rh = derived if rh is None else rh.fillna(derived)
    return rh


def ambient_rh_sources(ch):
    """Human-readable list of the configured ambient-RH sources (provenance)."""
    srcs = [s for s in [ch.rh_ambient_suffix, *ch.rh_ambient_fallback_suffixes]
            if s]
    if ch.rh_ambient_h2o_ppmv_suffix:
        srcs.append(f"derived from {ch.rh_ambient_h2o_ppmv_suffix} with "
                    f"{ch.rh_ambient_temp_c_suffix}/"
                    f"{ch.rh_ambient_press_hpa_suffix}")
    return srcs


def cloud_mask(df, cfg):
    """Boolean Series: True where in (or within cloud_pad_s of) cloud.

    A row is cloudy when any available probe exceeds the droplet-number or
    LWC threshold; missing probe data does not flag a row by itself.
    """
    ch, flt = cfg.channels, cfg.filters
    # FCDP values are scaled into the CDP's units (#/cm3, g/m3) before the
    # shared thresholds are applied; the FCDP ICARTT units are #/m^3, kg/m^3.
    pairs = [(ch.n_cdp_suffix, flt.cloud_n_max_cm3, 1.0),
             (ch.lwc_cdp_suffix, flt.cloud_lwc_max_gm3, 1.0)]
    if flt.use_fcdp:
        pairs += [(ch.n_fcdp_suffix, flt.cloud_n_max_cm3, flt.fcdp_n_scale_to_cm3),
                  (ch.lwc_fcdp_suffix, flt.cloud_lwc_max_gm3, flt.fcdp_lwc_scale_to_gm3)]
    cloudy = pd.Series(False, index=df.index)
    for suffix, thresh, scale in pairs:
        col = varmap.resolve(df, suffix, required=False)
        if col is not None:
            cloudy |= df[col] * scale > thresh
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
