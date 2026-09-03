"""Propagate instrument + structural uncertainties into retrieved products.

Implements the ensemble-gain design of UNCERTAINTY_MODULE_PLAN.md (v1):

* noise term: the chi2-wmean posterior over the CRI grid (recomputed here
  from the candidate cloud with the same instrument sigmas the retrieval
  used) mapped through local Jacobians of every product w.r.t.
  (RRI, IRI, kappa);
* correlated nuisances: coefficient shifts dy from cheap forward
  evaluations, mapped through the Kalman-style gain
  G = Cov_w(x,y) [Cov_w(y,y) + Sigma]^-1 built from the same candidate
  cloud, plus each nuisance's direct effect on the product at fixed CRI;
  evaluated at +/-1 sigma (secant) and averaged.

v1 simplifications (documented in the group attributes): kappa's response
to the PSD-side nuisances is not propagated (wet/dry ratio largely cancels
them); the nephelometer common-mode calibration cancels in the kappa fit
by construction (the wet target is derived from the same instrument);
nuisances are treated as independent.

Output: one row per window of 1-sigma columns using the same key
conventions as Retr_PSD outputs (m^-1 for coefficients), plus sigmas of
the measured window means and an ``uncertainty_flag``.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import uncertainty_models as um
from .windows import psd_col_name

WVL_UNION = None  # set per-run from cfg

FLAG_RRI_RAIL = 1
FLAG_IRI_RAIL = 2
FLAG_NEAR_GATE = 4
FLAG_LARGE_GF = 8

_SO = None  # sphere_optics module, per process
_SIZING = None  # sizing_correction state (set by _worker_init / run_all)


def _import_sphere_optics(isara_dir):
    global _SO
    if _SO is None:
        if isara_dir not in sys.path:
            sys.path.insert(0, isara_dir)
        import sphere_optics  # noqa: PLC0415
        _SO = sphere_optics
    return _SO


def _coeffs(dpg_um, dnd_cm3, rri, iri, wvls):
    """{wvl: (sca, abs, ext, ssa)} in Mm^-1 for one CRI (finite bins only)."""
    r = _SO.Model(np.asarray(wvls), size_equ={'m': 'cs'},
                  dndlogdp={'m': dnd_cm3 * 1e6}, dpg={'m': dpg_um},
                  RRI={'m': rri}, IRI={'m': iri}, nonabs_fraction={'m': 0},
                  shape={'m': 'sphere'}, density={'m': 1.0}, RH=0, kappa=0,
                  num_theta=2)
    out = {}
    for wv in wvls:
        ext = r[f'ext_coeff_{wv}_m-1'] * 1e6
        ssa = r[f'ssa_{wv}']
        out[wv] = (ssa * ext, (1 - ssa) * ext, ext, ssa)
    return out


def cov_parts(dpg, dnd_weighted, raw_dnd, pen_params, sca_meas, abs_meas,
              sca_wvls, abs_wvls, wvls, window_s, regime,
              ref_cri=(1.52, 0.005), lnd_sigma=None, n_scale_sigma=None):
    """Full observation+model covariance S [(Mm^-1)^2], channel order
    [sca..., abs...]. Measurement part: per-channel white+floor diagonals
    plus rank-1 common-mode calibration terms (neph f_rel across the sca
    block; the PSAP 0.016*b_sp scattering-subtraction across the abs
    block). Model part: rank-1 outer products of the secant coefficient
    shifts of each structural nuisance (PSD lnD scale, PSD concentration
    scale, impactor D50/steepness/density) evaluated at a reference CRI —
    residual patterns along these directions are thereby marginalized
    over in the chi^2 rather than rejected."""
    n_s, n_a = len(sca_wvls), len(abs_wvls)
    rri0, iri0 = ref_cri
    t = window_s

    def yvec(dpg_n, dnd_n):
        c = _coeffs(dpg_n, dnd_n, rri0, iri0, wvls)
        return np.array([c[w][0] for w in sca_wvls]
                        + [c[w][1] for w in abs_wvls])

    y0 = yvec(dpg, dnd_weighted)
    y_sca_meas = np.array([sca_meas[w] for w in sca_wvls])
    b_sp_near = np.array([sca_meas[min(sca_meas, key=lambda ws: abs(ws - wv))]
                          for wv in abs_wvls])
    abs_meas_v = np.array([abs_meas[w] for w in abs_wvls])

    S = np.zeros((n_s + n_a, n_s + n_a))
    # measurement. The UM f_rel values are MARGINAL per-channel sigmas whose
    # cross-channel correlation is unquantified (one gas calibration is
    # common; per-wavelength truncation corrections are not). Split the
    # variance evenly: half independent (diagonal), half common (rank-1),
    # which preserves each channel's marginal sigma exactly.
    half = 1.0 / np.sqrt(2.0)
    for i, wv in enumerate(sca_wvls):
        a = um.NEPH_A[int(wv)]
        S[i, i] = (a ** 2 * (um.NEPH_T_REF / t)
                   + (a * np.sqrt(um.NEPH_T_REF / um.NEPH_ZERO_DUR)) ** 2
                   + (half * um.NEPH_FREL[regime] * y_sca_meas[i]) ** 2)
    for j in range(n_a):
        t_eff = max(t, um.PSAP_T_INTERNAL)
        S[n_s + j, n_s + j] = (um.PSAP_A ** 2 * (um.PSAP_T_REF / t_eff)
                               + um.PSAP_FLOOR ** 2
                               + (um.PSAP_FREL * abs_meas_v[j]) ** 2
                               + (half * um.PSAP_FSCA_ERR * b_sp_near[j]) ** 2)
    v = np.r_[half * um.NEPH_FREL[regime] * y_sca_meas, np.zeros(n_a)]
    S += np.outer(v, v)
    v = np.r_[np.zeros(n_s), half * um.PSAP_FSCA_ERR * b_sp_near]
    S += np.outer(v, v)
    # model nuisances (secant dy at the reference CRI)
    sD = np.exp(um.OPC_DLND if lnd_sigma is None else lnd_sigma)
    sN = um.OPC_DN_SCALE if n_scale_sigma is None else n_scale_sigma
    dys = [(yvec(dpg * sD, dnd_weighted) - yvec(dpg / sD, dnd_weighted)) / 2,
           sN * y0]
    d50, gsd, rho = pen_params
    if d50 > 0:
        def pen_of(d50_, gsd_, rho_):
            sexp = np.log(5.25) / np.log(gsd_)
            return 1.0 / (1.0 + ((dpg * np.sqrt(rho_)) / d50_) ** sexp)
        for hi, lo in [((d50 * 1.1, gsd, rho), (d50 * 0.9, gsd, rho)),
                       ((d50, gsd * 1.09, rho),
                        (d50, max(gsd / 1.09, 1.01), rho)),
                       ((d50, gsd, rho + 0.2), (d50, gsd, rho - 0.2))]:
            dys.append((yvec(dpg, raw_dnd * pen_of(*hi))
                        - yvec(dpg, raw_dnd * pen_of(*lo))) / 2)
    return S, np.array(dys)


def build_obs_cov(*args, **kwargs):
    """Total covariance S = Sigma_meas + sum_k dy_k dy_k' (see cov_parts)."""
    S, D = cov_parts(*args, **kwargs)
    return S + D.T @ D


def _grown(dpg, rri, iri, kappa, rh):
    gf = (1 + kappa * rh / (100 - rh)) ** (1 / 3)
    rri_w = (rri + (gf ** 3 - 1) * 1.33) / gf ** 3
    iri_w = iri / gf ** 3
    return dpg * gf, rri_w, iri_w, gf


def _products(dpg, dnd, rri, iri, kappa, rh_wet, rh_amb, wvls):
    """Flat dict of every product, keyed like Retr_PSD outputs (coeffs Mm^-1
    here; converted to m^-1 at assembly)."""
    P = {"dry_RRI_unitless": rri, "dry_IRI_unitless": iri}
    states = [("dry", None)]
    if np.isfinite(kappa):
        P["kappa_unitless"] = kappa
        states.append(("wet", rh_wet))
        if rh_amb is not None and np.isfinite(rh_amb) and 0 < rh_amb < 100:
            states.append(("amb", rh_amb))
    for st, rh in states:
        if st == "dry":
            d, rr, ii = dpg, rri, iri
        else:
            d, rr, ii, gf = _grown(dpg, rri, iri, kappa, rh)
            P[f"{st}_gf_unitless"] = gf
            P[f"{st}_RRI_unitless"] = rr
            P[f"{st}_IRI_unitless"] = ii
        c = _coeffs(d, dnd, rr, ii, wvls)
        for wv, (sca, ab, ext, ssa) in c.items():
            P[f"{st}_cal_sca_coef_{wv}_m-1"] = sca
            P[f"{st}_cal_abs_coef_{wv}_m-1"] = ab
            P[f"{st}_cal_ext_coef_{wv}_m-1"] = ext
            P[f"{st}_cal_SSA_{wv}_unitless"] = ssa
        P[f"{st}_AE_unitless"] = (np.log(c[450][0] / c[700][0])
                                  / np.log(700 / 450))
    return P


def _pvec(P, keys):
    return np.array([P.get(k, np.nan) for k in keys])


def _window_uncertainty(item):
    """Compute all 1-sigma columns for one window. Never raises."""
    ts = item["ts"]
    try:
        return ts, _window_uncertainty_inner(item)
    except Exception as err:  # noqa: BLE001 - per-window robustness
        return ts, {"uncertainty_error": str(err)}


def _window_uncertainty_inner(it):
    _import_sphere_optics(it["isara_dir"])
    wvls = it["wvls"]
    dpg_all = it["dpg"]
    raw = it["dnd_raw"]
    pen = it["pen"]
    fin = np.isfinite(raw) & (dpg_all > 0)
    out = {}

    # ---- sigmas of the measured means (always available) -------------------
    t = it["window_s"]
    for wv, v in it["sca_meas"].items():
        out[f"Sc{wv}_dry_sigma"] = float(um.sigma_scattering(
            v, t, wv, it["regime"]))
    for wv, v in it["abs_meas"].items():
        near = min(it["sca_meas"], key=lambda ws: abs(ws - wv))
        out[f"Abs{wv}_sigma"] = float(um.sigma_absorption(
            v, it["sca_meas"][near], t))
    # ratio-based: calibration cancels in the synthesized wet/dry pair
    a_w = um.NEPH_A[int(it["wet_wvl"])]
    out["Sc_wet_sigma"] = float(np.sqrt(
        (0.01 * it["wet_meas"]) ** 2 + a_w ** 2 * (um.NEPH_T_REF / t)
        + (a_w * np.sqrt(um.NEPH_T_REF / um.NEPH_ZERO_DUR)) ** 2))
    n_eff = max(float(it["n_valid"]), 1.0)
    sig_bins = um.sigma_number(np.where(fin, raw, np.nan), n_eff,
                               Q=1.0, edge_bins=2)
    for d, sg in zip(dpg_all, sig_bins):
        out[f"psd_sigma_{psd_col_name(d)}"] = float(sg)

    if not np.isfinite(it["rri"]) or fin.sum() < 10:
        return out

    dpg = dpg_all[fin]
    dnd = (raw * pen)[fin]
    rri, iri, kappa = it["rri"], it["iri"], it["kappa"]
    cols_fin = np.where(fin)[0]
    lnd_sigma = it.get("lnd_sigma") or um.OPC_DLND
    n_scale = it.get("n_scale_sigma") or um.OPC_DN_SCALE
    rh_wet, rh_amb = it["rh_wet"], it["rh_amb"]

    # ---- candidate cloud, posterior, gain ----------------------------------
    grid_cri = it["cri_grid"]
    sca_w = it["sca_wvls"]
    abs_w = it["abs_wvls"]
    y = np.empty((len(grid_cri), len(sca_w) + len(abs_w)))
    from . import sizing_correction as szc  # noqa: PLC0415
    for k, (rr, ii) in enumerate(grid_cri):
        if _SIZING is not None:
            dpg_k, dnd_k = szc.apply(_SIZING, k, dpg, dnd, cols=cols_fin)
        else:
            dpg_k, dnd_k = dpg, dnd
        c = _coeffs(dpg_k, dnd_k, rr, ii, wvls)
        y[k] = [c[w][0] for w in sca_w] + [c[w][1] for w in abs_w]
    if _SIZING is not None:
        # base/products/nuisances evaluated at the reported CRI's correction
        kn = szc.nearest_candidate(_SIZING, rri, iri)
        dpg, dnd = szc.apply(_SIZING, kn, dpg, dnd, cols=cols_fin)
        raw_fin_corr = szc.apply(_SIZING, kn, dpg_all[fin], raw[fin],
                                 cols=cols_fin)[1]
    else:
        raw_fin_corr = raw[fin]
    y_meas = np.array([it["sca_meas"][w] for w in sca_w]
                      + [it["abs_meas"][w] for w in abs_w])
    sig = np.array([out[f"Sc{w}_dry_sigma"] for w in sca_w]
                   + [out[f"Abs{w}_sigma"] for w in abs_w])
    marg = it["marginalized"]
    if marg:
        S_meas, D = cov_parts(dpg, dnd, raw_fin_corr,
                              (it["d50"], it["gsd"], it["rho"]),
                              it["sca_meas"], it["abs_meas"], sca_w, abs_w,
                              wvls, it["window_s"], it["regime"],
                              lnd_sigma=lnd_sigma, n_scale_sigma=n_scale)
        S = S_meas + D.T @ D
        S_inv = np.linalg.inv(S)
        r = y - y_meas
        chi2 = np.einsum("ki,ij,kj->k", r, S_inv, r)
    else:
        chi2 = (((y - y_meas) / sig) ** 2).sum(axis=1)
    w = np.exp(-0.5 * (chi2 - chi2.min()))
    w /= w.sum()
    x = grid_cri
    xbar = w @ x
    dxc = x - xbar
    dyc = y - w @ y
    cov_x = (w[:, None] * dxc).T @ dxc                      # 2x2
    cov_xy = (w[:, None] * dxc).T @ dyc                     # 2x6
    cov_yy = (w[:, None] * dyc).T @ dyc                     # 6x6
    S_gain = S if marg else np.diag(sig ** 2)
    G = cov_xy @ np.linalg.inv(cov_yy + S_gain)              # 2x6

    # ---- product Jacobians -------------------------------------------------
    base = _products(dpg, dnd, rri, iri, kappa, rh_wet, rh_amb, wvls)
    keys = sorted(base)
    p0 = _pvec(base, keys)
    dR, dI, dK = 0.01, 0.002, 0.02
    sR = -dR if rri + dR > it["rri_max"] else dR
    sI = -dI if iri + dI > 0.030 else dI
    Jr = (_pvec(_products(dpg, dnd, rri + sR, iri, kappa, rh_wet, rh_amb,
                          wvls), keys) - p0) / sR
    Ji = (_pvec(_products(dpg, dnd, rri, iri + sI, kappa, rh_wet, rh_amb,
                          wvls), keys) - p0) / sI
    Jx = np.stack([Jr, Ji])                                  # 2 x P
    var_noise = np.einsum("ip,ij,jp->p", Jx, cov_x, Jx)
    if np.isfinite(kappa):
        Jk = (_pvec(_products(dpg, dnd, rri, iri, kappa + dK, rh_wet, rh_amb,
                              wvls), keys) - p0) / dK
        var_noise = var_noise + (Jk * it["kappa_std"]) ** 2

    # ---- nuisances ---------------------------------------------------------
    def y_of(dpg_n, dnd_n):
        c = _coeffs(dpg_n, dnd_n, rri, iri, wvls)
        return np.array([c[w][0] for w in sca_w] + [c[w][1] for w in abs_w])

    y0 = y_of(dpg, dnd)
    var_nuis = np.zeros_like(p0)

    def direct_secant(dpg_p, dnd_p, dpg_m, dnd_m):
        """Signed per-1-sigma direct product shift of one nuisance."""
        Pp = _pvec(_products(dpg_p, dnd_p, rri, iri, kappa, rh_wet, rh_amb,
                             wvls), keys)
        Pm = _pvec(_products(dpg_m, dnd_m, rri, iri, kappa, rh_wet, rh_amb,
                             wvls), keys)
        return (Pp - Pm) / 2.0, Pp, Pm

    # nuisance perturbation pairs, ORDER MATCHING cov_parts' D rows
    sDn = np.exp(lnd_sigma)
    pairs = [((dpg * sDn, dnd), (dpg / sDn, dnd)),
             ((dpg, dnd * (1.0 + n_scale)), (dpg, dnd * (1.0 - n_scale)))]
    if it["d50"] > 0:
        rawf = raw_fin_corr

        def pen_of(d50, gsd, rho):
            sexp = np.log(5.25) / np.log(gsd)
            return 1.0 / (1.0 + ((dpg * np.sqrt(rho)) / d50) ** sexp)

        b = (it["d50"], it["gsd"], it["rho"])
        for hi, lo in [((b[0] * 1.1, b[1], b[2]), (b[0] * 0.9, b[1], b[2])),
                       ((b[0], b[1] * 1.09, b[2]),
                        (b[0], max(b[1] / 1.09, 1.01), b[2])),
                       ((b[0], b[1], b[2] + 0.2), (b[0], b[1], b[2] - 0.2))]:
            pairs.append(((dpg, rawf * pen_of(*hi)), (dpg, rawf * pen_of(*lo))))

    if marg:
        # V9 joint-posterior accounting: condition the nuisance amplitudes on
        # the residual at the reported CRI. theta_k is in 1-sigma units; the
        # data constrain the components whose coefficient signatures (rows of
        # D) are observable, so directly-measured products collapse toward
        # measurement precision while unobserved directions stay at prior
        # width. Products are REPORTED at theta=0, so the posterior second
        # moment (Sigma_post + theta_hat theta_hat') is used, counting the
        # known-but-uncorrected shift as uncertainty.
        dPmat = np.stack([direct_secant(*p, *m)[0] for p, m in pairs])  # K x P
        K = D.shape[0]
        M = np.linalg.inv(S + cov_yy)   # S = S_meas + D'D (6x6 data space)
        r_hat = y_meas - y_of(dpg, dnd)
        theta_hat = D @ M @ r_hat
        Sig_theta = np.eye(K) - D @ M @ D.T
        E2 = Sig_theta + np.outer(theta_hat, theta_hat)
        var_nuis = var_nuis + np.einsum("kp,kl,lp->p", dPmat, E2, dPmat)
        out["sizing_lnD_shift_unitless"] = float(theta_hat[0] * lnd_sigma)
        ## MAP-fit PSD diagnostics: the nuisance-adjusted forward state at the
        ## reported CRI. theta_hat is in 1-sigma units; row 1 of D is the
        ## concentration-scale pattern (n_scale * y0), so the implied
        ## multiplicative PSD factor is 1 + theta_hat[1]*n_scale. y_fit is the
        ## first-order forward state at the MAP nuisances (all rows: lnD,
        ## N-scale, impactor parameters) -- "the scattering the retrieval
        ## actually attributes to the aerosol" given the PSD priors.
        out["psd_scale_factor_unitless"] = float(1.0 + theta_hat[1] * n_scale)
        y_fit = y0 + D.T @ theta_hat
        for i2, wv in enumerate(sca_w):
            out[f"Sc{wv}_dry_fit"] = float(y_fit[i2])
        for j2, wv in enumerate(abs_w):
            out[f"Abs{wv}_fit"] = float(y_fit[len(sca_w) + j2])
    else:
        for (pp, pm) in pairs:
            tot = np.zeros_like(p0)
            for dpg_n, dnd_n in (pp, pm):
                direct = _pvec(_products(dpg_n, dnd_n, rri, iri, kappa, rh_wet,
                                         rh_amb, wvls), keys) - p0
                dy = y_of(dpg_n, dnd_n) - y0
                tot += np.abs(direct + Jx.T @ (-G @ dy))
            var_nuis = var_nuis + (tot / 2.0) ** 2

    # data-side common modes: in marginalized mode these live inside S
    # (posterior already carries them); V7 mode adds them via the gain
    if not marg:
        n_s, n_a = len(sca_w), len(abs_w)
        for dy_meas in (np.r_[um.NEPH_FREL[it["regime"]] * y_meas[:n_s],
                              np.zeros(n_a)],
                        np.r_[np.zeros(n_s), um.PSAP_FSCA_ERR
                              * np.array([it["sca_meas"][min(it["sca_meas"],
                                          key=lambda ws: abs(ws - wv))]
                                          for wv in abs_w])]):
            dx = G @ dy_meas
            var_nuis += (Jx.T @ dx) ** 2

    sigma = np.sqrt(var_noise + var_nuis)
    for k, sg in zip(keys, sigma):
        if k.endswith("_m-1"):
            sg = sg * 1e-6   # store coefficient sigmas in m^-1 like Retr_PSD
        out[k] = float(sg)
    # kappa sigma: posterior only (v1; see module docstring)
    if np.isfinite(kappa):
        out["kappa_unitless"] = float(it["kappa_std"])

    flag = 0
    if rri > it["rri_max"] - 0.015 or rri < it["rri_min"] + 0.015:
        flag |= FLAG_RRI_RAIL
    if iri > 0.028:
        flag |= FLAG_IRI_RAIL
    if it["min_chi2"] > 0.8:
        flag |= FLAG_NEAR_GATE
    if np.isfinite(kappa) and base.get("amb_gf_unitless", 1.0) > 1.5:
        flag |= FLAG_LARGE_GF
    out["uncertainty_flag"] = flag
    return out


def _worker_init(isara_dir, sizing_state=None):
    global _SIZING
    _import_sphere_optics(isara_dir)
    _SIZING = sizing_state


def run_all(results_df, grid, cfg, progress=True):
    """Uncertainty columns for every window in ``results_df`` (the assembled
    windows+retrievals frame). Returns a DataFrame indexed like it."""
    from . import isara_bridge  # noqa: PLC0415 (grid helpers)

    ch = cfg.channels
    wvls = sorted({*ch.dry_wvl_sca, *ch.dry_wvl_abs, *ch.wet_wvl_sca,
                   *(ch.val_wvl or [])})
    regime = cfg.isara.neph_regime or (
        "pm1" if cfg.psd.impactor_d50_aero_um > 0 else "pm10")
    cri_grid = isara_bridge._cri_grid(cfg)
    pen = (grid.penetration if grid.penetration is not None
           else np.ones(len(grid)))
    sizing_state = isara_bridge.build_sizing_state(grid, cfg)
    lnd_sigma = (cfg.isara.sizing_residual_lnd if sizing_state is not None
                 else um.OPC_DLND)

    items = []
    for ts, row in results_df.iterrows():
        items.append({
            "ts": ts,
            "isara_dir": cfg.paths.isara_code_dir,
            "wvls": wvls,
            "sca_wvls": list(ch.dry_wvl_sca),
            "abs_wvls": list(ch.dry_wvl_abs),
            "dpg": grid.dpg_um,
            "pen": pen,
            "dnd_raw": np.array([row.get(psd_col_name(d), np.nan)
                                 for d in grid.dpg_um], float),
            "sca_meas": {wv: float(row.get(f"Sc{wv}_dry_mean", np.nan))
                         for wv in ch.dry_wvl_sca},
            "abs_meas": {wv: float(row.get(f"Abs{wv}_mean", np.nan))
                         for wv in ch.dry_wvl_abs},
            "wet_meas": float(row.get(f"Sc{ch.wet_wvl_sca[0]}_wet_mean", np.nan)),
            "wet_wvl": ch.wet_wvl_sca[0],
            "n_valid": float(row.get("n_valid", np.nan)),
            "window_s": float(cfg.window.window_s),
            "regime": regime,
            "cri_grid": cri_grid,
            "rri": float(row.get("dry_RRI_unitless", np.nan)),
            "iri": float(row.get("dry_IRI_unitless", np.nan)),
            "kappa": float(row.get("kappa_unitless", np.nan)),
            "kappa_std": float(row.get("kappa_std_unitless", np.nan)),
            "min_chi2": float(row.get("dry_CRI_min_chi2_unitless", np.nan)),
            "rh_wet": float(cfg.filters.wet_rh),
            "rh_amb": float(row.get("RH_amb_mean", np.nan)),
            "rri_min": cfg.isara.rri_min,
            "rri_max": cfg.isara.rri_max,
            "marginalized": cfg.isara.chi2_sigma == "instrument-cov",
            "lnd_sigma": lnd_sigma,
            "n_scale_sigma": cfg.isara.n_scale_sigma,
            "d50": cfg.psd.impactor_d50_aero_um,
            "gsd": cfg.psd.impactor_gsd,
            "rho": cfg.psd.impactor_rho_gcm3,
        })

    results = {}
    n_workers = cfg.isara.n_workers
    if n_workers <= 1:
        global _SIZING
        _import_sphere_optics(cfg.paths.isara_code_dir)
        _SIZING = sizing_state
        for i, item in enumerate(items):
            ts, res = _window_uncertainty(item)
            results[ts] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_worker_init,
                                 initargs=(cfg.paths.isara_code_dir,
                                           sizing_state)) as pool:
            for i, (ts, res) in enumerate(
                    pool.map(_window_uncertainty, items, chunksize=8)):
                results[ts] = res
                if progress and (i + 1) % 500 == 0:
                    print(f"  uncertainty {i + 1}/{len(items)} windows",
                          flush=True)

    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "time"
    return out.reindex(results_df.index)
