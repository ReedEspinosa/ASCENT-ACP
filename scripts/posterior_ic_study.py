#!/usr/bin/env python
"""Information-content study of the marginalized chi^2 posterior.

For a spread sample of good windows, re-run the CRI retrieval with the
observation covariance S assembled from different subsets of the model-
nuisance outer products, to quantify which term is responsible for
flattening the dry-RRI posterior:

  prod      S_meas + lnD + Nscale + impactor   (production V8+)
  no_lnD    drop the PSD lnD-scale term
  no_N      drop the 10% concentration-scale term
  no_lnD_N  drop both
  meas      S_meas only (instrument terms incl. common modes)
  emp       lnD term rescaled to the empirical theta-hat scatter sigma

Also computes, per window, the whitened-space alignment cos(dy_k, dy_RRI)
between each nuisance direction and the RRI signal direction, and the
RRI Fisher-information retention g'S_full^-1 g / g'S_meas^-1 g.

Usage: python scripts/posterior_ic_study.py CONFIG N_SAMPLE EMP_SIGMA OUT_PKL
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ASCENT_ACP import filtering, isara_bridge, pipeline, sizebins, windows
from ASCENT_ACP import sizing_correction as szc
from ASCENT_ACP import uncertainty_models as um
from ASCENT_ACP import uncertainty_propagation as up
from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP.windows import psd_col_name

VARIANTS = ["prod", "no_lnD", "no_N", "no_lnD_N", "meas", "emp"]


def cov_variants(row, dnd_weighted, grid, cfg, sizing_state, emp_sigma):
    """Variant covariances (m^-2) + alignment diagnostics, or None."""
    ch = cfg.channels
    fin = np.isfinite(dnd_weighted) & (grid.dpg_um > 0)
    if fin.sum() < 10:
        return None
    sca_meas = {w: float(row[f"Sc{w}_dry_mean"]) for w in ch.dry_wvl_sca}
    abs_meas = {w: float(row[f"Abs{w}_mean"]) for w in ch.dry_wvl_abs}
    if not (np.isfinite(list(sca_meas.values())).all()
            and np.isfinite(list(abs_meas.values())).all()):
        return None
    up._import_sphere_optics(cfg.paths.isara_code_dir)
    raw = np.array([row[psd_col_name(d)] for d in grid.dpg_um], float)
    regime = cfg.isara.neph_regime or (
        "pm1" if cfg.psd.impactor_d50_aero_um > 0 else "pm10")
    wvls = sorted({*ch.dry_wvl_sca, *ch.dry_wvl_abs})
    dpg_f, dnd_f, raw_f = grid.dpg_um[fin], dnd_weighted[fin], raw[fin]
    lnd_sigma = um.OPC_DLND
    if sizing_state is not None:
        k = szc.nearest_candidate(sizing_state, 1.52, 0.005)
        cols = np.where(fin)[0]
        dpg_f, dnd_f = szc.apply(sizing_state, k, dpg_f, dnd_f, cols=cols)
        _, raw_f = szc.apply(sizing_state, k, grid.dpg_um[fin], raw[fin],
                             cols=cols)
        lnd_sigma = cfg.isara.sizing_residual_lnd
    pen = (cfg.psd.impactor_d50_aero_um, cfg.psd.impactor_gsd,
           cfg.psd.impactor_rho_gcm3)
    S_meas, D = up.cov_parts(dpg_f, dnd_f, raw_f, pen, sca_meas, abs_meas,
                             list(ch.dry_wvl_sca), list(ch.dry_wvl_abs),
                             wvls, float(cfg.window.window_s), regime,
                             lnd_sigma=lnd_sigma)
    # RRI signal direction: secant of the forward coefficients over +-0.02
    def yvec(rri):
        c = up._coeffs(dpg_f, dnd_f, rri, 0.005, wvls)
        return np.array([c[w][0] for w in ch.dry_wvl_sca]
                        + [c[w][1] for w in ch.dry_wvl_abs])
    g = (yvec(1.54) - yvec(1.50)) / 2.0     # per 0.02 RRI

    L = np.linalg.cholesky(S_meas)
    zg = np.linalg.solve(L, g)
    align = {}
    for i, nm in enumerate(["lnD", "Nscale", "imp_d50", "imp_gsd",
                            "imp_rho"][: len(D)]):
        z = np.linalg.solve(L, D[i])
        align[f"cos_{nm}"] = float(z @ zg / (np.linalg.norm(z)
                                             * np.linalg.norm(zg) + 1e-300))
        align[f"amp_{nm}"] = float(np.linalg.norm(z))   # info removed, sigmas
    S_full = S_meas + D.T @ D
    info_full = float(zg @ zg) and float(
        g @ np.linalg.solve(S_full, g)) / float(g @ np.linalg.solve(S_meas, g))
    align["rri_info_retained"] = info_full
    align["snr_rri_meas"] = float(np.linalg.norm(zg))   # sigmas per 0.02 RRI

    D_emp = D.copy()
    D_emp[0] *= emp_sigma / lnd_sigma
    covs = {
        "prod": S_full,
        "no_lnD": S_meas + D[1:].T @ D[1:],
        "no_N": S_meas + np.delete(D, 1, 0).T @ np.delete(D, 1, 0),
        "no_lnD_N": S_meas + D[2:].T @ D[2:],
        "meas": S_meas,
        "emp": S_meas + D_emp.T @ D_emp,
    }
    return {k: v * 1e-12 for k, v in covs.items()}, align


def _run_window(item):
    ts, kwargs, covs = item
    out = {"ts": ts}
    for name in VARIANTS:
        kw = dict(kwargs)
        kw["obs_cov"] = covs[name]
        kw["wet_sca_coef"] = np.full_like(np.asarray(kw["wet_sca_coef"],
                                                     float), np.nan)
        try:
            r = isara_bridge._ISARA.Retr_PSD(
                **kw, lut=None, sizing_corr=isara_bridge._SIZING)
            for k_src, k_dst in [("dry_RRI_unitless", "rri"),
                                 ("dry_IRI_unitless", "iri"),
                                 ("dry_CRI_min_chi2_unitless", "chi2"),
                                 ("dry_CRI_n_accepted_unitless", "nacc"),
                                 ("dry_RRI_accepted_std_unitless", "rri_std"),
                                 ("attempt_flag_CRI_unitless", "flag")]:
                out[f"{name}_{k_dst}"] = float(np.atleast_1d(
                    r.get(k_src, np.nan))[0])
        except Exception as err:  # noqa: BLE001
            out[f"{name}_err"] = str(err)
    return out


def main(cfg_path, n_sample, emp_sigma, out_pkl):
    cfg = PipelineConfig.from_json(cfg_path)
    df, _ = pipeline.load_inputs(cfg)
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    good = wdf[wdf["window_qc_flag"] == 0]
    sel = good.iloc[np.unique(np.linspace(0, len(good) - 1,
                                          n_sample).astype(int))]
    print(f"{len(sel)} sample windows of {len(good)} good")

    isara_bridge.import_isara(cfg.paths.isara_code_dir)
    sizing_state = isara_bridge.build_sizing_state(grid, cfg)
    isara_bridge._SIZING = sizing_state

    items, aligns = [], []
    for ts, row in sel.iterrows():
        kwargs = isara_bridge.build_retr_kwargs(row, grid, cfg)
        dnd = np.array([row[psd_col_name(d)] for d in grid.dpg_um], float)
        if grid.penetration is not None:
            dnd = dnd * grid.penetration
        cv = cov_variants(row, dnd, grid, cfg, sizing_state, emp_sigma)
        if cv is None:
            continue
        covs, align = cv
        align["ts"] = ts
        aligns.append(align)
        kwargs.pop("obs_cov", None)
        items.append((ts, kwargs, covs))
    print(f"{len(items)} windows with covariance")

    scratch = cfg.paths.scratch_dir if hasattr(cfg.paths, "scratch_dir") \
        else "/tmp/isara_scratch"
    with ProcessPoolExecutor(
            max_workers=8, initializer=isara_bridge._worker_init,
            initargs=(cfg.paths.isara_code_dir, scratch, {},
                      sizing_state)) as ex:
        rows = list(ex.map(_run_window, items, chunksize=4))
    res = pd.DataFrame(rows).set_index("ts")
    al = pd.DataFrame(aligns).set_index("ts")

    print("\n--- alignment of nuisance directions with the RRI signal ---")
    for c in al.columns:
        v = al[c].astype(float)
        print(f"  {c:22s} med={v.median():+.3f}  "
              f"IQR=[{v.quantile(.25):+.3f},{v.quantile(.75):+.3f}]")

    print("\n--- retrieval variants ---")
    for name in VARIANTS:
        f = res.get(f"{name}_flag")
        ok = res[f == 2] if f is not None else res.iloc[0:0]
        if len(ok) == 0:
            print(f"  {name:9s} no successes")
            continue
        rri = ok[f"{name}_rri"]
        print(f"  {name:9s} n_ok={len(ok):4d}  RRI med={rri.median():.4f} "
              f"std={rri.std():.4f}  postSTD med="
              f"{ok[f'{name}_rri_std'].median():.4f}  "
              f"minchi2 med={ok[f'{name}_chi2'].median():.3f}  "
              f"n_acc med={ok[f'{name}_nacc'].median():.0f}")

    with open(out_pkl, "wb") as fh:
        pickle.dump({"results": res, "align": al,
                     "config": cfg_path, "emp_sigma": emp_sigma}, fh)
    print("wrote", out_pkl)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
