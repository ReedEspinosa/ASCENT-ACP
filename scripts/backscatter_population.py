#!/usr/bin/env python
"""Population check: does the b sizing-immunity at 550/700 survive across
real PSD variability?

The single-window fingerprint (backscatter_fingerprint.py) can mislead: by
size-parameter scaling, a PSD only ~20% coarser moves the blue-channel sizing
response into the green/red channels, and PSD SHAPE (width, coarse tail)
matters beyond Deff. This script loops ~200 windows spanning the
effective-diameter distribution and, per window, computes db/b (TSI-truncated
geometry) for sizing +5% lnD and RRI +0.04, plus a per-window linear-Fisher
sigma_RRI for the joint [Sc450/550/700, b450/550/700] system with all
nuisances (sizing, concentration, calibration, b-common).

Usage: python scripts/backscatter_population.py [CONFIG]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from backscatter_fingerprint import load_curves, coeffs, WVL_NM
from ASCENT_ACP import filtering, isara_bridge, pipeline, sizebins, windows
from ASCENT_ACP import sizing_correction as szc
from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP.windows import psd_col_name


def fisher(g, nus, noise):
    S = np.diag(np.asarray(noise) ** 2)
    for dvec in nus:
        S = S + np.outer(dvec, dvec)
    return 1.0 / np.sqrt(g @ np.linalg.solve(S, g))


def main(cfg_path):
    cfg = PipelineConfig.from_json(cfg_path)
    df, _ = pipeline.load_inputs(cfg)
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    good = wdf[wdf["window_qc_flag"] == 0]

    isara_bridge.import_isara(cfg.paths.isara_code_dir)
    st = isara_bridge.build_sizing_state(grid, cfg)
    psd_cols = [psd_col_name(d) for d in grid.dpg_um]

    recs = []
    for ts, row in good.iterrows():
        dnd = row[psd_cols].to_numpy(float)
        if grid.penetration is not None:
            dnd = dnd * grid.penetration
        fin = np.isfinite(dnd) & (grid.dpg_um > 0)
        if fin.sum() < 25 or row["Sc550_dry_mean"] < 5:
            continue
        dpg, dv = grid.dpg_um[fin], dnd[fin]
        if st is not None:
            k = szc.nearest_candidate(st, 1.52, 0.005)
            dpg, dv = szc.apply(st, k, dpg, dv, cols=np.where(fin)[0])
        lnD = np.log(dpg)
        deff = np.trapz(dv * dpg ** 3, lnD) / np.trapz(dv * dpg ** 2, lnD)
        recs.append((ts, deff, dpg, dv))
    recs.sort(key=lambda r: r[1])
    deffs = np.array([r[1] for r in recs])
    print(f"{len(recs)} usable windows; Deff um: "
          f"p5={np.percentile(deffs, 5):.3f} p50={np.percentile(deffs, 50):.3f} "
          f"p95={np.percentile(deffs, 95):.3f}")

    curves = {mr: load_curves(f"{mr:.4f}") for mr in (1.48, 1.52, 1.56)}

    def b_and_s(mr, dpg, dv):
        b, s = {}, {}
        for wv in WVL_NM:
            _, _, st_, bt = coeffs(*curves[mr], dpg, dv, wv / 1000.0)
            b[wv], s[wv] = bt / st_, st_
        return b, s

    idx = np.linspace(0, len(recs) - 1, min(200, len(recs))).astype(int)
    sig_all, sig_sca, db_siz, db_rri = [], [], [], []
    for i in idx:
        ts, de, dpg, dv = recs[i]
        b0, s0 = b_and_s(1.52, dpg, dv)
        bS, sS = b_and_s(1.52, dpg * np.exp(0.05), dv)
        bR, sR = b_and_s(1.56, dpg, dv)
        dbS = [bS[w] / b0[w] - 1 for w in WVL_NM]
        dbR = [bR[w] / b0[w] - 1 for w in WVL_NM]
        dsS = [sS[w] / s0[w] - 1 for w in WVL_NM]
        dsR = [sR[w] / s0[w] - 1 for w in WVL_NM]
        g = np.array(dsR + dbR) / 0.04
        nus = [np.array(dsS + dbS),                   # sizing 5% lnD
               np.r_[[0.10] * 3, [0.0] * 3],          # concentration
               np.r_[[0.08] * 3, [0.0] * 3],          # neph cal common
               np.r_[[0.0] * 3, [0.02] * 3]]          # b common
        noise = [0.02] * 3 + [0.04] * 3
        sig_all.append(fisher(g, nus, noise))
        sig_sca.append(fisher(g[:3], [n[:3] for n in nus[:3]], noise[:3]))
        db_siz.append(dbS)
        db_rri.append(dbR)
    sig_all, sig_sca = np.array(sig_all), np.array(sig_sca)
    db_siz, db_rri = 100 * np.array(db_siz), 100 * np.array(db_rri)

    pct = lambda a, f: " ".join(f"p{q:02d}={np.percentile(a, q):{f}}"
                                for q in (5, 25, 50, 75, 95))
    print("\nsigma_RRI (sca+b): ", pct(sig_all, ".4f"))
    print("sigma_RRI (sca):   ", pct(sig_sca, ".4f"))
    for j, wv in enumerate(WVL_NM):
        print(f"db/b sizing+5% @{wv}: ", pct(db_siz[:, j], "+.2f"))
    for j, wv in enumerate(WVL_NM):
        print(f"db/b RRI+0.04  @{wv}: ", pct(db_rri[:, j], "+.2f"))
    print(f"\nfrac windows |sizing db/b @550| > 2%: "
          f"{np.mean(np.abs(db_siz[:, 1]) > 2):.2f}")
    print(f"frac windows sigma(sca+b) > 0.035:    "
          f"{np.mean(sig_all > 0.035):.2f}")
    de_s = [recs[i][1] for i in idx]
    print(f"corr(Deff, sigma_all) = {np.corrcoef(de_s, sig_all)[0, 1]:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/activate_2021_full.json")
