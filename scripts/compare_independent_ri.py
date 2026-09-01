#!/usr/bin/env python
"""Scatter ISARA retrievals against independent in-situ retrievals merged in
the same campaign frame: PI-Neph GRASP refractive indices (Real_*/Imag_*)
and DASH-SP growth factors (converted to kappa via kappa-Kohler).

Usage:
  python scripts/compare_independent_ri.py CONFIG.json BUNDLE.pkl OUT_DIR
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, BLUE = "#1a1a2e", "#6b6b7b", "#3987e5"


def window_mean(df, col, index, window_s):
    half = pd.Timedelta(seconds=window_s / 2)
    lo = index - half
    s = df[col].dropna()
    if s.empty:
        return pd.Series(np.nan, index=index)
    out = s.resample(f"{window_s}s", origin=lo[0]).mean()
    centers = out.index + half
    return pd.Series(out.values, index=centers).reindex(index, method="nearest",
                                                        tolerance=half)


def panel(ax, x, y, xlabel, ylabel, title):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        ax.set_visible(False)
        return 0
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo + 1e-12)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(x, y, s=14, color=BLUE, alpha=0.55, linewidths=0, zorder=2)
    r = np.corrcoef(x, y)[0, 1]
    bias = np.median(y - x)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    ax.annotate(f"n={len(x)}  r={r:.2f}\nmed Δ={bias:+.3g}", (0.03, 0.97),
                xycoords="axes fraction", va="top", fontsize=8, color=MUTED)
    ax.grid(color="#e8e8ee", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    return len(x)


def main(cfg_path, bundle_path, out_dir):
    cfg = json.load(open(cfg_path))
    b = pickle.load(open(bundle_path, "rb"))
    res = b["results"]
    ok = res[res["attempt_flag_CRI_unitless"] == 2]
    df = pd.read_pickle(cfg["paths"]["input_pkl"])
    win_s = cfg.get("window", {}).get("window_s", 60)

    def col_like(*subs):
        hits = [c for c in df.columns
                if all(s.lower() in c.lower() for s in subs)]
        return hits[0] if len(hits) == 1 else (hits[0] if hits else None)

    made = []
    # ---- PI-Neph RI ---------------------------------------------------------
    rri_col = col_like("PI-Neph", "Real_532")
    iri_col = col_like("PI-Neph", "Imag_532")
    if rri_col:
        pn_rri = window_mean(df, rri_col, ok.index, win_s)
        pn_iri = window_mean(df, iri_col, ok.index, win_s)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150)
        fig.patch.set_facecolor("white")
        n1 = panel(axes[0], pn_rri.to_numpy(float),
                   ok["dry_RRI_unitless"].to_numpy(float),
                   "PI-Neph GRASP Real(m) 532 nm", "ISARA dry RRI",
                   "Real refractive index")
        n2 = panel(axes[1], pn_iri.to_numpy(float),
                   ok["dry_IRI_unitless"].to_numpy(float),
                   "PI-Neph GRASP Imag(m) 532 nm", "ISARA dry IRI",
                   "Imaginary refractive index")
        fig.suptitle(f"{cfg['campaign']} {cfg['year']}: ISARA vs PI-Neph "
                     "refractive index (co-windowed)", fontsize=11, color=INK)
        fig.text(0.01, 0.01,
                 "PI-Neph: GRASP retrieval from polarized phase functions "
                 "(own inlet, ~2 um radius cut); ISARA: dry submicron fit. "
                 "Sampling and RH states differ — spread expected.",
                 fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        out = f"{out_dir}/{cfg['campaign']}_{cfg['year']}_ISARA_vs_PINeph_RI.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        print(f"wrote {out} (n={n1}/{n2})")
        made.append(out)
    else:
        print("no PI-Neph RI columns found")

    # ---- DASH-SP growth factor -> kappa ------------------------------------
    gf_col = col_like("DASH-SP", "GF")
    rh_col = col_like("DASH-SP", "RH")
    kk = res[res["attempt_flag_kappa_unitless"] == 2]
    if gf_col and rh_col and len(kk):
        gf = window_mean(df, gf_col, kk.index, win_s)
        rh = window_mean(df, rh_col, kk.index, win_s)
        dash_kappa = (gf ** 3 - 1) * (100 - rh) / rh
        fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
        fig.patch.set_facecolor("white")
        n = panel(ax, dash_kappa.to_numpy(float),
                  kk["kappa_unitless"].to_numpy(float),
                  "DASH-SP kappa (from GF at instrument RH)",
                  "ISARA kappa", "Hygroscopicity")
        fig.suptitle(f"{cfg['campaign']} {cfg['year']}: ISARA vs DASH-SP kappa",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0.02, 1, 0.92))
        out = f"{out_dir}/{cfg['campaign']}_{cfg['year']}_ISARA_vs_DASHSP_kappa.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        print(f"wrote {out} (n={n})")
        made.append(out)
    else:
        print("no DASH-SP GF/RH columns (or no kappa retrievals)")
    return made


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
