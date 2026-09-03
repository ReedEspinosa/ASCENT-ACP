#!/usr/bin/env python
"""Scatter ISARA retrievals against the independent in-situ retrievals carried
in the exported netCDF itself: PI-Neph GRASP refractive indices
(/observations/pineph_retrievals) and DASH-SP growth factors converted to
kappa via kappa-Kohler (/observations/hygroscopic_growth).

Unlike compare_independent_ri.py (bundle + merged pickle), this reads only the
final netCDF product. Windowed retrievals are repeated at 1 Hz within each
60-s window; window blocks are recovered from contiguous constant-value runs.

Usage:
  python scripts/compare_independent_ri_nc.py OUT_DIR FILE.nc [FILE2.nc ...]
"""
import sys
from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, BLUE = "#1a1a2e", "#6b6b7b", "#3987e5"


def _filled(var):
    return np.ma.filled(var[:].astype(float), np.nan)


def window_pairs(ds):
    """Return per-window arrays: ISARA rri/iri/kappa (window value) and
    window-mean PI-Neph Real/Imag 532 and DASH-SP kappa."""
    ret = ds["windowed/retrievals"]
    rri = _filled(ret["refractive_index_real"])
    iri = _filled(ret["refractive_index_imag"])
    kap = _filled(ret["kappa"])
    af_cri = np.ma.filled(ret["attempt_flag_cri"][:], -1)
    af_kap = np.ma.filled(ret["attempt_flag_kappa"][:], -1)

    pn = ds["observations/pineph_retrievals"]
    if "refractive_index_real" in pn.variables:  # v5 layout: local wvl axis
        i532 = list(np.asarray(pn["wavelength"][:])).index(532.0)
        pn_rri = _filled(pn["refractive_index_real"])[:, :, i532]
        pn_iri = _filled(pn["refractive_index_imag"])[:, :, i532]
    else:                                        # v4/V2 layout: scalars
        pn_rri = _filled(pn["Real_532"])
        pn_iri = _filled(pn["Imag_532"])

    hg = ds["observations/hygroscopic_growth"]
    gf = _filled(hg["GF"])
    rh = _filled(hg["RH"])
    with np.errstate(invalid="ignore", divide="ignore"):
        dash_kap = (gf ** 3 - 1.0) * (100.0 - rh) / rh
    dash_kap[(rh <= 0) | (rh >= 100)] = np.nan

    try:  # v5+: explicit window ids
        widx = np.ma.filled(ds["windowed/observations/window_index"][:], -1)
    except (KeyError, IndexError):
        widx = None

    out = {k: [] for k in ("rri", "iri", "kap", "pn_rri", "pn_iri", "dash_kap",
                           "cri_ok", "kap_ok")}
    n_fl = rri.shape[0]
    for fl in range(n_fl):
        active = (af_cri[fl] >= 0) | (af_kap[fl] >= 0)
        idx = np.where(active)[0]
        if idx.size == 0:
            continue
        if widx is not None:
            wid = widx[fl, idx]
        else:
            # window boundary: time gap, or any repeated windowed value changes
            sig = np.column_stack([rri[fl, idx], iri[fl, idx], kap[fl, idx],
                                   af_cri[fl, idx], af_kap[fl, idx]])
            prev, cur = sig[:-1], sig[1:]
            changed = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur))).all(axis=1)
            boundary = np.r_[True, (np.diff(idx) != 1) | changed]
            wid = np.cumsum(boundary)
        for w in np.unique(wid[wid >= 0] if widx is not None else wid):
            sel = idx[wid == w]
            out["rri"].append(rri[fl, sel[0]])
            out["iri"].append(iri[fl, sel[0]])
            out["kap"].append(kap[fl, sel[0]])
            out["cri_ok"].append(af_cri[fl, sel[0]] == 2)
            out["kap_ok"].append(af_kap[fl, sel[0]] == 2)
            out["pn_rri"].append(np.nanmean(pn_rri[fl, sel])
                                 if np.isfinite(pn_rri[fl, sel]).any() else np.nan)
            out["pn_iri"].append(np.nanmean(pn_iri[fl, sel])
                                 if np.isfinite(pn_iri[fl, sel]).any() else np.nan)
            out["dash_kap"].append(np.nanmean(dash_kap[fl, sel])
                                   if np.isfinite(dash_kap[fl, sel]).any() else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


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


def run_file(nc_path, out_dir):
    ds = nc.Dataset(nc_path)
    tag = Path(nc_path).stem.replace("ISARA_", "")
    win_s = int(ds.getncattr("window_seconds")) if "window_seconds" in ds.ncattrs() else 60
    w = window_pairs(ds)
    made = []

    cri = w["cri_ok"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    n1 = panel(axes[0], w["pn_rri"][cri], w["rri"][cri],
               "PI-Neph GRASP Real(m) 532 nm", "ISARA dry RRI",
               "Real refractive index")
    n2 = panel(axes[1], w["pn_iri"][cri], w["iri"][cri],
               "PI-Neph GRASP Imag(m) 532 nm", "ISARA dry IRI",
               "Imaginary refractive index")
    fig.suptitle(f"{tag}: ISARA vs PI-Neph refractive index "
                 f"(co-windowed, {win_s} s)", fontsize=11, color=INK)
    fig.text(0.01, 0.01,
             "PI-Neph: GRASP retrieval from polarized phase functions "
             "(own inlet, ~2 um radius cut); ISARA: dry submicron fit. "
             "Sampling and RH states differ — spread expected.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = f"{out_dir}/{tag}_ISARA_vs_PINeph_RI.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out} (n={n1}/{n2})")
    made.append(out)

    kok = w["kap_ok"].astype(bool)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    n = panel(ax, w["dash_kap"][kok], w["kap"][kok],
              "DASH-SP kappa (from GF at instrument RH)",
              "ISARA kappa", "Hygroscopicity")
    fig.suptitle(f"{tag}: ISARA vs DASH-SP kappa", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    out = f"{out_dir}/{tag}_ISARA_vs_DASHSP_kappa.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out} (n={n})")
    made.append(out)
    ds.close()
    return made


def main(out_dir, nc_paths):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = []
    for p in nc_paths:
        made += run_file(p, out_dir)
    return made


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
