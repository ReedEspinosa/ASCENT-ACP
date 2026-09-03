#!/usr/bin/env python
"""Sanity-check plots from exported netCDFs: ISARA-calculated vs measured dry
SSA (per wavelength) and the distribution (PDF) of retrieved dry IRI.

Works on v4-layout files (windowed values repeated at 1 Hz; window blocks are
recovered from constant-value runs) and v5+ files (window_index used directly).

Usage:
  python scripts/plot_ssa_iri_check.py OUT_DIR FILE.nc [FILE2.nc ...]
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


def per_window(ds):
    """Collapse repeat-at-cadence windowed arrays to one row per window.

    Returns dict with per-window ssa_meas/ssa_calc (n_win, n_wvl), iri, and
    the wavelength axis. Windows limited to attempted CRI retrievals.
    """
    ret = ds["windowed/retrievals"]
    obs = ds["windowed/observations"]
    wvl = np.asarray(ds["wavelength"][:], float)
    rri = _filled(ret["refractive_index_real"])
    iri = _filled(ret["refractive_index_imag"])
    af = np.ma.filled(ret["attempt_flag_cri"][:], -1)
    ssa_m = _filled(obs["ssa_measured"])
    ssa_c = _filled(ret["ssa_dry_calculated"])

    try:
        widx = np.ma.filled(ds["windowed/observations/window_index"][:], -1)
    except (KeyError, IndexError):
        widx = None

    out = {"ssa_meas": [], "ssa_calc": [], "iri": [], "cri_ok": []}
    for fl in range(rri.shape[0]):
        idx = np.where(af[fl] >= 0)[0]
        if idx.size == 0:
            continue
        if widx is not None:
            wid = widx[fl, idx]
        else:
            sig = np.column_stack([rri[fl, idx], iri[fl, idx], af[fl, idx]])
            prev, cur = sig[:-1], sig[1:]
            changed = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur))).all(axis=1)
            wid = np.cumsum(np.r_[True, (np.diff(idx) != 1) | changed])
        for wd in np.unique(wid[wid >= 0] if widx is not None else wid):
            first = idx[wid == wd][0]
            out["ssa_meas"].append(ssa_m[fl, first])
            out["ssa_calc"].append(ssa_c[fl, first])
            out["iri"].append(iri[fl, first])
            out["cri_ok"].append(af[fl, first] == 2)
    return {"wvl": wvl,
            "ssa_meas": np.asarray(out["ssa_meas"], float),
            "ssa_calc": np.asarray(out["ssa_calc"], float),
            "iri": np.asarray(out["iri"], float),
            "cri_ok": np.asarray(out["cri_ok"], bool)}


def ssa_panel(ax, x, y, wvl_nm):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        ax.set_visible(False)
        return 0
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.03 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(x, y, s=10, color=BLUE, alpha=0.4, linewidths=0, zorder=2)
    r = np.corrcoef(x, y)[0, 1]
    ax.annotate(f"n={len(x)}  r={r:.2f}\nmed Δ={np.median(y - x):+.3f}",
                (0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=8, color=MUTED)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured dry SSA", fontsize=9, color=INK)
    ax.set_ylabel("calculated dry SSA", fontsize=9, color=INK)
    ax.set_title(f"{wvl_nm:.0f} nm", fontsize=10, color=INK, pad=5)
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
    w = per_window(ds)
    ds.close()

    ok = w["cri_ok"]
    # wavelengths with both measured and calculated SSA
    have = [i for i in range(len(w["wvl"]))
            if np.isfinite(w["ssa_meas"][ok, i]).any()
            and np.isfinite(w["ssa_calc"][ok, i]).any()]
    ncol = len(have) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.9), dpi=150)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes[:-1], have):
        ssa_panel(ax, w["ssa_meas"][ok, i], w["ssa_calc"][ok, i], w["wvl"][i])

    ax = axes[-1]
    iri = w["iri"][ok]
    iri = iri[np.isfinite(iri)]
    bins = np.arange(0, max(0.031, np.nanmax(iri) + 0.002), 0.001)
    ax.hist(iri, bins=bins, density=True, color=BLUE, alpha=0.75)
    ax.annotate(f"n={len(iri)}\nmed={np.median(iri):.4f}\n"
                f"p90={np.percentile(iri, 90):.4f}",
                (0.97, 0.97), xycoords="axes fraction", va="top", ha="right",
                fontsize=8, color=MUTED)
    ax.set_xlabel("retrieved dry IRI", fontsize=9, color=INK)
    ax.set_ylabel("PDF", fontsize=9, color=INK)
    ax.set_title("dry IRI distribution", fontsize=10, color=INK, pad=5)
    ax.grid(color="#e8e8ee", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)

    fig.suptitle(f"{tag}: calculated vs measured dry SSA + dry IRI PDF "
                 f"(per {win_s}-s window, CRI success)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    out = f"{out_dir}/{tag}_SSA_IRI_check.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return out, (tag, iri)


def main(out_dir, nc_paths):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made, iri_sets = [], []
    for p in nc_paths:
        out, iri = run_file(p, out_dir)
        made.append(out)
        iri_sets.append(iri)

    if len(iri_sets) > 1:  # overlay of all IRI PDFs
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        fig.patch.set_facecolor("white")
        for tag, iri in iri_sets:
            bins = np.arange(0, 0.032, 0.001)
            ax.hist(iri, bins=bins, density=True, histtype="step", lw=1.8,
                    label=f"{tag} (n={len(iri)})")
        ax.set_xlabel("retrieved dry IRI", fontsize=10, color=INK)
        ax.set_ylabel("PDF", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("dry IRI PDFs, all files", fontsize=11, color=INK)
        ax.grid(color="#e8e8ee", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = f"{out_dir}/ALL_dry_IRI_pdfs.png"
        fig.savefig(out, facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")
        made.append(out)
    return made


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
