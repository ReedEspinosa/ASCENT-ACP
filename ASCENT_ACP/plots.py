"""Sanity / verification plots for pipeline output."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .windows import psd_col_name


def make_all(results_df, grid, cfg, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    made = []
    made.append(_retrieval_histograms(results_df, plot_dir))
    made.append(_scattering_closure(results_df, cfg, plot_dir))
    made.append(_qc_summary(results_df, plot_dir))
    made.append(_mean_psd(results_df, grid, plot_dir))
    return [m for m in made if m]


def _retrieval_histograms(res, plot_dir):
    cols = ["dry_RRI_unitless", "dry_IRI_unitless", "kappa_unitless"]
    if not all(c in res for c in cols):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = ["retrieved RRI", "retrieved IRI", "retrieved kappa"]
    for ax, col, lab in zip(axes, cols, labels):
        vals = res[col].dropna()
        ax.hist(vals, bins=30, color="steelblue")
        ax.set_xlabel(lab)
        ax.set_title(f"n={len(vals)}, median={vals.median():.3g}" if len(vals) else "no data")
    fig.tight_layout()
    p = plot_dir / "retrieval_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _scattering_closure(res, cfg, plot_dir):
    wvls = cfg.channels.dry_wvl_sca
    have = [w for w in wvls if f"dry_cal_sca_coef_{w}_m-1" in res]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(4 * len(have), 4))
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, have):
        meas = res[f"Sc{w}_dry_mean"] * 1e-6
        calc = res[f"dry_cal_sca_coef_{w}_m-1"]
        ok = meas.notna() & calc.notna()
        ax.plot(meas[ok] * 1e6, calc[ok] * 1e6, ".", ms=4, alpha=0.6)
        lim = [0, max(1.0, 1.1 * meas[ok].max() * 1e6)] if ok.any() else [0, 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"measured dry Sc{w} (Mm$^{{-1}}$)")
        ax.set_ylabel(f"MOPSMAP Sc{w} (Mm$^{{-1}}$)")
        ax.set_xlim(lim), ax.set_ylim(lim)
    fig.suptitle("Dry scattering closure (retrieved CRI)")
    fig.tight_layout()
    p = plot_dir / "scattering_closure.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _qc_summary(res, plot_dir):
    cols = ["n_valid", "n_cloudy", "n_inlet_bad", "n_low_signal", "n_low_ssa"]
    cols = [c for c in cols if c in res]
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for c in cols:
        ax.plot(res.index, res[c], lw=0.8, label=c)
    ax.set_ylabel("1 Hz samples per window")
    ax.legend(ncol=len(cols), fontsize=8)
    ax.set_title("Window QC composition")
    fig.tight_layout()
    p = plot_dir / "qc_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _mean_psd(res, grid, plot_dir):
    psd = np.column_stack([res[psd_col_name(d)].to_numpy(float) for d in grid.dpg_um])
    if not np.isfinite(psd).any():
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    med = np.nanmedian(psd, axis=0)
    q1, q3 = np.nanpercentile(psd, [25, 75], axis=0)
    ax.loglog(grid.dpg_um, med, "-o", ms=3, label="median")
    ax.fill_between(grid.dpg_um, q1, q3, alpha=0.3, label="25-75%")
    ax.set_xlabel("diameter ($\\mu$m)")
    ax.set_ylabel("dN/dlogD$_p$ (cm$^{-3}$)")
    ax.set_title("Window-mean PSD across all retrieved windows")
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "mean_psd.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
