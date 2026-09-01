#!/usr/bin/env python
"""Boxplots of ambient-state retrieved values and their 1-sigma uncertainties
vs ambient extinction, per campaign year (V7 bundles).

Four figures: {values, uncertainties} x {2020, 2021}; panels SSA(550),
ambient RRI, ambient IRI, ambient scattering AE(450-700); x = ambient
532 nm extinction bins in km^-1 (0-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.3,
0.3-0.5, >=0.5; bins with <5 windows dropped). Boxes: median/IQR,
whiskers 5-95%, outliers hidden; n annotated per bin.
"""
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/wrespino/Synced/ACMAP_Meloe/SuborbitalDataSets/ACTIVATE/isara_output/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, np.inf]
LABELS = ["0–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.3",
          "0.3–0.5", "≥0.5"]
# ordinal blue ramp (dataviz reference palette, steps 250->550 + interpolants)
RAMP = ["#86b6ef", "#5f9fe9", "#3987e5", "#2a72c8", "#1c5cab", "#154a8a"]
INK, MUTED = "#1a1a2e", "#6b6b7b"

VARS = [
    ("SSA (550 nm)", "amb_cal_SSA_550_unitless", 1.0),
    ("ambient RRI", "amb_RRI_unitless", 1.0),
    ("ambient IRI", "amb_IRI_unitless", 1.0),
    ("ambient AE (450–700)", "amb_AE", 1.0),
]


def load(year, tag):
    b = pickle.load(open(f"{BASE}ISARA_ACTIVATE_{year}_{tag}_60s_V7.pkl", "rb"))
    return b["results"], b["uncertainty"]


def frame(res, unc):
    ext_km = res["amb_cal_ext_coef_532_m-1"].astype(float) * 1e3  # m^-1 -> km^-1
    vals = {
        "amb_cal_SSA_550_unitless": res["amb_cal_SSA_550_unitless"].astype(float),
        "amb_RRI_unitless": res["amb_RRI_unitless"].astype(float),
        "amb_IRI_unitless": res["amb_IRI_unitless"].astype(float),
        "amb_AE": (np.log(res["amb_cal_sca_coef_450_m-1"].astype(float)
                          / res["amb_cal_sca_coef_700_m-1"].astype(float))
                   / np.log(700 / 450)),
    }
    sigs = {
        "amb_cal_SSA_550_unitless": unc["amb_cal_SSA_550_unitless"].astype(float),
        "amb_RRI_unitless": unc["amb_RRI_unitless"].astype(float),
        "amb_IRI_unitless": unc["amb_IRI_unitless"].astype(float),
        "amb_AE": unc["amb_AE_unitless"].astype(float),
    }
    return ext_km, vals, sigs


def boxfig(ext_km, data, title, fname, sigma=False):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, (label, key, scale) in zip(axes.ravel(), VARS):
        series = data[key] * scale
        groups, labels, colors, ns = [], [], [], []
        for i in range(len(EDGES) - 1):
            m = (ext_km >= EDGES[i]) & (ext_km < EDGES[i + 1]) & np.isfinite(series)
            if m.sum() >= 5:
                groups.append(series[m].to_numpy())
                labels.append(LABELS[i])
                colors.append(RAMP[i])
                ns.append(int(m.sum()))
        if not groups:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(groups, whis=(5, 95), showfliers=False, widths=0.62,
                        patch_artist=True, medianprops=dict(color=INK, lw=1.6))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor("white")
            patch.set_linewidth(1.0)
        for part in ("whiskers", "caps"):
            for ln in bp[part]:
                ln.set_color(MUTED)
                ln.set_linewidth(1.0)
        ax.set_xticklabels(labels, fontsize=8, color=INK)
        ylo = ax.get_ylim()[0]
        for k, n in enumerate(ns):
            ax.annotate(f"n={n}", (k + 1, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(("σ of " if sigma else "") + label, fontsize=10,
                     color=INK, pad=6)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e8e8ee", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
    for ax in axes[1]:
        ax.set_xlabel("ambient extinction 532 nm (km$^{-1}$)", fontsize=9,
                      color=INK)
    fig.suptitle(title, fontsize=12, color=INK)
    fig.text(0.01, 0.005,
             "boxes: median/IQR; whiskers 5–95%; outliers hidden; "
             "bins with <5 windows dropped",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(fname, facecolor="white")
    plt.close(fig)
    print("wrote", fname)


for year, tag in [("2021", "submicron"), ("2020", "total")]:
    res, unc = load(year, tag)
    ext_km, vals, sigs = frame(res, unc)
    boxfig(ext_km, vals,
           f"ACTIVATE {year} ({tag}) — ambient-state retrievals vs "
           "ambient extinction (V7)",
           f"{OUT}/V7_{year}_ambient_values_boxplots.png")
    boxfig(ext_km, sigs,
           f"ACTIVATE {year} ({tag}) — 1σ uncertainties of "
           "ambient-state retrievals (V7)",
           f"{OUT}/V7_{year}_ambient_uncertainty_boxplots.png", sigma=True)
