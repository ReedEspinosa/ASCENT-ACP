#!/usr/bin/env python
"""Hemispheric-backscatter-fraction fingerprint: does TSI b = Bsp/Sp separate
RRI from sizing?

Computes, from the MOPSMAP sphere phase functions (a1 Legendre coefficients),
the backscatter fraction b at 450/550/700 for a representative retrieved PSD
under: RRI 1.52 -> 1.56 / 1.48, lnD +5% / +2%, concentration +10%. Reported
for the ideal hemispheric integral (90-180) and the TSI 3563 truncated
geometry (total 7-170, backscatter 90-170).

Usage: python scripts/backscatter_fingerprint.py [CONFIG]
"""
import glob
import os
import sys

import numpy as np
import netCDF4 as nc
from numpy.polynomial import legendre

sys.path.insert(0, ".")
from ASCENT_ACP import filtering, isara_bridge, pipeline, sizebins, windows
from ASCENT_ACP import sizing_correction as szc
from ASCENT_ACP.config import PipelineConfig
from ASCENT_ACP.windows import psd_col_name

SRC = ("/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/"
       "optical_dataset/spheres/")
MI = "0.004300"
XMAX = 60.0
WVL_NM = [450, 550, 700]


def load_curves(mr_str):
    """qsca and angular fractions vs size parameter for one sphere file."""
    f = os.path.join(SRC, f"sphere_{mr_str}_{MI}.nc")
    th = np.linspace(0.0, np.pi, 1441)
    mu, sinth = np.cos(th), np.sin(th)
    masks = {
        "hemi_back": (th >= np.pi / 2),
        "tsi_total": (th >= np.deg2rad(7)) & (th <= np.deg2rad(170)),
        "tsi_back": (th >= np.pi / 2) & (th <= np.deg2rad(170)),
    }
    with nc.Dataset(f) as d:
        sp = d["sizepara"][:].filled(np.nan)
        qsca = d["qsca"][:].filled(np.nan)
        lmax = d["lmax"][:].filled(0).astype(int)
        offs = np.r_[0, np.cumsum(lmax + 1)]
        a1 = d["a1"][:].filled(np.nan)
        sel = np.where(sp <= XMAX)[0]
        frac = {k: np.full(len(sel), np.nan) for k in masks}
        for n, k in enumerate(sel):
            p11 = legendre.legval(mu, a1[offs[k]:offs[k] + lmax[k] + 1])
            for name, m in masks.items():
                frac[name][n] = 0.5 * np.trapz(np.where(m, p11 * sinth, 0), th)
    return sp[sel], qsca[sel], frac


def coeffs(sp, qsca, frac, dpg_um, dnd, lam_um):
    """(sca_total, bsp_hemi, sca_tsi, bsp_tsi) integrals over the PSD."""
    x = np.pi * dpg_um / lam_um
    ok = (x >= sp.min()) & (x <= sp.max()) & np.isfinite(dnd)
    lx, lsp = np.log(x[ok]), np.log(sp)
    q = np.interp(lx, lsp, qsca)
    area = 0.25 * np.pi * dpg_um[ok] ** 2 * dnd[ok]
    lnD = np.log(dpg_um[ok])
    def integ(qq):
        return np.trapz(qq * area, lnD)
    return (integ(q),
            integ(q * np.interp(lx, lsp, frac["hemi_back"])),
            integ(q * np.interp(lx, lsp, frac["tsi_total"])),
            integ(q * np.interp(lx, lsp, frac["tsi_back"])))


def main(cfg_path):
    cfg = PipelineConfig.from_json(cfg_path)
    df, _ = pipeline.load_inputs(cfg)
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    good = wdf[wdf["window_qc_flag"] == 0]
    sc = good["Sc550_dry_mean"].astype(float)
    cand = good[(sc > sc.quantile(0.4)) & (sc < sc.quantile(0.6))]
    isara_bridge.import_isara(cfg.paths.isara_code_dir)
    st = isara_bridge.build_sizing_state(grid, cfg)
    for ts, row in cand.iterrows():
        dnd = np.array([row[psd_col_name(d)] for d in grid.dpg_um], float)
        if grid.penetration is not None:
            dnd = dnd * grid.penetration
        fin = np.isfinite(dnd) & (grid.dpg_um > 0)
        if fin.sum() >= 25:
            break
    dpg_f, dnd_f = grid.dpg_um[fin], dnd[fin]
    if st is not None:
        k = szc.nearest_candidate(st, 1.52, 0.005)
        dpg_f, dnd_f = szc.apply(st, k, dpg_f, dnd_f, cols=np.where(fin)[0])
    print(f"window {ts}  Sc550_meas={row['Sc550_dry_mean']:.1f} Mm-1  "
          f"bins={fin.sum()}")

    curves = {mr: load_curves(f"{mr:.4f}") for mr in (1.48, 1.52, 1.56)}

    def b_frac(mr, dpg, dnd_v):
        out = {}
        for wv in WVL_NM:
            s, bh, st_, bt = coeffs(*curves[mr], dpg, dnd_v, wv / 1000.0)
            out[wv] = (bh / s, bt / st_)   # (ideal hemi b, TSI-truncated b)
        return out

    b0 = b_frac(1.52, dpg_f, dnd_f)
    cases = [
        ("RRI 1.52 -> 1.56", b_frac(1.56, dpg_f, dnd_f)),
        ("RRI 1.52 -> 1.48", b_frac(1.48, dpg_f, dnd_f)),
        ("sizing +5% lnD", b_frac(1.52, dpg_f * np.exp(0.05), dnd_f)),
        ("sizing +2% lnD", b_frac(1.52, dpg_f * np.exp(0.02), dnd_f)),
        ("concentration +10%", b_frac(1.52, dpg_f, dnd_f * 1.10)),
    ]
    hdr = "".join(f"  {wv}nm" for wv in WVL_NM)
    print(f"\nbaseline b (ideal 90-180):   "
          + "  ".join(f"{b0[wv][0]:.4f}" for wv in WVL_NM))
    print(f"baseline b (TSI 90-170):     "
          + "  ".join(f"{b0[wv][1]:.4f}" for wv in WVL_NM))
    for geom, gi in [("ideal 90-180/full", 0), ("TSI 90-170/7-170", 1)]:
        print(f"\n--- db/b %% ({geom}) ---{hdr}")
        for lab, bb in cases:
            d = [100 * (bb[wv][gi] / b0[wv][gi] - 1) for wv in WVL_NM]
            print(f"{lab:22s}" + "".join(f" {v:+6.2f}" for v in d))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/activate_2021_full.json")
