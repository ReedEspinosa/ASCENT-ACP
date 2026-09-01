#!/usr/bin/env python
"""Per-flight SMPS-vs-OPC sizing offsets from the PSD overlap region.

SMPS diameters are electrical-mobility (refractive-index independent), so the
lnD shift that best aligns each optical sizer's flight-mean spectrum onto the
SMPS spectrum in their overlap is a direct measurement of that sizer's total
sizing error (RI-driven + everything else). Positive shift = the OPC
undersizes (its diameters must grow), the same convention as the retrieval's
sizing_lnD_shift.

Fit: SMPS(l) ~ a * OPC(l - s) in log space over the overlap grid, grid search
on s with the scale a analytic.

Usage: python scripts/smps_opc_overlap.py MERGED_PKL
"""
import re
import sys

import numpy as np
import pandas as pd


def bin_cols(df, pattern):
    out = []
    for c in df.columns:
        m = re.search(pattern, c)
        if m:
            out.append((float(m.group(1)) / 1000.0, c))
    out.sort()
    return np.array([d for d, _ in out]), [c for _, c in out]


def fit_shift(lnd_s, y_s, lnd_o, y_o, lo, hi):
    """(shift, ln-scale, rms) aligning OPC onto SMPS over [lo, hi] in lnD."""
    grid = np.linspace(-0.20, 0.30, 101)
    l_eval = np.linspace(max(lo, lnd_s.min()), min(hi, lnd_s.max()), 40)
    ls = np.interp(l_eval, lnd_s, y_s)
    best = (np.nan, np.nan, np.inf)
    for s in grid:
        le = l_eval - s
        ok = (le >= lnd_o.min()) & (le <= lnd_o.max())
        if ok.sum() < 10:
            continue
        lo_i = np.interp(le[ok], lnd_o, y_o)
        a = np.mean(ls[ok] - lo_i)
        rms = np.sqrt(np.mean((ls[ok] - lo_i - a) ** 2))
        if rms < best[2]:
            best = (s, a, rms)
    return best


def main(pkl):
    df = pd.read_pickle(pkl)
    d_sm, c_sm = bin_cols(df, r"_SMPS_(\d+)nm$")
    d_la, c_la = bin_cols(df, r"_LAS_(\d+)nm$")
    d_uh, c_uh = bin_cols(df, r"_UHSAS_(\d+)nm_AmmSO4$")
    print(f"SMPS {len(d_sm)} bins {d_sm.min():.3f}-{d_sm.max():.3f} um; "
          f"LAS {len(d_la)} bins from {d_la.min():.3f}; "
          f"UHSAS {len(d_uh)} bins from {d_uh.min() if len(d_uh) else np.nan:.3f}")

    days = df.index.normalize()
    rows = []
    for day in sorted(days.unique()):
        sub = df[days == day]
        specs = {}
        for tag, dd, cc in [("SMPS", d_sm, c_sm), ("LAS", d_la, c_la),
                            ("UHSAS", d_uh, c_uh)]:
            if not len(dd):
                continue
            v = sub[cc].astype(float)
            n_rows = int(v.notna().any(axis=1).sum())
            m = v.mean(axis=0, skipna=True).to_numpy()
            good = np.isfinite(m) & (m > 0)
            if good.sum() >= 8 and n_rows >= 300:
                specs[tag] = (np.log(dd[good]), np.log(m[good]), n_rows)
        if "SMPS" not in specs:
            continue
        rec = {"flight": str(day.date()), "n_smps_s": specs["SMPS"][2]}
        for tag, lo in [("LAS", 0.105), ("UHSAS", 0.075)]:
            if tag not in specs:
                continue
            s, a, rms = fit_shift(specs["SMPS"][0], specs["SMPS"][1],
                                  specs[tag][0], specs[tag][1],
                                  np.log(lo), np.log(0.30))
            rec[f"{tag}_shift"] = s
            rec[f"{tag}_lnscale"] = a
            rec[f"{tag}_rms"] = rms
        rows.append(rec)
    res = pd.DataFrame(rows).set_index("flight")
    pd.set_option("display.width", 150)
    print(res.round(3).to_string())
    print("\n--- campaign summary (lnD shift, positive = OPC undersizes) ---")
    for tag in ["LAS", "UHSAS"]:
        c = f"{tag}_shift"
        if c in res:
            v = res[c].dropna()
            print(f"  {tag:6s} n={len(v):2d}  med={v.median():+.3f}  "
                  f"IQR=[{v.quantile(.25):+.3f},{v.quantile(.75):+.3f}]  "
                  f"std={v.std():.3f}")
            c2 = f"{tag}_lnscale"
            w = res[c2].dropna()
            print(f"         concentration ln-scale med={w.median():+.3f} "
                  f"(SMPS/OPC), std={w.std():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
