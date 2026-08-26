#!/usr/bin/env python
"""Quantify the retrieval shift from changing the dry RRI search grid.

Compares two results bundles (the .pkl checkpoints written by the pipeline)
for the same campaign year, e.g. the 1.51-1.54 grid vs the 1.47-1.56 grid::

    python scripts/compare_cri_grids.py OLD.pkl NEW.pkl [-o report.txt]

Reports per-variable shift statistics over windows retrieved in BOTH runs,
plus counts of windows that gained or lost a retrieval.
"""

import argparse
import pickle
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)["results"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old_pkl")
    ap.add_argument("new_pkl")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)

    old, new = load(args.old_pkl), load(args.new_pkl)
    both = old.index.intersection(new.index)
    old, new = old.loc[both], new.loc[both]

    lines = [f"CRI-grid comparison: {args.old_pkl} -> {args.new_pkl}",
             f"windows in both runs: {len(both)}"]

    ok_old = old.get("attempt_flag_CRI_unitless") == 2
    ok_new = new.get("attempt_flag_CRI_unitless") == 2
    lines.append(f"CRI success: {int(ok_old.sum())} -> {int(ok_new.sum())} "
                 f"(gained {int((~ok_old & ok_new).sum())}, "
                 f"lost {int((ok_old & ~ok_new).sum())})")
    kap_old = old.get("attempt_flag_kappa_unitless") == 2
    kap_new = new.get("attempt_flag_kappa_unitless") == 2
    lines.append(f"kappa success: {int(kap_old.sum())} -> {int(kap_new.sum())}")

    for col, label in [
        ("dry_RRI_unitless", "dry RRI"),
        ("dry_IRI_unitless", "dry IRI"),
        ("kappa_unitless", "kappa"),
        ("dry_cal_SSA_550_unitless", "dry SSA 550"),
        ("wet_cal_sca_coef_550_m-1", "wet sca 550 (m-1)"),
    ]:
        if col not in old or col not in new:
            continue
        a = old[col].to_numpy(float)
        b = new[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if not m.sum():
            continue
        d = b[m] - a[m]
        lines.append(
            f"{label}: n={m.sum()}  mean shift {np.mean(d):+.4g}  "
            f"median {np.median(d):+.4g}  p5 {np.percentile(d, 5):+.4g}  "
            f"p95 {np.percentile(d, 95):+.4g}  unchanged {int((d == 0).sum())}")

    if "dry_CRI_n_accepted_unitless" in new:
        n_acc = new["dry_CRI_n_accepted_unitless"].to_numpy(float)
        n_acc = n_acc[np.isfinite(n_acc)]
        if n_acc.size:
            lines.append(f"new-grid n_accepted: median {np.median(n_acc):.0f}  "
                         f"p95 {np.percentile(n_acc, 95):.0f}  max {n_acc.max():.0f}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
