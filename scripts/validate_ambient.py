#!/usr/bin/env python
"""Validate the V4 ambient-state products against LARGE's ambient variables.

LARGE computes Sc550_submicron_amb (and Ext532_submicron_amb, SSA_amb_550nm)
upstream with the same gamma parameterization and DLH ambient RH, so our
window-mean ambient products should track theirs closely. This is the
independent check of the ambient branch (the 80%-RH wet state cannot be
validated this way because it is the kappa fitting target).

Usage::

    python scripts/validate_ambient.py FILE_V4.nc [-o report.txt]

Compares, per QC-passing window:
  scattering_ambient_synthesized      vs window-mean Sc550_submicron_amb
  scattering_ambient_calculated@550   vs window-mean Sc550_submicron_amb
  extinction_ambient_calculated@532   vs window-mean Ext532_submicron_amb
  ssa_ambient_calculated@550          vs window-mean SSA_amb_550nm
Window means of the LARGE 1 Hz products use the same seconds that carry the
window (all rows, not only QC-valid ones, since LARGE applies its own QC).
"""

import argparse
import sys

import netCDF4
import numpy as np


def _fl(var):
    return np.ma.filled(var[:].astype(float), np.nan)


def window_mean_of_obs(obs_1hz, window_id):
    """Mean of a 1 Hz (flight,time) field over each 60 s window id."""
    out = np.full_like(obs_1hz, np.nan)
    flat_obs = obs_1hz.ravel()
    flat_id = window_id.ravel()
    ok = np.isfinite(flat_id)
    ids = flat_id[ok].astype(np.int64)
    vals = flat_obs[ok]
    good = np.isfinite(vals)
    if not ids.size:
        return out
    size = int(ids.max()) + 1
    n = np.bincount(ids[good], minlength=size)
    s = np.bincount(ids[good], weights=vals[good], minlength=size)
    with np.errstate(invalid="ignore"):
        mean_by_id = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    res = np.full(flat_obs.shape, np.nan)
    res[ok] = mean_by_id[ids]
    return res.reshape(obs_1hz.shape)


def stats(ours, theirs, label, lines):
    m = np.isfinite(ours) & np.isfinite(theirs)
    if m.sum() < 10:
        lines.append(f"{label}: <10 paired samples, skipped")
        return
    x, y = theirs[m], ours[m]
    r = np.corrcoef(x, y)[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nanmedian(np.where(np.abs(x) > 1e-9, y / x, np.nan))
    bias = np.mean(y - x)
    rmsd = np.sqrt(np.mean((y - x) ** 2))
    lines.append(f"{label}: n={m.sum()}  r={r:.4f}  median(ours/LARGE)={ratio:.4f}  "
                 f"bias={bias:+.3g}  rmsd={rmsd:.3g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nc")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)

    ds = netCDF4.Dataset(args.nc)
    wl = list(ds["wavelength"][:])
    i550, i532 = wl.index(550.0), wl.index(532.0)
    opt = ds["observations/optical"]
    wo = ds["windowed/observations"]
    wr = ds["windowed/retrievals"]

    # window id per second: derive from the time axis (60 s blocks per flight)
    t = ds["time"][:]
    nflight = ds.dimensions["flight"].size
    win_of_t = np.floor(t / 60.0).astype(np.int64)
    nwin_per_flight = win_of_t.max() + 1
    wid = (np.arange(nflight)[:, None] * nwin_per_flight + win_of_t[None, :]).astype(float)

    qc = np.ma.filled(wo["window_qc_flag"][:].astype(float), np.nan)
    passing = qc == 0

    large_sc = window_mean_of_obs(_fl(opt["scattering_submicron_amb"][..., i550]), wid)
    large_ext = window_mean_of_obs(_fl(opt["extinction_submicron_amb"][..., i532]), wid)
    large_ssa = window_mean_of_obs(_fl(opt["ssa_amb"][..., i550]), wid)

    lines = [f"ambient-state validation vs LARGE: {args.nc}",
             f"QC-passing seconds: {int(passing.sum())}"]
    pairs = [
        ("scattering_ambient_synthesized vs LARGE Sc550_amb",
         _fl(wo["scattering_ambient_synthesized"][:]), large_sc),
        ("scattering_ambient_calculated@550 vs LARGE Sc550_amb",
         _fl(wr["scattering_ambient_calculated"][..., i550]), large_sc),
        ("extinction_ambient_calculated@532 vs LARGE Ext532_amb",
         _fl(wr["extinction_ambient_calculated"][..., i532]), large_ext),
        ("ssa_ambient_calculated@550 vs LARGE SSA_amb_550",
         _fl(wr["ssa_ambient_calculated"][..., i550]), large_ssa),
    ]
    for label, ours, theirs in pairs:
        stats(np.where(passing, ours, np.nan), np.where(passing, theirs, np.nan),
              label, lines)
        rh = np.ma.filled(wo["rh_ambient"][:].astype(float), np.nan)
        lowrh = passing & (rh < 90)
        stats(np.where(lowrh, ours, np.nan), np.where(lowrh, theirs, np.nan),
              f"  [RH<90% subset] {label}", lines)

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
