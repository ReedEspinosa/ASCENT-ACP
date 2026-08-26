#!/usr/bin/env python
"""Dump and diff per-variable statistics of the pipeline's grouped netCDF files.

Used as the regression harness for netCDF format changes (V3 -> V4): capture a
baseline dump of the old file, then diff the new file against it with an
explicit rename/scale map so every variable must be accounted for.

Usage::

    python scripts/compare_nc_versions.py dump FILE.nc -o baseline.json
    python scripts/compare_nc_versions.py diff baseline.json NEW.nc \
        [--map rename_map.json] [--rtol 1e-6]

The rename map (JSON) describes intentional changes; unmapped variables must
match by identical path. Entries::

    {
      "renames":  {"/old/path/name": "/new/path/name"},
      "scales":   {"/old/path/name": 1e6},          # new = old * scale
      "dropped":  ["/old/path/gone"],               # deliberately removed
      "added":    ["/new/path/brandnew"],           # deliberately new
      "wavelength_slices": {"/old/path/Sc550_x": ["/new/path/scattering_x", 550]}
    }

A wavelength_slices entry checks the old 2-D variable against one wavelength
slice of a new 3-D (flight, time, wavelength) variable.
"""

import argparse
import hashlib
import json
import sys

import netCDF4
import numpy as np


def _walk(group, path=""):
    for name, var in group.variables.items():
        yield f"{path}/{name}", var
    for gname, sub in group.groups.items():
        yield from _walk(sub, f"{path}/{gname}")


def _var_stats(var):
    out = {
        "dims": list(var.dimensions),
        "shape": list(var.shape),
        "dtype": str(var.dtype),
        "units": getattr(var, "units", None),
    }
    if var.dtype == str or var.dtype.kind in ("S", "U", "O"):
        vals = [str(x) for x in np.ravel(var[:])]
        out["n"] = len(vals)
        out["sha"] = hashlib.sha256("\n".join(vals).encode()).hexdigest()[:16]
        return out
    arr = _to_float(var[:])
    finite = np.isfinite(arr)
    out["n_finite"] = int(finite.sum())
    if out["n_finite"]:
        vals = arr[finite]
        out.update(min=float(vals.min()), max=float(vals.max()),
                   mean=float(vals.mean()), std=float(vals.std()))
        out["sha"] = hashlib.sha256(
            np.ascontiguousarray(vals, dtype=np.float64).tobytes()).hexdigest()[:16]
    return out


def dump(nc_path):
    ds = netCDF4.Dataset(nc_path)
    try:
        return {path: _var_stats(var) for path, var in _walk(ds)}
    finally:
        ds.close()


def _to_float(data):
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data.astype(float), np.nan)
    return np.asarray(data, float)


def _values(var):
    return _to_float(var[:])


def _compare_arrays(old_stats, new_arr, scale, rtol, label, problems):
    """Compare baseline stats against freshly-read (possibly sliced) values."""
    finite = np.isfinite(new_arr)
    n_new = int(finite.sum())
    if n_new != old_stats.get("n_finite"):
        problems.append(f"{label}: n_finite {old_stats.get('n_finite')} -> {n_new}")
        return
    if not n_new:
        return
    for key in ("min", "max", "mean"):
        old = old_stats.get(key)
        new = float(getattr(np, key if key != "mean" else "mean")(new_arr[finite]))
        expect = old * scale
        tol = rtol * max(abs(expect), abs(new), 1e-300)
        if abs(new - expect) > tol:
            problems.append(f"{label}: {key} {expect:.8g} (baseline*scale) vs {new:.8g}")


def diff(baseline_path, nc_path, map_path=None, rtol=1e-6):
    base = json.load(open(baseline_path))
    m = json.load(open(map_path)) if map_path else {}
    renames = m.get("renames", {})
    scales = m.get("scales", {})
    dropped = set(m.get("dropped", []))
    added = set(m.get("added", []))
    wslices = m.get("wavelength_slices", {})

    ds = netCDF4.Dataset(nc_path)
    try:
        new_vars = dict(_walk(ds))
        problems, checked_new = [], set()

        for old_path, stats in base.items():
            if old_path in dropped:
                continue
            scale = scales.get(old_path, 1.0)
            if old_path in wslices:
                new_path, wvl = wslices[old_path]
                var = new_vars.get(new_path)
                if var is None:
                    problems.append(f"{old_path}: mapped slice target {new_path} missing")
                    continue
                checked_new.add(new_path)
                grp = var.group()
                wvl_var = None
                while grp is not None and wvl_var is None:
                    wvl_var = grp.variables.get("wavelength")
                    grp = grp.parent if hasattr(grp, "parent") else None
                if wvl_var is None:
                    problems.append(f"{new_path}: no wavelength coordinate found")
                    continue
                k = int(np.argmin(np.abs(wvl_var[:] - wvl)))
                _compare_arrays(stats, _values(var)[..., k], scale, rtol,
                                f"{old_path} -> {new_path}[{wvl}nm]", problems)
                continue
            new_path = renames.get(old_path, old_path)
            var = new_vars.get(new_path)
            if var is None:
                problems.append(f"{old_path}: missing in new file (expected {new_path})")
                continue
            checked_new.add(new_path)
            if "sha" in stats and scale == 1.0 and stats.get("dtype") == str(var.dtype):
                fresh = _var_stats(var)
                if fresh.get("sha") == stats["sha"]:
                    continue
            if "n_finite" in stats:
                _compare_arrays(stats, _values(var), scale, rtol,
                                f"{old_path} -> {new_path}", problems)
            elif "sha" in stats:  # string variable with differing sha
                fresh = _var_stats(var)
                if fresh.get("sha") != stats["sha"]:
                    problems.append(f"{old_path}: string content changed")

        # every wavelength-slice target counts as covered even when several
        # old columns map into one new variable
        for new_path in list(new_vars):
            if new_path in checked_new or new_path in added:
                continue
            problems.append(f"{new_path}: NEW variable not declared in map 'added'")

        return problems
    finally:
        ds.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("nc")
    d.add_argument("-o", "--out", required=True)
    c = sub.add_parser("diff")
    c.add_argument("baseline")
    c.add_argument("nc")
    c.add_argument("--map", default=None)
    c.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    if args.cmd == "dump":
        json.dump(dump(args.nc), open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return 0
    problems = diff(args.baseline, args.nc, args.map, args.rtol)
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        return 1
    print("OK: all variables accounted for and matching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
