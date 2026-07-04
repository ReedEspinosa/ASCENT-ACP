"""Unconditional 60 s window averages of every raw merged-pickle column.

Distinct from ``windows.aggregate``: those means are over QC-valid 1 Hz rows
only (the ISARA inputs). Here we average **all** rows in each window so the raw
passthrough in ``/windowed/raw`` is a faithful straight mean, independent of QA.

Wind is vector-averaged: the speed/direction pair named by
``channels.wind_speed_suffix``/``wind_dir_suffix`` is decomposed into u/v
components, averaged, and recomposed, so the direction mean is meaningful
across the 0/360 wrap. Direction spread uses the Yamartino (1984) estimator.

The window grid (left-labelled, centered) matches ``windows.aggregate`` exactly,
so the result shares the retrieval time coordinate.
"""

import numpy as np
import pandas as pd


def find_wind_pair(columns, channels):
    """(speed_col, dir_col) resolved by suffix, or (None, None)."""
    spd = [c for c in columns if c.endswith(channels.wind_speed_suffix)]
    drn = [c for c in columns if c.endswith(channels.wind_dir_suffix)]
    if len(spd) == 1 and len(drn) == 1:
        return spd[0], drn[0]
    return None, None


def _vector_wind_stats(speed, direction, grouper):
    """Per-window vector-mean speed/direction and Yamartino direction std.

    ``direction`` is meteorological (degrees FROM, clockwise from north).
    Returns (speed_mean, dir_mean, dir_std) Series on the grouper's grid.
    """
    th = np.deg2rad(direction)
    u = -speed * np.sin(th)  # eastward wind component
    v = -speed * np.cos(th)  # northward wind component
    ub = u.groupby(grouper).mean()
    vb = v.groupby(grouper).mean()
    speed_mean = np.hypot(ub, vb)
    dir_mean = (np.rad2deg(np.arctan2(-ub, -vb))) % 360.0

    # Yamartino: unit-vector spread, independent of speed
    sa = np.sin(th).groupby(grouper).mean()
    ca = np.cos(th).groupby(grouper).mean()
    eps = np.sqrt(np.clip(1.0 - (sa**2 + ca**2), 0.0, 1.0))
    dir_std = np.rad2deg(np.arcsin(eps) * (1.0 + 0.1547 * eps**3))
    return speed_mean, dir_mean, dir_std


def aggregate_raw(df, cfg):
    """Return per-window mean/std/count for all numeric columns of ``df``.

    Index is the window *center* time (matching ``windows.aggregate``). Columns:
    ``<col>_mean``, ``<col>_std`` for every numeric column, plus ``n_points``
    (rows in the window). Means/stds skip NaNs; windows with no rows are
    dropped. The wind pair is vector-averaged (see module docstring), with the
    scalar speed mean preserved as ``<speed_col>_scalar_mean``.
    """
    w = cfg.window
    grouper = pd.Grouper(freq=f"{w.window_s}s", label="left")
    numeric = df.select_dtypes(include=[np.number])

    g = numeric.groupby(grouper)
    n_points = g.size()
    keep = n_points > 0
    mean = g.mean().rename(columns=lambda c: f"{c}_mean")
    std = g.std().rename(columns=lambda c: f"{c}_std")

    out = pd.concat([mean, std], axis=1)
    out["n_points"] = n_points

    ws, wd = find_wind_pair(numeric.columns, cfg.channels)
    if ws is not None:
        spd_vec, dir_vec, dir_std = _vector_wind_stats(numeric[ws], numeric[wd], grouper)
        out[f"{ws}_scalar_mean"] = out[f"{ws}_mean"]
        out[f"{ws}_mean"] = spd_vec
        out[f"{wd}_mean"] = dir_vec
        out[f"{wd}_std"] = dir_std

    out = out[keep]
    half = pd.Timedelta(seconds=w.window_s / 2)
    out.index = out.index + half
    out.index.name = "time"
    return out
