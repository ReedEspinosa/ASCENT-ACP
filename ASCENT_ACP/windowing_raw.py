"""Unconditional 60 s window averages of every raw merged-pickle column.

Distinct from ``windows.aggregate``: those means are over QC-valid 1 Hz rows
only (the ISARA inputs). Here we average **all** rows in each window so the raw
passthrough in ``/windowed/raw`` is a faithful straight mean, independent of QA.

The window grid (left-labelled, centered) matches ``windows.aggregate`` exactly,
so the result shares the retrieval time coordinate.
"""

import numpy as np
import pandas as pd


def aggregate_raw(df, cfg):
    """Return per-window mean/std/count for all numeric columns of ``df``.

    Index is the window *center* time (matching ``windows.aggregate``). Columns:
    ``<col>_mean``, ``<col>_std`` for every numeric column, plus ``n_points``
    (rows in the window). Means/stds skip NaNs; windows with no rows are dropped.
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
    out = out[keep]

    half = pd.Timedelta(seconds=w.window_s / 2)
    out.index = out.index + half
    out.index.name = "time"
    return out
