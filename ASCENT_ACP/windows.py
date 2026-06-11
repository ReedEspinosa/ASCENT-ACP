"""Block-average QC'd 1 Hz data into retrieval windows.

Windows follow the stability screening of Kacenelenbogen et al. (2022) A1.2:
a window survives only if it contains enough valid 1 Hz samples and the
scattering Angstrom exponent is steady within it.
"""

import numpy as np
import pandas as pd

# window_qc_flag bits
FLAG_TOO_FEW_POINTS = 1
FLAG_AE_UNSTABLE = 2
FLAG_MISSING_OPTICS = 4

FLAG_MEANINGS = {
    FLAG_TOO_FEW_POINTS: "fewer_valid_1Hz_points_than_min_valid_points",
    FLAG_AE_UNSTABLE: "scattering_angstrom_exponent_std_dev_above_threshold",
    FLAG_MISSING_OPTICS: "window_mean_scattering_or_absorption_missing",
}


def psd_col_name(dpg_um):
    """Column name for a PSD bin mean in the windows DataFrame."""
    return f"psd_{dpg_um:.6g}um"


def aggregate(df, optical, masks, grid, cfg):
    """Return one row per window with means/stds/counts and a QC bitmask.

    ``optical`` and ``masks`` are the outputs of ``filtering``; ``grid`` is the
    PSDGrid whose ``columns`` are averaged bin-by-bin. The index is the window
    *center* time; ``time_start``/``time_end`` carry the bounds.
    """
    w = cfg.window
    grouper = pd.Grouper(freq=f"{w.window_s}s", label="left")

    # Per-window rejection bookkeeping over ALL rows (not just valid ones).
    # pd.Grouper generates every calendar window between the first and last
    # sample; keep only windows that contain at least one 1 Hz row.
    n_total = masks.groupby(grouper).size()
    reason_counts = masks.groupby(grouper).sum().astype(int)
    reason_counts.columns = [f"n_{c}" for c in reason_counts.columns]
    reason_counts = reason_counts[n_total > 0]

    work = optical.join(df[grid.columns])
    sub = work[masks["valid"]]
    g = sub.groupby(grouper)
    mean, std, count = g.mean(), g.std(), g.count()

    out = pd.DataFrame(index=reason_counts.index)
    out["n_valid"] = reason_counts["n_valid"]
    for c in reason_counts.columns:
        if c != "n_valid":
            out[c] = reason_counts[c]

    scalar_cols = [c for c in optical.columns]
    for c in scalar_cols:
        out[f"{c}_mean"] = mean.get(c)
        out[f"{c}_std"] = std.get(c)
    # PSD bins: window mean, NaN'd where too few samples contributed
    for col, dpg in zip(grid.columns, grid.dpg_um):
        m = mean.get(col)
        n = count.get(col)
        if m is None:
            out[psd_col_name(dpg)] = np.nan
        else:
            out[psd_col_name(dpg)] = m.where(n >= w.min_valid_points_per_bin)

    # Window-level QC flag
    flag = pd.Series(0, index=out.index, dtype=int)
    flag[out["n_valid"] < w.min_valid_points] |= FLAG_TOO_FEW_POINTS

    ae_mean, ae_std = out["AE_mean"], out["AE_std"]
    if w.ae_std_mode == "relative":
        ae_bad = (ae_std / ae_mean.abs()) > w.ae_max_relstd
    elif w.ae_std_mode == "absolute":
        ae_bad = ae_std > w.ae_max_relstd
    else:
        raise ValueError(f"Unknown ae_std_mode '{w.ae_std_mode}'")
    # Missing AE statistics cannot demonstrate stability -> unstable
    flag[ae_bad | ae_mean.isna() | ae_std.isna()] |= FLAG_AE_UNSTABLE

    needed = [f"Sc{wvl}_dry_mean" for wvl in cfg.channels.sca_suffixes]
    needed += [f"Abs{wvl}_mean" for wvl in cfg.channels.abs_suffixes]
    flag[out[needed].isna().any(axis=1)] |= FLAG_MISSING_OPTICS
    out["window_qc_flag"] = flag

    half = pd.Timedelta(seconds=w.window_s / 2)
    out["time_start"] = out.index
    out["time_end"] = out.index + 2 * half
    out.index = out.index + half
    out.index.name = "time"
    return out
