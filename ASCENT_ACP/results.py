"""Assemble window-level measurements and ISARA retrievals into one table."""

import pickle
from pathlib import Path


def assemble(windows_df, retrievals_df, grid, cfg):
    """Outer-join retrieval outputs onto the windowed measurements.

    Windows that were rejected by QC (or whose retrieval was skipped) keep
    their measured columns with NaN retrievals, so the output preserves the
    full flight timeline and the QC bookkeeping.
    """
    if retrievals_df.empty:
        results = windows_df.copy()
    else:
        results = windows_df.join(retrievals_df, how="left")
    return results


def save_checkpoint(results_df, grid, cfg, path):
    """Pickle the full result bundle (the pre-netCDF checkpoint)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "results": results_df,
        "grid": grid,
        "config_json": cfg.to_json(),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path
