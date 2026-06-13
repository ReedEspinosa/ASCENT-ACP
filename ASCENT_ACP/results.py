"""Assemble window-level measurements and ISARA retrievals into one table."""

import pickle
from pathlib import Path

import numpy as np

# retrieval_qc_flag values (per window)
RETRIEVAL_OK = 0          # ISARA ran and CRI succeeded
RETRIEVAL_SKIPPED_QA = 1  # not attempted: window failed QA (window_qc_flag != 0)
RETRIEVAL_FAILED = 2      # attempted (window passed QA) but ISARA did not succeed

RETRIEVAL_QC_MEANINGS = {
    RETRIEVAL_OK: "retrieved",
    RETRIEVAL_SKIPPED_QA: "not_attempted_failed_window_qc",
    RETRIEVAL_FAILED: "attempted_but_retrieval_failed",
}


def add_retrieval_qc_flag(results_df):
    """Add ``retrieval_qc_flag`` summarizing why each window has/has-no retrieval.

    Keyed off ``window_qc_flag`` (the QA gate) and ``attempt_flag_CRI_unitless``
    (== 2 on success). Windows failing QA were never attempted (flag 1); windows
    passing QA are OK (0) if CRI succeeded else failed (2).
    """
    n = len(results_df)
    passed = results_df.get("window_qc_flag")
    passed = (passed == 0).to_numpy() if passed is not None else np.zeros(n, bool)
    cri = results_df.get("attempt_flag_CRI_unitless")
    cri_ok = (cri == 2).to_numpy() if cri is not None else np.zeros(n, bool)

    flag = np.full(n, RETRIEVAL_SKIPPED_QA, dtype=int)
    flag[passed & cri_ok] = RETRIEVAL_OK
    flag[passed & ~cri_ok] = RETRIEVAL_FAILED
    results_df["retrieval_qc_flag"] = flag
    return results_df


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
    add_retrieval_qc_flag(results)
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
