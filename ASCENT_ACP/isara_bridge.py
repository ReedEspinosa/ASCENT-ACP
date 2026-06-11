"""Bridge between windowed ASCENT-ACP data and the ISARA retrieval library.

ISARA (sibling repo, not pip-installed) is imported by path from
``cfg.paths.isara_code_dir``. Each surviving window becomes one call to
``ISARA.Retr_PSD`` (a MOPSMAP grid search over refractive index and kappa);
windows are distributed over a process pool.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .windows import psd_col_name

# Set per-process by _worker_init (workers) or import_isara (serial path)
_ISARA = None


def import_isara(isara_code_dir):
    """Import the ISARA module from a sibling checkout; returns the module."""
    global _ISARA
    if isara_code_dir not in sys.path:
        sys.path.insert(0, isara_code_dir)
    import ISARA  # noqa: PLC0415

    _ISARA = ISARA
    return ISARA


def build_retr_kwargs(row, grid, cfg):
    """Translate one windows-DataFrame row into Retr_PSD keyword arguments.

    Scattering/absorption means are converted Mm^-1 -> m^-1; bin centers
    become radii. NaN PSD bins are passed through (Retr_PSD drops them).
    """
    ch = cfg.channels
    dndlogdp = np.array([row[psd_col_name(d)] for d in grid.dpg_um], dtype=float)
    kwargs = {
        "radii_um": grid.dpg_um / 2.0,
        "dndlogdp_cm3": dndlogdp,
        "dry_sca_coef": np.array(
            [row[f"Sc{w}_dry_mean"] for w in ch.dry_wvl_sca], dtype=float
        )
        * 1e-6,
        "dry_abs_coef": np.array(
            [row[f"Abs{w}_mean"] for w in ch.dry_wvl_abs], dtype=float
        )
        * 1e-6,
        "dry_wvl": {"sca": list(ch.dry_wvl_sca), "abs": list(ch.dry_wvl_abs)},
        "wet_sca_coef": np.array(
            [row[f"Sc{w}_wet_mean"] for w in ch.wet_wvl_sca], dtype=float
        )
        * 1e-6,
        "wet_wvl": {"sca": list(ch.wet_wvl_sca)},
        "RH_wet": cfg.filters.wet_rh,
        "val_wvl": np.array(ch.val_wvl) if ch.val_wvl else None,
        "size_equ": cfg.isara.size_equ,
        "nonabs_fraction": cfg.isara.nonabs_fraction,
        "shape": cfg.isara.shape,
        "rho_dry": cfg.isara.rho_dry,
        "rho_wet": cfg.isara.rho_wet,
        "num_theta": cfg.isara.num_theta,
        "path_optical_dataset": cfg.paths.optical_dataset_dir,
        "path_mopsmap_executable": cfg.paths.mopsmap_executable,
    }
    return kwargs


def _retrieve_one(item):
    """Run one retrieval; never raises (failures become attempt flags of 0)."""
    timestamp, kwargs = item
    try:
        result = _ISARA.Retr_PSD(**kwargs)
    except ValueError as err:  # e.g. <2 valid PSD bins
        result = {
            "attempt_flag_CRI_unitless": 0,
            "attempt_flag_kappa_unitless": 0,
            "retrieval_error": str(err),
        }
    return timestamp, result


def _worker_init(isara_code_dir, scratch_dir):
    """Per-worker setup: cwd for MOPSMAP temp files and a unique RNG state.

    mopsmap_wrapper names its temp files from time.time() and np.random.randn();
    forked workers inherit identical RNG state, so reseed per PID to avoid
    temp-file collisions.
    """
    os.makedirs(scratch_dir, exist_ok=True)
    os.chdir(scratch_dir)
    np.random.seed(os.getpid() & 0xFFFFFFFF)
    import_isara(isara_code_dir)


def run_all_windows(windows_df, grid, cfg, progress=True):
    """Retrieve every window with window_qc_flag == 0.

    Returns a DataFrame of Retr_PSD outputs indexed by window center time
    (only for attempted windows; join back onto windows_df afterwards).
    """
    good = windows_df[windows_df["window_qc_flag"] == 0]
    if good.empty:
        return pd.DataFrame()
    items = [(ts, build_retr_kwargs(row, grid, cfg)) for ts, row in good.iterrows()]

    results = {}
    n_workers = cfg.isara.n_workers
    if n_workers <= 1:
        import_isara(cfg.paths.isara_code_dir)
        os.makedirs(cfg.paths.scratch_dir, exist_ok=True)
        prev_cwd = os.getcwd()
        os.chdir(cfg.paths.scratch_dir)
        try:
            for i, item in enumerate(items):
                ts, res = _retrieve_one(item)
                results[ts] = res
                if progress and (i + 1) % 10 == 0:
                    print(f"  retrieved {i + 1}/{len(items)} windows")
        finally:
            os.chdir(prev_cwd)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(cfg.paths.isara_code_dir, cfg.paths.scratch_dir),
        ) as pool:
            for i, (ts, res) in enumerate(pool.map(_retrieve_one, items, chunksize=1)):
                results[ts] = res
                if progress and (i + 1) % 25 == 0:
                    print(f"  retrieved {i + 1}/{len(items)} windows")

    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "time"
    return out
