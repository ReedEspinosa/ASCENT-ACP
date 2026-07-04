"""End-to-end ASCENT-ACP pipeline driver.

merged pickle -> row QC -> window averaging -> ISARA retrieval -> netCDF.

CLI::

    python -m ASCENT_ACP.pipeline --config configs/activate_2021.json \
        [--dates 2021-05-13 ...] [--max-windows 20] [--no-netcdf] [--plots]
"""

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

from . import filtering, isara_bridge, netcdf_export, results, sizebins, windows
from .config import PipelineConfig


def load_inputs(cfg):
    df = pd.read_pickle(cfg.paths.input_pkl)
    meta = None
    if cfg.paths.meta_pickle and Path(cfg.paths.meta_pickle).exists():
        with open(cfg.paths.meta_pickle, "rb") as f:
            meta = pickle.load(f)
    return df, meta


def run_pipeline(cfg, dates=None, max_windows=None, write_nc=True, make_plots=False):
    """Run the full chain; returns (results_df, grid, output paths dict)."""
    t0 = time.time()
    print(f"[1/5] Loading {cfg.paths.input_pkl}")
    df, meta = load_inputs(cfg)
    if dates:
        keep = df.index.normalize().isin(pd.to_datetime(list(dates), utc=True))
        df = df[keep]
        print(f"      date filter {sorted(dates)}: {len(df)} rows remain")
    if df.empty:
        raise ValueError("No rows to process after date selection")

    print("[2/5] Row-level QC and RH adjustment")
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    print(
        f"      {int(masks['valid'].sum())}/{len(masks)} 1 Hz rows valid "
        f"(cloudy {int(masks['cloudy'].sum())}, inlet {int(masks['inlet_bad'].sum())}, "
        f"low-signal {int(masks['low_signal'].sum())}, low-SSA {int(masks['low_ssa'].sum())})"
    )

    print(f"[3/5] Averaging into {cfg.window.window_s} s windows")
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    n_good = int((wdf["window_qc_flag"] == 0).sum())
    print(f"      {n_good}/{len(wdf)} windows pass QC")
    if max_windows is not None and n_good > max_windows:
        good_idx = wdf.index[wdf["window_qc_flag"] == 0][max_windows:]
        wdf = wdf.drop(index=good_idx)
        print(f"      --max-windows: retrieving first {max_windows} good windows only")

    print(f"[4/5] ISARA retrieval ({cfg.isara.n_workers} workers)")
    retr = isara_bridge.run_all_windows(wdf, grid, cfg)
    if not retr.empty:
        n_cri = int((retr.get("attempt_flag_CRI_unitless") == 2).sum())
        n_kap = int((retr.get("attempt_flag_kappa_unitless") == 2).sum())
        print(f"      CRI success {n_cri}/{len(retr)}, kappa success {n_kap}/{len(retr)}")

    res = results.assemble(wdf, retr, grid, cfg)
    out_dir = Path(cfg.paths.output_dir)
    paths = {}
    stem = netcdf_export.output_filename(cfg).removesuffix(".nc")
    paths["checkpoint"] = results.save_checkpoint(res, grid, cfg, out_dir / f"{stem}.pkl")

    if write_nc:
        print("[5/5] Writing grouped netCDF")
        paths["netcdf"] = netcdf_export.export(df, masks, res, grid, cfg, meta=meta,
                                               path=out_dir / f"{stem}.nc")
        print(f"      {paths['netcdf']}")
    if make_plots:
        from . import plots

        paths["plots"] = plots.make_all(res, grid, cfg, out_dir / f"plots_{stem}")
    print(f"Done in {time.time() - t0:.0f} s")
    return res, grid, paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the ASCENT-ACP ISARA pipeline")
    ap.add_argument("--config", required=True, help="path to pipeline JSON config")
    ap.add_argument("--dates", nargs="*", help="restrict to flight dates (YYYY-MM-DD)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="retrieve at most this many QC-passing windows (debug)")
    ap.add_argument("--no-netcdf", action="store_true", help="skip the netCDF export")
    ap.add_argument("--plots", action="store_true", help="produce sanity plots")
    args = ap.parse_args(argv)

    cfg = PipelineConfig.from_json(args.config)
    run_pipeline(
        cfg,
        dates=args.dates,
        max_windows=args.max_windows,
        write_nc=not args.no_netcdf,
        make_plots=args.plots,
    )


if __name__ == "__main__":
    main()
