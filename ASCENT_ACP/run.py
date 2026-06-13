"""Single-pass, campaign-agnostic driver: ICARTT -> grouped netCDF.

Chains the four stages and checkpoints to pickle between each, so a crash
resumes without redoing finished work::

    merge     ICARTT files          -> <merged>.pkl + <meta>.pickle
    align     <merged>.pkl          -> <merged>_timeShifted.pkl + shift_diagnostics.csv
    retrieve  <timeShifted>.pkl     -> <stem>.pkl   (windows + QC + ISARA bundle)
    export    bundle + df + meta    -> <stem>.nc    (grouped CF-1.8 netCDF)

Each stage skips when its checkpoint already exists, unless ``--force`` or
``--from-stage`` names it (or an earlier stage). The merge engine and
``apply_clock_alignment.py`` are invoked unchanged; only orchestration lives
here.

CLI::

    python -m ASCENT_ACP.run --config configs/activate_2021_full.json
        [--dates YYYY-MM-DD ...] [--from-stage merge|align|retrieve|export]
        [--force] [--max-windows N] [--no-plots]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import filtering, isara_bridge, merge, netcdf_export, pipeline, results, sizebins, windows
from .config import PipelineConfig

STAGES = ["merge", "align", "retrieve", "export"]
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _stem(cfg):
    return netcdf_export.output_filename(cfg).removesuffix(".nc")


def _paths(cfg):
    out_dir = Path(cfg.paths.output_dir)
    return {
        "merged": merge.merged_pickle_path(cfg),
        "meta": Path(cfg.paths.meta_pickle),
        "shifted": Path(cfg.paths.input_pkl),
        "bundle": out_dir / f"{_stem(cfg)}.pkl",
        "netcdf": out_dir / f"{_stem(cfg)}.nc",
    }


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_merge(cfg, dates):
    print(f"\n===== STAGE merge ({cfg.campaign} {cfg.year}) =====", flush=True)
    merge.run_merge(cfg, dates=dates)


def stage_align(cfg, make_plots):
    print(f"\n===== STAGE align ({cfg.campaign} {cfg.year}) =====", flush=True)
    p = _paths(cfg)
    cmd = [
        sys.executable, "apply_clock_alignment.py", str(p["merged"]),
        "--output-pkl", str(p["shifted"]),
    ]
    if cfg.paths.shift_table_csv:
        cmd += ["--shift-table", cfg.paths.shift_table_csv]
    if cfg.paths.shift_diagnostics_csv:
        cmd += ["--diagnostics-csv", str((_REPO_ROOT / cfg.paths.shift_diagnostics_csv))]
    if not make_plots:
        cmd += ["--plot-dir", ""]
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def stage_retrieve(cfg, dates, max_windows):
    """Run row QC -> windowing -> ISARA; save the results bundle. Return state."""
    print(f"\n===== STAGE retrieve ({cfg.campaign} {cfg.year}) =====", flush=True)
    df, meta = pipeline.load_inputs(cfg)
    if dates:
        import pandas as pd
        keep = df.index.normalize().isin(pd.to_datetime(list(dates), utc=True))
        df = df[keep]
    if df.empty:
        raise ValueError("No rows to process after date selection")

    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    n_good = int((wdf["window_qc_flag"] == 0).sum())
    print(f"  {n_good}/{len(wdf)} windows pass QC", flush=True)
    if max_windows is not None and n_good > max_windows:
        drop = wdf.index[wdf["window_qc_flag"] == 0][max_windows:]
        wdf = wdf.drop(index=drop)

    retr = isara_bridge.run_all_windows(wdf, grid, cfg)
    res = results.assemble(wdf, retr, grid, cfg)
    bundle = _paths(cfg)["bundle"]
    results.save_checkpoint(res, grid, cfg, bundle)
    print(f"  saved bundle {bundle}", flush=True)
    return {"res": res, "grid": grid, "df": df, "masks": masks, "meta": meta}


def stage_export(cfg, state=None):
    """Build and write the grouped netCDF, reloading from disk if needed."""
    print(f"\n===== STAGE export ({cfg.campaign} {cfg.year}) =====", flush=True)
    if state is None:
        import pickle
        bundle = _paths(cfg)["bundle"]
        with open(bundle, "rb") as f:
            b = pickle.load(f)
        res, grid = b["results"], b["grid"]
        df, meta = pipeline.load_inputs(cfg)
        optical = filtering.derive_optical_columns(df, cfg)
        masks = filtering.row_qc(df, optical, cfg)
    else:
        res, grid, df, masks, meta = (state[k] for k in ("res", "grid", "df", "masks", "meta"))

    dt = netcdf_export.build_datatree(df, masks, res, grid, cfg, meta=meta)
    out = netcdf_export.write(dt, _paths(cfg)["netcdf"], cfg)
    print(f"  wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(cfg, dates=None, from_stage=None, force=False, max_windows=None, make_plots=True):
    t0 = time.time()
    p = _paths(cfg)
    exists = {
        "merge": p["merged"].exists() and p["meta"].exists(),
        "align": p["shifted"].exists(),
        "retrieve": p["bundle"].exists(),
        "export": p["netcdf"].exists(),
    }
    from_idx = STAGES.index(from_stage) if from_stage else None
    ran_earlier = False
    state = None

    for idx, stage in enumerate(STAGES):
        should = force or ran_earlier or (from_idx is not None and idx >= from_idx) or not exists[stage]
        if not should:
            print(f"----- skip {stage} (checkpoint exists: {p[_ck(stage)]}) -----", flush=True)
            continue
        ran_earlier = True
        if stage == "merge":
            stage_merge(cfg, dates)
        elif stage == "align":
            stage_align(cfg, make_plots)
        elif stage == "retrieve":
            state = stage_retrieve(cfg, dates, max_windows)
        elif stage == "export":
            stage_export(cfg, state)

    print(f"\n===== {cfg.campaign} {cfg.year} done in {time.time() - t0:.0f} s =====", flush=True)


def _ck(stage):
    return {"merge": "merged", "align": "shifted", "retrieve": "bundle", "export": "netcdf"}[stage]


def main(argv=None):
    ap = argparse.ArgumentParser(description="ASCENT-ACP single-pass driver (ICARTT -> netCDF)")
    ap.add_argument("--config", required=True, help="path to pipeline JSON config")
    ap.add_argument("--dates", nargs="*", default=None, help="restrict to flight dates (YYYY-MM-DD)")
    ap.add_argument("--from-stage", choices=STAGES, default=None,
                    help="force re-run from this stage onward")
    ap.add_argument("--force", action="store_true", help="re-run all stages")
    ap.add_argument("--max-windows", type=int, default=None, help="retrieve at most N good windows (debug)")
    ap.add_argument("--no-plots", action="store_true", help="skip clock-alignment plots")
    args = ap.parse_args(argv)

    cfg = PipelineConfig.from_json(args.config)
    run(cfg, dates=args.dates, from_stage=args.from_stage, force=args.force,
        max_windows=args.max_windows, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
