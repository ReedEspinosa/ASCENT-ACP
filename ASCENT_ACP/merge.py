"""Campaign-agnostic ICARTT -> merged-pickle stage.

Config-driven generalization of the former ``run_full_activate_merge.py``
wrapper: instrument set, filename pattern, source directory and timeline spacing
all come from ``MergeConfig``. The merge engine
(``icartt_read_and_merge.icartt_merger``) is called unchanged.

Memory-safe by construction: a year-long native-cadence master timeline would
exhaust RAM, so each flight date is merged on its own one-day timeline and the
(already dropna'd) per-date frames are row-concatenated. Flights are assumed not
to cross UTC midnight (a warning is logged otherwise).
"""

import argparse
import os
import pickle
import re
import time
import traceback
from pathlib import Path

import pandas as pd

import icartt_read_and_merge as ict
from icartt_read_and_merge.icartt_read_and_merge import _merge_meta_dicts

from .config import PipelineConfig


def merged_pickle_path(cfg):
    """Pre-shift merged pickle path: input_pkl with any '_timeShifted' removed."""
    p = Path(cfg.paths.input_pkl)
    return p.with_name(p.name.replace("_timeShifted", ""))


def _staging_dir(cfg):
    return cfg.merge.staging_dir or f"/tmp/{cfg.campaign}_merge_staging"


def stage_files(cfg, year):
    """Symlink the configured instruments for ``year`` into staging; return dates.

    Files are matched by ``cfg.merge.filename_regex`` (named groups ``instr``,
    ``date``). A file is kept only if ``instr`` is in ``cfg.merge.instruments``
    (empty list = keep all) and ``date`` parses to the requested year.
    """
    staging = _staging_dir(cfg)
    os.makedirs(staging, exist_ok=True)
    for f in os.listdir(staging):
        os.unlink(os.path.join(staging, f))

    regex = re.compile(cfg.merge.filename_regex)
    keep_instr = set(cfg.merge.instruments)
    dates, n = set(), 0
    for fname in sorted(os.listdir(cfg.merge.icartt_dir)):
        m = regex.match(fname)
        if not m:
            continue
        if keep_instr and m.group("instr") not in keep_instr:
            continue
        dt = pd.to_datetime(m.group("date"), format=cfg.merge.date_format)
        if f"{dt.year}" != str(year):
            continue
        os.symlink(os.path.join(cfg.merge.icartt_dir, fname),
                   os.path.join(staging, fname))
        dates.add(dt.strftime("%Y-%m-%d"))
        n += 1
    print(f"Staged {n} files for {year} into {staging} ({len(dates)} flight dates)")
    return sorted(dates), staging


def run_merge(cfg, dates=None):
    """Merge ``cfg.year`` ICARTT files to the pre-shift pickle; return its path.

    ``dates`` (optional list of YYYY-MM-DD) restricts which flight dates to merge.
    Writes ``<merged>.pkl`` and ``<meta_pickle>``.
    """
    year = cfg.year
    all_dates, staging = stage_files(cfg, year)
    if dates:
        sel = set(dates)
        all_dates = [d for d in all_dates if d in sel]
    if not all_dates:
        raise RuntimeError(f"No flight dates to merge for {year}")

    step = cfg.merge.master_timeline_step_s
    exclude = [re.compile(p) for p in cfg.merge.exclude_regexes]
    frames, metas, failed = [], [], []
    for i, iso in enumerate(all_dates, 1):
        t0 = time.time()
        print(f"\n##### [{year}] {iso} ({i}/{len(all_dates)}) #####", flush=True)
        try:
            df, meta = ict.icartt_merger(
                icartt_directory=staging,
                mode_input=cfg.merge.merge_mode,
                master_timeline=[f"{iso} 00:00:00", f"{iso} 23:59:59", step],
                pickle_directory=None,
                prefix_instr_name=cfg.merge.prefix_instr_name,
                n_workers=cfg.merge.n_workers,
            )
        except Exception:
            print(f"!!!!! [{year}] {iso} FAILED:\n{traceback.format_exc()}", flush=True)
            failed.append(iso)
            continue
        frames.append(df)
        metas.append(meta)
        print(f"##### [{year}] {iso} done: {df.shape[0]} x {df.shape[1]} "
              f"in {time.time() - t0:.0f} s", flush=True)

    if not frames:
        raise RuntimeError(f"No dates merged successfully for {year}")

    df_year = pd.concat(frames, axis=0).sort_index()
    del frames
    n_dup = int(df_year.index.duplicated().sum())
    if n_dup:
        print(f"WARNING [{year}]: {n_dup} duplicate timestamps after concat; keeping first")
        df_year = df_year[~df_year.index.duplicated(keep="first")]
    if exclude:
        drop = [c for c in df_year.columns if any(p.search(c) for p in exclude)]
        if drop:
            print(f"[{year}] dropping {len(drop)} excluded columns")
            df_year = df_year.drop(columns=drop)
    meta_year = _merge_meta_dicts(metas)

    out_pkl = merged_pickle_path(cfg)
    out_meta = Path(cfg.paths.meta_pickle)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    df_year.to_pickle(out_pkl)
    with open(out_meta, "wb") as f:
        pickle.dump(meta_year, f)
    print(f"\n===== [{year}] saved {out_pkl}: {df_year.shape[0]} x {df_year.shape[1]}, "
          f"{df_year.index.min()} .. {df_year.index.max()}")
    if failed:
        print(f"===== [{year}] FAILED dates: {failed}")
    return out_pkl


def main(argv=None):
    ap = argparse.ArgumentParser(description="ICARTT -> merged pickle (campaign-agnostic)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dates", nargs="*", default=None, help="restrict to YYYY-MM-DD dates")
    args = ap.parse_args(argv)
    cfg = PipelineConfig.from_json(args.config)
    run_merge(cfg, dates=args.dates)


if __name__ == "__main__":
    main()
