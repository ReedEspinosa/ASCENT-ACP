"""Merge the full ACTIVATE ICARTT dataset into per-year V2 pickles.

Memory-safe variant of the ACTIVATE_TEST merge: a year-long 1 s master
timeline would require >100 GB during align2master_timeline, so each
flight date is merged on its own one-day timeline and the (already
dropna'd) per-date frames are row-concatenated. Flights never span
midnight UTC, so the result matches a single whole-year merge.

Instrument subset replicates ACTIVATE_TEST (notably excluding
DLH-H2O-20Hz, whose sub-second data breaks align2master_timeline).
"""

import argparse
import os
import pickle
import re
import sys
import time
import traceback

import pandas as pd

import icartt_read_and_merge as ict
from icartt_read_and_merge.icartt_read_and_merge import _merge_meta_dicts

SOURCE_DIR = "/Users/wrespino/Synced/ACMAP_Meloe/SuborbitalDataSets/ACTIVATE"
OUTPUT_DIR = SOURCE_DIR
STAGING_DIR = "/tmp/activate_merge_staging"

# Same instrument set as the ACTIVATE_TEST merge (both filename case variants)
INSTRUMENTS = [
    "ACTIVATE-DLH-H2O",  # exact match below keeps DLH-H2O-20Hz/20HZ out
    "ACTIVATE-FCDP",
    "ACTIVATE-LARGE-AMS",
    "ACTIVATE-LARGE-AMS-CVI",
    "ACTIVATE-LARGE-CAS",
    "ACTIVATE-LARGE-CCN",
    "ACTIVATE-LARGE-CDP",
    "ACTIVATE-LARGE-InletFlag",
    "ACTIVATE-LARGE-INLETFLAG",
    "ACTIVATE-LARGE-LAS",
    "ACTIVATE-LARGE-OPTICAL",
    "ACTIVATE-LARGE-PILS",
    "ACTIVATE-LARGE-SMPS",
    "ACTIVATE-SUMMARY",
    "ACTIVATE-TraceGas-CH4",
    "ACTIVATE-TraceGas-CO",
    "ACTIVATE-TraceGas-CO2",
    "ACTIVATE-TraceGas-O3",
    "ACTIVATE-TRACEGAS-CH4",
    "ACTIVATE-TRACEGAS-CO",
    "ACTIVATE-TRACEGAS-CO2",
    "ACTIVATE-TRACEGAS-O3",
]
FILE_RE = re.compile(
    r"^(?P<instr>" + "|".join(re.escape(i) for i in INSTRUMENTS) + r")"
    r"_HU25_(?P<date>\d{8})_R\d+(_L\d)?\.ict$"
)


def stage_files(years):
    """Symlink the instrument subset for the requested years; return date list per year."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    for f in os.listdir(STAGING_DIR):
        os.unlink(os.path.join(STAGING_DIR, f))
    dates_by_year = {y: set() for y in years}
    n_staged = 0
    for fname in sorted(os.listdir(SOURCE_DIR)):
        m = FILE_RE.match(fname)
        if not m:
            continue
        year = m.group("date")[:4]
        if year not in years:
            continue
        os.symlink(os.path.join(SOURCE_DIR, fname), os.path.join(STAGING_DIR, fname))
        dates_by_year[year].add(m.group("date"))
        n_staged += 1
    print(f"Staged {n_staged} files into {STAGING_DIR}")
    for y in years:
        print(f"  {y}: {len(dates_by_year[y])} flight dates")
    return {y: sorted(d) for y, d in dates_by_year.items()}


def merge_year(year, dates, n_workers):
    frames, metas, failed = [], [], []
    for i, d in enumerate(dates, 1):
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        t0 = time.time()
        print(f"\n##### [{year}] {iso} ({i}/{len(dates)}) #####", flush=True)
        try:
            df, meta = ict.icartt_merger(
                icartt_directory=STAGING_DIR,
                mode_input="Merge_Beside",
                master_timeline=[f"{iso} 00:00:00", f"{iso} 23:59:59", 1],
                pickle_directory=None,
                prefix_instr_name=True,
                n_workers=n_workers,
            )
        except Exception:
            print(f"!!!!! [{year}] {iso} FAILED:\n{traceback.format_exc()}", flush=True)
            failed.append(iso)
            continue
        frames.append(df)
        metas.append(meta)
        print(f"##### [{year}] {iso} done: {df.shape[0]} rows x {df.shape[1]} cols "
              f"in {time.time() - t0:.0f} s", flush=True)

    if not frames:
        raise RuntimeError(f"No dates merged successfully for {year}")

    df_year = pd.concat(frames, axis=0)
    del frames
    df_year = df_year.sort_index()
    n_dup = int(df_year.index.duplicated().sum())
    if n_dup:
        print(f"WARNING [{year}]: {n_dup} duplicate timestamps after concat; keeping first")
        df_year = df_year[~df_year.index.duplicated(keep="first")]
    meta_year = _merge_meta_dicts(metas)

    base = os.path.join(OUTPUT_DIR, f"merged1sec_allInstruments_{year}_V2")
    df_year.to_pickle(base + ".pkl")
    with open(base + "_meta.pickle", "wb") as f:
        pickle.dump(meta_year, f)
    print(f"\n===== [{year}] saved {base}.pkl: "
          f"{df_year.shape[0]} rows x {df_year.shape[1]} cols, "
          f"{df_year.index.min()} .. {df_year.index.max()}")
    if failed:
        print(f"===== [{year}] FAILED dates: {failed}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=["2020", "2021"])
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--dates", nargs="*", default=None,
                    help="restrict to specific YYYYMMDD dates (debug)")
    args = ap.parse_args()

    dates_by_year = stage_files(args.years)
    all_failed = {}
    for y in args.years:
        dates = dates_by_year[y]
        if args.dates:
            dates = [d for d in dates if d in set(args.dates)]
        if not dates:
            print(f"[{y}] no dates selected, skipping")
            continue
        all_failed[y] = merge_year(y, dates, args.n_workers)

    print("\n===== MERGE COMPLETE =====")
    for y, failed in all_failed.items():
        print(f"  {y}: {'all dates OK' if not failed else f'FAILED dates: {failed}'}")
    sys.exit(1 if any(all_failed.values()) else 0)


if __name__ == "__main__":
    main()
