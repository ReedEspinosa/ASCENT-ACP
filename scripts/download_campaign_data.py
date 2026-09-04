#!/usr/bin/env python
"""Bulk-download airborne campaign data (ICARTT etc.) from NASA Earthdata/ASDC.

Searches NASA's Common Metadata Repository (CMR) for a campaign's collections
and downloads every granule to a local directory, skipping files already on
disk. Downloads authenticate through Earthdata Login using ~/.netrc:

  machine urs.earthdata.nasa.gov login <username> password <password>

(chmod 600). Listing/searching needs no credentials.

Usage:
  # See what collections a campaign has (project names: ACTIVATE, SEAC4RS,
  # KORUS-AQ, INTEXB, DC3, ARCTAS, NAAMES, CAMP2EX, FIREX-AQ, ...):
  python scripts/download_campaign_data.py list ACTIVATE

  # Download selected collections (substring match on collection short_name):
  python scripts/download_campaign_data.py fetch ACTIVATE \
      --collections Aerosol_AircraftInSitu Cloud_AircraftInSitu MetNav

  # Everything in a campaign, to a custom directory, excluding 2DS zips:
  python scripts/download_campaign_data.py fetch SEAC4RS --collections '' \
      --outdir /path/to/SEAC4RS --exclude 2DS --dry-run
"""
import argparse
import concurrent.futures
import csv
import io
import netrc
import os
import re
import sys
from pathlib import Path

import requests

CMR = "https://cmr.earthdata.nasa.gov/search"
DEFAULT_BASE = Path("~/Synced/ACMAP_Meloe/SuborbitalDataSets").expanduser()
PAGE_SIZE = 2000


def cmr_paged(session, endpoint, params):
    """Yield entries from a CMR search, following CMR-Search-After pagination."""
    headers = {}
    while True:
        r = session.get(f"{CMR}/{endpoint}.json", params=params, headers=headers,
                        timeout=60)
        r.raise_for_status()
        entries = r.json()["feed"]["entry"]
        if not entries:
            return
        yield from entries
        after = r.headers.get("CMR-Search-After")
        if not after or len(entries) < params["page_size"]:
            return
        headers["CMR-Search-After"] = after


def find_collections(session, project, patterns):
    cols = list(cmr_paged(session, "collections",
                          {"project": project, "page_size": PAGE_SIZE}))
    if patterns is None:
        return cols
    keep = []
    for c in cols:
        name = c.get("short_name", "")
        if any(p.lower() in name.lower() for p in patterns) or not patterns:
            keep.append(c)
    return keep


def granule_urls(session, concept_id, temporal=None):
    """Direct archive URLs for every granule in a collection.

    Uses CMR's CSV response, which omits the per-granule spatial metadata
    that makes the JSON response orders of magnitude slower for some
    collections (48 s vs 0.7 s for SEAC4RS aerosol).
    """
    urls, missing, headers = [], 0, {}
    params = {"collection_concept_id": concept_id, "page_size": PAGE_SIZE}
    if temporal:
        params["temporal"] = f"{temporal[0]}T00:00:00Z,{temporal[1]}T23:59:59Z"
    while True:
        r = session.get(f"{CMR}/granules.csv", params=params, headers=headers,
                        timeout=90)
        r.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(r.text)))
        if not rows:
            break
        for row in rows:
            best = None
            for u in (row.get("Online Access URLs") or "").split(","):
                u = u.strip()
                if (u.startswith("https") and "search.earthdata" not in u
                        and "cmr.earthdata" not in u):
                    best = u
                    break
            if best:
                urls.append(best)
            else:
                missing += 1
        after = r.headers.get("CMR-Search-After")
        if not after or len(rows) < PAGE_SIZE:
            break
        headers["CMR-Search-After"] = after
    return urls, missing


def check_netrc():
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if not auth or "YOUR_EARTHDATA" in (auth[0] or "") + (auth[2] or ""):
        sys.exit("Earthdata credentials not set: add to ~/.netrc (chmod 600):\n"
                 "  machine urs.earthdata.nasa.gov login <user> password <pass>")


def download_one(url, dest, clobber=False):
    """Fetch one file. Returns (status, dest) with status in
    {'ok', 'skipped', 'failed'}."""
    if dest.exists() and dest.stat().st_size > 0 and not clobber:
        return "skipped", dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Fresh session per worker thread: requests.Session is not thread-safe,
    # and each session picks up ~/.netrc for the Earthdata login redirect.
    session = requests.Session()
    try:
        with session.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            first = next(r.iter_content(chunk_size=65536), b"")
            if b"<html" in first[:512].lower() or "text/html" in ctype:
                raise RuntimeError(
                    "got an HTML page instead of data (bad Earthdata "
                    "credentials or unapproved application?)")
            with open(tmp, "wb") as f:
                f.write(first)
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
        return "ok", dest
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"  FAILED {dest.name}: {e}", file=sys.stderr)
        return "failed", dest


def unzip_all(outdir, jobs):
    """Extract downloaded .zip granules (FCDP, 2DS, ...) into outdir,
    skipping members already on disk."""
    import zipfile
    n = 0
    for u in jobs:
        path = outdir / os.path.basename(u)
        if path.suffix != ".zip" or not path.exists():
            continue
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                name = os.path.basename(m.filename)  # flatten, drop dirs
                if m.is_dir() or not name or (outdir / name).exists():
                    continue
                with z.open(m) as src, open(outdir / name, "wb") as dst:
                    dst.write(src.read())
                n += 1
    print(f"unzipped {n} new files")


def cmd_list(args):
    session = requests.Session()
    cols = find_collections(session, args.project, None)
    if not cols:
        sys.exit(f"No CMR collections for project={args.project!r} "
                 "(try another spelling, e.g. KORUS-AQ vs KORUSAQ, INTEXB)")
    print(f"{len(cols)} collections for project={args.project}:")
    for c in sorted(cols, key=lambda c: c.get("short_name", "")):
        r = session.get(f"{CMR}/granules.json",
                        params={"collection_concept_id": c["id"], "page_size": 1},
                        timeout=60)
        hits = r.headers.get("CMR-Hits", "?")
        print(f"  {c.get('short_name'):55s} {hits:>6s} granules  "
              f"[{c.get('data_center')}]")


def cmd_fetch(args):
    session = requests.Session()
    cols = find_collections(session, args.project, args.collections)
    if not cols:
        sys.exit(f"No collections matched {args.collections} for "
                 f"project={args.project!r} -- run the 'list' command first.")
    outdir = Path(args.outdir) if args.outdir else DEFAULT_BASE / args.project
    print(f"{len(cols)} collections -> {outdir}")

    jobs = []
    for c in cols:
        urls, missing = granule_urls(session, c["id"], args.temporal)
        if missing:
            print(f"  WARNING {c.get('short_name')}: {missing} granules have "
                  "no direct download link", file=sys.stderr)
        for u in urls:
            if not (args.exclude and re.search(args.exclude,
                                               os.path.basename(u))):
                jobs.append(u)
        print(f"  {c.get('short_name')}: {len(urls)} granules")

    # Same file can appear under several collections; keep one copy.
    jobs = sorted(set(jobs))
    print(f"{len(jobs)} unique files")
    if args.dry_run:
        for u in jobs[:20]:
            print("   ", os.path.basename(u))
        if len(jobs) > 20:
            print(f"    ... and {len(jobs) - 20} more")
        return

    check_netrc()
    outdir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download_one, u,
                               outdir / os.path.basename(u), args.clobber)
                   for u in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            status, dest = fut.result()
            counts[status] += 1
            if status == "ok":
                print(f"  [{i}/{len(jobs)}] {dest.name}")
    print(f"done: {counts['ok']} downloaded, {counts['skipped']} already "
          f"present, {counts['failed']} failed")
    if args.unzip:
        unzip_all(outdir, jobs)
    if counts["failed"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list a campaign's CMR collections")
    p.add_argument("project", help="CMR project name, e.g. ACTIVATE, SEAC4RS")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("fetch", help="download granules from matching collections")
    p.add_argument("project")
    p.add_argument("--collections", nargs="*", default=["AircraftInSitu"],
                   help="substring filters on collection short_name "
                        "(default: AircraftInSitu; pass '' for everything)")
    p.add_argument("--outdir", help=f"default: {DEFAULT_BASE}/<PROJECT>")
    p.add_argument("--exclude", help="regex on filename to skip (e.g. 2DS)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--clobber", action="store_true",
                   help="re-download files already on disk")
    p.add_argument("--temporal", nargs=2, metavar=("START", "END"),
                   help="granule date range, e.g. --temporal 2021-11-01 2022-12-31")
    p.add_argument("--unzip", action="store_true",
                   help="extract .zip granules (FCDP, 2DS, ...) after download")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be downloaded, then stop")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
