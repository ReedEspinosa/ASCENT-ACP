"""Generic ICARTT (FFI 1001) header scanning for the netCDF export.

Reads one header per instrument token from the source .ict directory and
exposes three things the exporter needs but the merged pickle lost:

* per-variable metadata (units, standard-name token, description) and the
  STP-vs-ambient measurement basis encoded in them,
* size-distribution bin tables (centers and/or edges) for bin-column
  compaction, parsed from the variable descriptions or OTHER_COMMENTS,
* per-flight time envelopes (first/last data time of each file; _L1/_L2
  files are separate flights) used to build the (flight x time) grid.

Everything is best-effort: a missing directory, file, or comment simply
yields ``None``/empty results and the exporter degrades gracefully.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Mirrors icartt_read_and_merge.char_cleaner so cleaned titles match the
# merged-pickle column prefixes exactly.
_BAD_CHARS = [' ', ',', '.', '"', '*', '!', '@', '#', '$', '^', '&',
              '(', ')', '=', '?', '/', '\\', ':', ';', '~', '`', '<',
              '>', ']', '[', '{', '}']


def clean_title(text):
    after = text.strip()
    after = after.replace('%', 'percent')
    after = after.replace('_+_', '+')
    after = after.replace('-->', 'to')
    for ch in _BAD_CHARS:
        after = after.replace(ch, '_')
    return after


@dataclass
class VarInfo:
    name: str
    units: str = ""
    standard: str = ""       # ICARTT standardized-name token (3rd field)
    description: str = ""


@dataclass
class Header:
    path: str
    token: str = ""          # instrument token from the filename
    title: str = ""          # raw line-4 description
    title_clean: str = ""    # char-cleaned title == merged column prefix
    variables: list = field(default_factory=list)  # [VarInfo]
    data_info: str = ""
    other_comments: str = ""
    n_header: int = 0

    def var(self, name):
        for v in self.variables:
            if v.name == name:
                return v
        return None


@dataclass
class BinTable:
    """Ordered size-distribution bins of one instrument."""

    columns: list            # short variable names, ascending bin order
    center_um: np.ndarray
    lower_um: np.ndarray
    upper_um: np.ndarray
    source: str              # where the sizes came from


def parse_header(path):
    """Parse the header block of one FFI-1001 ICARTT file."""
    path = Path(path)
    with open(path, "r", errors="replace") as f:
        first = f.readline()
        try:
            n_header = int(first.split(",")[0])
        except ValueError:
            return None
        lines = [first.rstrip("\n")]
        for _ in range(n_header - 1):
            lines.append(f.readline().rstrip("\n"))

    hdr = Header(path=str(path), n_header=n_header)
    hdr.title = lines[3].strip() if len(lines) > 3 else ""
    hdr.title_clean = clean_title(hdr.title)

    # dependent-variable lines: count on line 10 (idx 9), definitions start
    # after the scale-factor (idx 10) and missing-value (idx 11) lines.
    try:
        n_dep = int(lines[9].split(",")[0])
        for ln in lines[12:12 + n_dep]:
            parts = [p.strip() for p in ln.split(",", 3)]
            parts += [""] * (4 - len(parts))
            hdr.variables.append(VarInfo(*parts[:4]))
    except (IndexError, ValueError):
        pass

    # keyed special comments; OTHER_COMMENTS may continue over several lines
    key_re = re.compile(r"^([A-Z_]{4,})\s*:")
    current = None
    for ln in lines[12 + len(hdr.variables):]:
        m = key_re.match(ln)
        if m:
            current = m.group(1)
            rest = ln[m.end():].strip()
            if current == "DATA_INFO":
                hdr.data_info = rest
            elif current == "OTHER_COMMENTS":
                hdr.other_comments = rest
        elif current == "OTHER_COMMENTS":
            hdr.other_comments += "\n" + ln.strip()
    return hdr


def scan_headers(icartt_dir, filename_regex, instruments=None, date_range=None):
    """{instrument token: Header} for the files in (optional) date range.

    Variable names can differ across a campaign (e.g. ACTIVATE OPTICAL
    ``_total`` in 2020 vs ``_submicron`` in 2021), so the first and last file
    per token are parsed and their variable lists unioned (first file wins on
    duplicates; its DATA_INFO/OTHER_COMMENTS are kept).
    ``date_range`` is (min, max) strings in the filename's date format.
    """
    icartt_dir = Path(icartt_dir) if icartt_dir else None
    if not icartt_dir or not icartt_dir.is_dir():
        return {}
    rx = re.compile(filename_regex)
    by_token = {}
    for p in sorted(icartt_dir.iterdir()):
        m = rx.match(p.name)
        if not m:
            continue
        tok = m.group("instr")
        if instruments and tok not in instruments:
            continue
        if date_range and not (date_range[0] <= m.group("date") <= date_range[1]):
            continue
        by_token.setdefault(tok, []).append(p)
    out = {}
    for tok, paths in by_token.items():
        hdr = parse_header(paths[0])
        if hdr is None:
            continue
        hdr.token = tok
        if len(paths) > 1:
            extra = parse_header(paths[-1])
            if extra is not None:
                have = {v.name for v in hdr.variables}
                hdr.variables += [v for v in extra.variables if v.name not in have]
        out[tok] = hdr
    return out


# --------------------------------------------------------------------------- #
# STP / ambient measurement basis
# --------------------------------------------------------------------------- #
def measurement_conditions(varinfo, instrument_data_info=""):
    """'STP' | 'ambient' | 'not_applicable' | 'unspecified' for one variable.

    The ICARTT standardized-name token ends in _STP or _AMB for LARGE
    products; note the *variable-name* suffix _amb refers to ambient RH,
    not ambient T/P, so the standard token is checked first.
    """
    if varinfo is None:
        return "unspecified"
    std = (varinfo.standard or "").strip()
    if std:
        last = std.split("_")[-1].upper()
        if last == "STP" or "MASSSTP" in std.upper().replace("_", ""):
            return "STP"
        if last == "AMB":
            return "ambient"
    name = (varinfo.name or "").lower()
    if name.endswith("_stp") or "stdpt" in name:
        return "STP"
    units = (varinfo.units or "").lower()
    if any(t in units for t in ("ppm", "ppb", "ppt")):
        return "not_applicable"  # mole fractions are T/P-independent
    if any(t in units for t in ("unitless", "percent", "fraction", "second",
                                "degree", "none")) or "frac" in name:
        return "not_applicable"  # flags, ratios, RH, durations
    info = (instrument_data_info or "").lower().replace("_", " ")
    if "standard temperature" in info or "stp" in info:
        return "STP"
    if "ambient temperature" in info:
        return "ambient"
    return "unspecified"


# --------------------------------------------------------------------------- #
# size-distribution bin tables
# --------------------------------------------------------------------------- #
_BIN_NAME = re.compile(r"(?:^|_)(?:Bin|bin)0*(\d+)$|^dNdlogD_0*(\d+)_")
_CENTER_IN_DESC = re.compile(r"bin_center_([0-9.]+)\s*um", re.IGNORECASE)
_EDGE_LIST = re.compile(
    r"Bin\s+(Lower|Upper)\s+Edges?\s*\(in\s*um\)\s*=\s*\[([^\]]*)\]", re.IGNORECASE)
_BOUNDS_NM = re.compile(
    r"(Lower Bounds|Mid points|Upper Bounds)\s*:\s*([0-9.,\s]+)", re.IGNORECASE)


def _bin_vars(header):
    """Ordered (name, bin_number) of this header's size-distribution bins."""
    out = []
    for v in header.variables:
        m = _BIN_NAME.search(v.name)
        if m:
            out.append((v, int(m.group(1) or m.group(2))))
    out.sort(key=lambda t: t[1])
    return [v for v, _ in out]


def _edges_from_centers(centers):
    """Derive bin edges as midpoints between adjacent centers (ends mirrored)."""
    c = np.asarray(centers, float)
    if len(c) < 2:
        return c * 0.9, c * 1.1
    mid = 0.5 * (c[:-1] + c[1:])
    lower = np.concatenate([[c[0] - (mid[0] - c[0])], mid])
    upper = np.concatenate([mid, [c[-1] + (c[-1] - mid[-1])]])
    return lower, upper


def bin_table(header):
    """BinTable for one instrument header, or None if it has no bin columns.

    Size sources tried in order: per-variable ``bin_center_X um`` descriptions
    (CDP/CAS), ``Bin Lower/Upper Edges (in um) = [...]`` in OTHER_COMMENTS
    (FCDP), and nm-valued Lower/Mid/Upper bounds lines (SMPS/LAS).
    """
    bins = _bin_vars(header)
    if not bins:
        return None
    names = [v.name for v in bins]

    centers = [(_CENTER_IN_DESC.search(v.description or "") or
                _CENTER_IN_DESC.search(v.standard or "")) for v in bins]
    if all(centers):
        c = np.array([float(m.group(1)) for m in centers])
        lo, up = _edges_from_centers(c)
        return BinTable(names, c, lo, up, "bin_center in variable descriptions"
                                          " (edges derived as midpoints)")

    edges = {m.group(1).lower(): m.group(2)
             for m in _EDGE_LIST.finditer(header.other_comments or "")}
    if {"lower", "upper"} <= set(edges):
        lo = np.array([float(x) for x in re.findall(r"[0-9.]+", edges["lower"])])
        up = np.array([float(x) for x in re.findall(r"[0-9.]+", edges["upper"])])
        if len(lo) == len(up) == len(names):
            return BinTable(names, np.sqrt(lo * up), lo, up,
                            "edge lists in OTHER_COMMENTS (centers geometric mean)")

    nm = {m.group(1).lower(): m.group(2)
          for m in _BOUNDS_NM.finditer(header.other_comments or "")}
    if {"lower bounds", "mid points", "upper bounds"} <= set(nm):
        arrs = {k: np.array([float(x) for x in re.findall(r"[0-9.]+", v)]) / 1e3
                for k, v in nm.items()}
        if all(len(a) == len(names) for a in arrs.values()):
            return BinTable(names, arrs["mid points"], arrs["lower bounds"],
                            arrs["upper bounds"], "nm bounds lines in OTHER_COMMENTS")

    return BinTable(names, np.full(len(names), np.nan),
                    np.full(len(names), np.nan), np.full(len(names), np.nan),
                    "sizes not found in header")


# --------------------------------------------------------------------------- #
# flight envelopes
# --------------------------------------------------------------------------- #
_LEG = re.compile(r"_L(\d+)\.ict$", re.IGNORECASE)


@dataclass
class FlightSpan:
    date: str            # YYYY-MM-DD of takeoff day (from the filename)
    leg: int             # 0 = single flight, 1/2/... = _L1/_L2 files
    start_epoch_s: float  # first data time, seconds since 1970 (UTC)
    end_epoch_s: float
    path: str = ""

    @property
    def flight_id(self):
        return self.date.replace("-", "") + (f"_L{self.leg}" if self.leg else "")


def _file_time_range(path, n_header):
    """(first, last) value of the ICARTT time column (s after midnight UTC)."""
    first = last = None
    with open(path, "r", errors="replace") as f:
        for i, ln in enumerate(f):
            if i < n_header:
                continue
            tok = ln.split(",")[0].strip()
            if not tok:
                continue
            if first is None:
                first = float(tok)
            last = tok
    return (first, float(last)) if first is not None else (None, None)


def flight_spans(icartt_dir, filename_regex, marker_token, date_format="%Y%m%d"):
    """Per-flight time envelopes from the marker instrument's ICARTT files."""
    icartt_dir = Path(icartt_dir) if icartt_dir else None
    if not icartt_dir or not icartt_dir.is_dir() or not marker_token:
        return []
    rx = re.compile(filename_regex)
    spans = []
    for p in sorted(icartt_dir.iterdir()):
        m = rx.match(p.name)
        if not m or m.group("instr") != marker_token:
            continue
        try:
            n_header = int(open(p).readline().split(",")[0])
        except (ValueError, OSError):
            continue
        t0, t1 = _file_time_range(p, n_header)
        if t0 is None:
            continue
        midnight = datetime.strptime(m.group("date"), date_format).replace(
            tzinfo=timezone.utc).timestamp()
        leg = _LEG.search(p.name)
        spans.append(FlightSpan(
            date=datetime.strptime(m.group("date"), date_format).strftime("%Y-%m-%d"),
            leg=int(leg.group(1)) if leg else 0,
            start_epoch_s=midnight + t0, end_epoch_s=midnight + t1, path=str(p)))
    spans.sort(key=lambda s: s.start_epoch_s)
    return spans
