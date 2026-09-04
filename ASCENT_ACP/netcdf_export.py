"""Export the ASCENT-ACP pipeline products to a grouped CF-style netCDF (v4).

One file per campaign year, organized into netCDF-4 groups (see
NETCDF_OUTPUT_SPEC.md):

  /observations            every raw merged-pickle column at native cadence,
                           split into instrument families, + row_qc_flag
  /windowed                shared window coordinates (wavelengths, PSD bins)
  /windowed/observations   60 s QC flag/counts and QC-valid measured means
  /windowed/retrievals     ISARA retrieval outputs
  /clock_alignment         per (flight_date x shift_group) applied clock shifts

Raw variables are carried only in /observations (native cadence). We do NOT
emit 60 s window means of them: that would just be each observation re-averaged
and written back at the native cadence, redundant against /observations. Only
the ISARA retrieval inputs/outputs (which genuinely need a windowed form) live
under /windowed.

v5 layout changes vs v4:

* GROUP-LOCAL axes under /observations: each instrument family folds its
  per-wavelength variables onto its own ``wavelength`` dimension (exactly
  the wavelengths it archives — PI-Neph 473/532/671 nm, CRDS 405/532/662,
  LARGE 450..700), instead of the root union with fill at foreign
  wavelengths. Size-bin dimensions (``diameter_<tag>``) are likewise
  created inside the owning family group. The root ``wavelength`` union
  remains and serves the /windowed* products.
* PI-Neph GRASP products restructured: Real_*/Imag_*/SSA_* fold to
  refractive_index_real/imag/ssa on the local wavelength axis, and the
  15 'PSD-dNdlogr<N>' columns become one psd_dndlogr(flight, time, radius)
  with the GRASP radii (parsed from the header descriptions) as coordinate.
* FCDP 'cbinNN'/'nbinNN' columns fold onto bin dimensions with edges parsed
  from the header descriptions; non-dN/dlogDp bin series keep their own
  name and units instead of being written as ``dndlogd_*``.
* Full instrument-title prefixes are stripped from variable names (fixes
  'chemical_speciated_..._AMS_Starttime' residue) which also restores the
  ICARTT units/descriptions lookup for those variables.
* ICARTT units are normalized to one CF/udunits spelling per quantity
  ('none'/'unitless' -> '1', '#/cm3' -> 'cm-3', ...); the original string
  is kept in ``icartt_units`` when it differed.
* /windowed/observations/window_index: sequential window id at native
  cadence for direct window-block recovery.
* Start/stop/mid time-of-day bookkeeping columns are omitted (recorded in
  the family group's ``omitted_time_bookkeeping_columns`` attribute).

v4 layout changes vs v3:

* One shared root ``wavelength`` dimension (union of all measured/calculated
  wavelengths). Per-wavelength variable copies (Sc550_submicron, SSA_450nm,
  dry_cal_sca_coef_550, ...) are merged into single variables on that axis,
  holding fill at wavelengths they do not cover.
* Root ``latitude``/``longitude``/``altitude`` coordinates at native cadence;
  every (flight, time) data variable references them via a CF ``coordinates``
  attribute so generic tools can georeference variables in any group.
* /windowed is split into /windowed/observations (window QC + QC-valid
  measured means) and /windowed/retrievals (ISARA outputs).
* All optical coefficients in Mm-1.

v3 layout changes vs v2:

* All data live on a (flight, time) grid: flight number in takeoff order
  within the campaign year, and seconds since UTC midnight of that flight's
  takeoff day (the axis extends past 86400 when a flight crosses midnight).
  Flight envelopes come from the marker instrument's ICARTT files
  (see ``flights.py``); merged rows outside every envelope - synthetic
  interpolation the merge engine fills between two same-day flights - are
  dropped.
* The 60 s products are repeated onto the native grid (each second carries
  its window's value); there is no separate coarse time dimension.
* Per-bin size-distribution columns are compacted into one variable per
  instrument with a size dimension whose coordinate is bin-center radius
  (bin-center diameter and bin edges ride along as companion variables).
* Per-variable units/descriptions and the STP-vs-ambient measurement basis
  are read from the source ICARTT headers (``measurement_conditions`` attr).

All optical coefficients are stored in Mm-1, matching the ICARTT sources.
Retrievals are filled (NaN) and flagged where window QA failed.
"""

import datetime
import re
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from . import families, flights, humidify, icartt_headers, varmap
from . import results as results_mod, windows as windows_mod
from .windows import psd_col_name

# Names of the root georeferencing coordinate variables; every (flight, time)
# data variable points at them via a CF "coordinates" attribute so generic
# tools (Panoply etc.) can georeference variables in any group.
_COORD_NAMES = ("latitude", "longitude", "altitude")


def wavelength_union(ch):
    """Sorted union of every wavelength (nm) appearing in the output."""
    return sorted({int(x) for x in
                   [*ch.dry_wvl_sca, *ch.dry_wvl_abs, *ch.wet_wvl_sca,
                    *(ch.val_wvl or [])]})

_MM_PER_M = 1.0e6  # m-1 -> Mm-1 (ISARA outputs SI; the file convention is Mm-1)

# Retr_PSD output keys that carry a wavelength, e.g. dry_cal_sca_coef_550_m-1
_RETR_KEY = re.compile(
    r"^(?P<state>dry|wet|amb)_(?P<kind>cal|meas)_(?P<quant>sca_coef|abs_coef|ext_coef|SSA)"
    r"_(?P<wvl>\d+)_(?P<unit>m-1|unitless)$"
)
_STATE_NAME = {"dry": "dry", "wet": "wet", "amb": "ambient"}
_STATE_RH_NOTE = {
    "dry": "dry state (as measured, growth factor 1)",
    "wet": "humidified to the fixed wet-state RH (see wet_rh in config_json)",
    "amb": "humidified to the window-mean ambient RH (see rh_ambient)",
}

# Row QC bitmask (native cadence), mirroring filtering.row_qc mask columns.
_ROW_QC_BITS = [("cloudy", 1), ("inlet_bad", 2), ("low_signal", 4), ("low_ssa", 8)]
_ROW_QC_MEANINGS = {
    1: "cloud_contaminated",
    2: "inlet_flag_nonzero_or_missing",
    4: "below_min_dry_sc450",
    8: "below_min_ssa",
}

# Best-effort units for raw passthrough variables lacking header metadata.
_UNIT_HINTS = {
    "Latitude": "degrees_north", "Longitude": "degrees_east",
    "GPS_altitude": "m", "Pressure_altitude": "m",
}
_UNIT_SUFFIX = [
    ("_ppm", "ppm"), ("_ppb", "ppb"), ("_ppt", "ppt"),
    ("_cm3", "cm-3"), ("_percent", "percent"), ("_degC", "degC"),
]

_WINDOW_CM = "time: mean within {w} s window (value repeated at native cadence)"

# ICARTT units strings -> one CF/udunits spelling per quantity. Matching is
# case-insensitive on the stripped string; unmapped strings pass through
# unchanged. When a mapping fires, the archive's original spelling is kept in
# the ``icartt_units`` attribute.
_UNITS_NORM = {
    "none": "1", "unitless": "1", "no units": "1", "dimensionless": "1",
    "%": "percent", "percent": "percent", "pct": "percent",
    "#/cm3": "cm-3", "#/cm^3": "cm-3", "1/cm3": "cm-3", "1/cm^3": "cm-3",
    "cm-3": "cm-3", "/cm3": "cm-3", "#/scm3": "cm-3",
    "#/l": "L-1", "#/liter": "L-1", "1/l": "L-1",
    "#/liter/um": "L-1 um-1", "#/l/um": "L-1 um-1",
    "um2/cm3": "um2 cm-3", "um^2/cm^3": "um2 cm-3",
    "um3/cm3": "um3 cm-3", "um^3/cm^3": "um3 cm-3",
    "g/m^3": "g m-3", "g/m3": "g m-3",
    "ug/m3": "ug m-3", "ug/m^3": "ug m-3", "ug m-3": "ug m-3",
    "1/km": "km-1", "1/mm": "Mm-1", "1/m": "m-1",
    "celsius": "degC", "deg c": "degC", "degrees c": "degC",
    "kelvin": "K", "mb": "hPa", "hpa": "hPa",
    "feet": "ft", "kts": "knot", "knots": "knot",
    "deg": "degree", "degrees": "degree",
    "deg (0-360)": "degree", "deg (+-180)": "degree",
    "seconds": "s", "sec": "s", "s": "s",
    "meters": "m", "m/s": "m s-1", "g/kg": "g kg-1",
    "nm": "nm", "um": "um", "mach": "1",
    "#": "1", "liter": "L",
}


def _normalize_units(units):
    """(normalized, original-or-None): CF/udunits spelling for an ICARTT
    units string; original returned only when a rewrite actually happened."""
    if units is None:
        return None, None
    key = str(units).strip()
    norm = _UNITS_NORM.get(key.lower())
    if norm is None or norm == key:
        return key, None
    return norm, key

# size-distribution bin columns, four conventions: '<TAG>_BinNN' (ACTIVATE),
# 'dNdlogD_NNN_<TAG>', center-diameter-named '<TAG>_<NNN>nm[_<CAL>]'
# (SEAC4RS LARGE, where CAL is a calibration token like PSL/AmmSO4), and
# bare 'cbinNN'/'nbinNN' (SPEC FCDP). The nm form can false-match
# per-wavelength scalars, so _split_bin_columns demotes tags with < 4 columns.
_BIN_SHORT = re.compile(
    r"(?:^|_)([A-Za-z0-9]+)_Bin(\d+)$|^dNdlogD_0*(\d+)_([A-Za-z0-9]+)$"
    r"|(?:^|_)([A-Za-z0-9]+)_(\d+)nm(?:_([A-Za-z0-9]+))?$"
    r"|^([cn]bin)0*(\d+)$")

# ordinal PSD series with sizes only in the header descriptions, e.g.
# PI-Neph GRASP 'PSD-dNdlogr1'..'PSD-dNdlogr15' ("... at r=0.05um");
# folded onto a group-local radius (r) / diameter (d) dimension.
_ORDINAL_PSD = re.compile(r"^(?P<base>.*dndlog(?P<rd>[rd]))0*(?P<num>\d+)$",
                          re.IGNORECASE)
_SIZE_IN_DESC = re.compile(r"at\s*(?P<rd>[rd])\s*=\s*(?P<val>[0-9.]+)\s*um",
                           re.IGNORECASE)

# Observation variables that are per-wavelength copies of one quantity
# (Sc550_submicron, Abs470_total, SSA_450nm, Ext532_submicron_amb,
# Real_532/Imag_532 [PI-Neph GRASP], ext_TD_405nm [CRDS], ...); merged onto
# the owning family's group-local wavelength dimension at export. Base
# tokens match case-insensitively (CRDS names are lowercase).
_WVL_BASES = {"sc": "scattering", "abs": "absorption",
              "ext": "extinction", "ssa": "ssa",
              "real": "refractive_index_real",
              "imag": "refractive_index_imag"}
_WVL_SHORT = re.compile(
    r"^(?P<pre>[A-Za-z]+(?:_[A-Za-z]+)*_?)(?P<wvl>\d{3})(?P<post>nm|_[A-Za-z0-9_]+)?$")


def _parse_wvl_short(short):
    """(merged_variable_name, wavelength_nm) for a per-wavelength column,
    or None when the name is not one (gamma550, AEscat_450to700nm, ...)."""
    m = _WVL_SHORT.match(short)
    if not m:
        return None
    tokens = m.group("pre").rstrip("_").split("_")
    base = _WVL_BASES.get(tokens[0].lower())
    if base is None:
        return None
    parts = [base] + tokens[1:]
    post = m.group("post") or ""
    if post and post != "nm":
        parts.append(post.lstrip("_"))
    return "_".join(parts), int(m.group("wvl"))


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def _git_sha(repo_dir):
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _instrument_metadata_text(meta):
    if not meta:
        return "unavailable"
    lines = []
    for title in meta.get("Data_Info", {}):
        lines.append(f"INSTRUMENT: {title}")
        for field in ("PI_Info", "Institution_Info", "Uncertainty", "Revision", "Stipulations"):
            val = meta.get(field, {}).get(title)
            if val:
                lines.append(f"  {field}: {val}")
    return "\n".join(lines)


def _meta_titles(meta):
    return list((meta or {}).get("Data_Info", {}))


def _short_name(col, title):
    """Column name with its instrument-title prefix removed.

    '/' is replaced (netCDF would interpret it as a subgroup separator —
    e.g. SAGA's 'Cl_ug/m3' columns).
    """
    short = col[len(title) + 1:] if title and col.startswith(title + "_") else col
    return short.lstrip("_").replace("/", "_per_")


def _guess_units(short):
    if short in _UNIT_HINTS:
        return _UNIT_HINTS[short]
    for suf, u in _UNIT_SUFFIX:
        if short.endswith(suf):
            return u
    return None


def _shift_group_map(cfg):
    """variable (full column name) -> shift_group, from the shift table CSV."""
    path = cfg.paths.shift_table_csv
    if not path or not Path(path).exists():
        return {}
    tbl = pd.read_csv(path)
    if "variable" not in tbl or "shift_group" not in tbl:
        return {}
    return dict(zip(tbl["variable"].astype(str), tbl["shift_group"].astype(str)))


def _row_qc_flag(masks):
    flag = np.zeros(len(masks), dtype=np.int16)
    for name, bit in _ROW_QC_BITS:
        if name in masks:
            flag |= (masks[name].to_numpy(bool) * bit).astype(np.int16)
    return flag


def _bin_tag(short):
    """('cdp', 3) style (tag, bin number) for a bin column short name."""
    m = _BIN_SHORT.search(short)
    if not m:
        return None, None
    if m.group(1) is not None:
        return m.group(1).lower(), int(m.group(2))
    if m.group(3) is not None:
        return m.group(4).lower(), int(m.group(3))
    if m.group(8) is not None:                # cbinNN / nbinNN (FCDP)
        return m.group(8).lower(), int(m.group(9))
    tag = m.group(5).lower()
    if m.group(7):
        tag = f"{tag}_{m.group(7).lower()}"   # e.g. uhsas_ammso4
    return tag, int(m.group(6))               # ordered by center nm


# --------------------------------------------------------------------------- #
# per-column metadata resolved from the source ICARTT headers
# --------------------------------------------------------------------------- #
class ColumnMeta:
    """units / long_name / standard token / STP-basis for merged columns."""

    def __init__(self, headers, fammap):
        self._headers = list(headers.values())
        self._by_title = {h.title_clean: h for h in self._headers}
        self._fammap = fammap

    def header_for_title(self, title):
        return self._by_title.get(title)

    def titles(self):
        """Cleaned titles of every scanned header (== merged column prefixes)."""
        return list(self._by_title)

    def attrs(self, col, title, fam, short=None):
        short = short if short is not None else _short_name(col, title)
        out = {"long_name": short, "source_column": col}
        # '/' in ICARTT names is sanitized to '_per_' for netCDF (SAGA
        # 'Cl_ug/m3'); look the header variable up under both spellings
        lookups = [short]
        if "_per_" in short:
            lookups.append(short.replace("_per_", "/"))
        hdr = self._by_title.get(title)
        vi = next((v for s in lookups for v in [hdr.var(s)] if v), None) \
            if hdr else None
        if vi is None:
            # titles can drift between the header and the merged prefix
            # (e.g. DLH), and AMS/AMS-CVI share one title; fall back to the
            # header that actually defines this variable name
            for h in self._headers:
                v2 = next((v for s in lookups for v in [h.var(s)] if v), None)
                if v2 is not None:
                    hdr, vi = h, v2
                    break
        if vi is not None:
            if vi.units:
                norm, orig = _normalize_units(vi.units)
                out["units"] = norm
                if orig:
                    out["icartt_units"] = orig
            if vi.description:
                out["long_name"] = vi.description
            elif vi.standard:
                # 3-field ICARTT variable lines put their prose in the
                # standardized-name slot (DASH-SP, PI-Neph, stdPT/Inlet)
                out["long_name"] = vi.standard
            if vi.standard:
                out["icartt_standard_name"] = vi.standard
        elif _guess_units(short):
            out["units"] = _guess_units(short)
        if fam == "state_nav":
            cond = "not_applicable"
        else:
            cond = icartt_headers.measurement_conditions(
                vi, hdr.data_info if hdr else "")
        out["measurement_conditions"] = cond
        return out


def _bin_tables_from_headers(headers):
    """{tag: (BinTable, Header)} for every header that has bin columns.

    One header can carry several bin series (FCDP cbinNN + nbinNN); the
    table is registered under each tag so both series find their sizes.
    """
    out = {}
    for hdr in headers.values():
        bt = icartt_headers.bin_table(hdr)
        if bt is None:
            continue
        for name in bt.columns:
            tag, _ = _bin_tag(name)
            if tag and tag not in out:
                out[tag] = (bt, hdr)
    return out


def _fallback_bin_table(tag, df, cfg):
    """SMPS/LAS bin table from the packaged CSVs when headers are missing."""
    from . import sizebins
    csv = {"smps": cfg.psd.smps_bins_csv, "las": cfg.psd.las_bins_csv}.get(tag)
    if not csv or not Path(csv).exists():
        return None
    try:
        bins = sizebins.load_bins(csv)
        cols = varmap.resolve_bins(df, tag.upper())
    except Exception:
        return None
    if len(cols) != len(bins["dpg"]):
        return None
    shorts = [c.split("_")[-2] + "_" + c.split("_")[-1] for c in cols]  # e.g. SMPS_Bin01
    return icartt_headers.BinTable(shorts, bins["dpg"], bins["dpl"], bins["dpu"],
                                   f"packaged CSV {Path(csv).name}")


# --------------------------------------------------------------------------- #
# streaming grouped writer on the (flight, time) grid
# --------------------------------------------------------------------------- #
class _Writer:
    def __init__(self, path, cfg, fgrid):
        self.cfg = cfg
        self.fg = fgrid
        self.nc = netCDF4.Dataset(path, "w", format="NETCDF4")
        self.comp = dict(zlib=cfg.output.compression_level > 0,
                         complevel=cfg.output.compression_level, shuffle=True)
        self.nc.createDimension("flight", fgrid.n_flights)
        self.nc.createDimension("time", fgrid.n_seconds)
        self._chunk2d = (1, fgrid.n_seconds)

    # ---- structure -------------------------------------------------------
    def group(self, path):
        g = self.nc
        for part in path.strip("/").split("/"):
            if part:
                g = g.groups.get(part) or g.createGroup(part)
        return g

    def dim(self, name, size, gpath=None):
        """Create a dimension; ``gpath`` scopes it to that group (netCDF
        resolves dimension names from the variable's group upward, so a
        group-local axis shadows a root one of the same name)."""
        g = self.group(gpath) if gpath else self.nc
        if name not in g.dimensions:
            g.createDimension(name, size)

    # ---- low-level variable creation --------------------------------------
    def raw_var(self, gpath, name, dims, data, attrs=None, dtype=None, fill=None):
        """Create a variable and write ``data`` directly (already gridded)."""
        g = self.group(gpath) if gpath else self.nc
        data = np.asarray(data)
        if dtype is None:
            dtype = data.dtype
            if self.cfg.output.float32 and np.issubdtype(dtype, np.floating):
                dtype = np.float32
        if dtype is str or dtype == "str":
            v = g.createVariable(name, str, dims)
            for i, s in enumerate(data):
                v[i] = str(s)
        else:
            kw = dict(self.comp)
            if fill is None and np.issubdtype(np.dtype(dtype), np.floating):
                fill = np.dtype(dtype).type(np.nan)
            if fill is not None:
                kw["fill_value"] = fill
            if dims[:2] == ("flight", "time"):
                kw["chunksizes"] = self._chunk2d + (1,) * (len(dims) - 2)
            v = g.createVariable(name, dtype, dims, **kw)
            v[...] = data
        attrs = dict(attrs or {})
        if dims[:2] == ("flight", "time") and name not in _COORD_NAMES:
            attrs.setdefault("coordinates", "latitude longitude")
        for k, val in attrs.items():
            v.setncattr(k, val)
        return v

    # ---- (flight, time) helpers -------------------------------------------
    def scatter2d(self, gpath, name, per_row, attrs=None, dtype=np.float32, fill=None):
        """Scatter per-merged-row values onto (flight, time) and write."""
        if fill is None:
            fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else -1
        per_row = np.asarray(per_row, float)
        if not np.issubdtype(np.dtype(dtype), np.floating):
            per_row = np.where(np.isnan(per_row), fill, per_row)
        arr = flights.scatter(self.fg, per_row, fill=fill, dtype=dtype)
        return self.raw_var(gpath, name, ("flight", "time"), arr,
                            attrs=attrs, dtype=dtype, fill=fill)

    def scatter3d(self, gpath, name, per_row_slices, extra_dim, attrs=None,
                  dtype=np.float32):
        """3-D (flight, time, extra) written one extra-dim slice at a time."""
        g = self.group(gpath)
        fill = np.dtype(dtype).type(np.nan)
        v = g.createVariable(name, dtype, ("flight", "time", extra_dim),
                             fill_value=fill,
                             chunksizes=self._chunk2d + (1,), **self.comp)
        for k, per_row in enumerate(per_row_slices):
            if per_row is None:  # wavelength not covered -> leave as fill
                continue
            v[:, :, k] = flights.scatter(self.fg, per_row, dtype=dtype)
        attrs = dict(attrs or {})
        attrs.setdefault("coordinates", "latitude longitude")
        for kk, val in attrs.items():
            v.setncattr(kk, val)
        return v

    def group_attrs(self, gpath, attrs):
        g = self.group(gpath)
        for k, v in attrs.items():
            if v is not None and v != "":
                g.setncattr(k, v)

    def close(self):
        self.nc.close()


# --------------------------------------------------------------------------- #
# window -> row broadcast
# --------------------------------------------------------------------------- #
def _window_row_index(df_index, window_index, window_s):
    """Per merged row: position of its 60 s window in ``window_index`` (-1 none)."""
    epoch = np.asarray(df_index.view("int64"), dtype="int64") / 1e9
    row_center = np.floor(epoch / window_s) * window_s + window_s / 2.0
    win_epoch = np.asarray(window_index.view("int64"), dtype="int64") / 1e9
    pos = np.searchsorted(win_epoch, row_center)
    pos_c = np.clip(pos, 0, max(len(win_epoch) - 1, 0))
    ok = (pos < len(win_epoch)) & (np.abs(win_epoch[pos_c] - row_center) < 1e-3)
    return np.where(ok, pos_c, -1).astype(np.int64)


def _broadcast(win_values, win_idx, fill=np.nan):
    vals = np.asarray(win_values, float)
    out = np.where(win_idx >= 0, vals[np.clip(win_idx, 0, None)], fill)
    return out


# --------------------------------------------------------------------------- #
# group builders
# --------------------------------------------------------------------------- #
def _write_root(w, df, cfg, fgrid, meta):
    here = Path(__file__).resolve().parent.parent
    fg = fgrid
    ch = cfg.channels
    w.raw_var("", "flight", ("flight",), fg.flight_number, dtype=np.int32, attrs={
        "long_name": "flight number, takeoff order within the campaign year"})
    w.raw_var("", "time", ("time",), fg.time_axis_s(), dtype=np.float64, attrs={
        "units": "s",
        "long_name": "seconds since UTC midnight of the flight's takeoff day",
        "comment": ("exceeds 86400 when a flight crosses UTC midnight; absolute "
                    "time of a sample = midnight_epoch(flight) + time")})
    w.raw_var("", "flight_id", ("flight",), np.array(fg.flight_id), dtype=str, attrs={
        "long_name": "takeoff date (YYYYMMDD) with _L<n> suffix for same-day flights"})
    w.raw_var("", "flight_date", ("flight",), np.array(fg.date), dtype=str, attrs={
        "long_name": "UTC date of takeoff (YYYY-MM-DD)"})
    w.raw_var("", "midnight_epoch", ("flight",), fg.midnight_epoch_s,
              dtype=np.int64, attrs={
        "units": "seconds since 1970-01-01T00:00:00Z",
        "long_name": "UTC midnight of the takeoff day"})
    w.raw_var("", "takeoff_time", ("flight",), fg.takeoff_sod, dtype=np.float64,
              attrs={"units": "s",
                     "long_name": "first data time, seconds since takeoff-day midnight"})
    w.raw_var("", "landing_time", ("flight",), fg.landing_sod, dtype=np.float64,
              attrs={"units": "s",
                     "long_name": "last data time, seconds since takeoff-day midnight"})

    # shared wavelength axis for the /windowed* retrieval products
    # (observation families carry their own group-local wavelength axes)
    wvls = wavelength_union(ch)
    w.dim("wavelength", len(wvls))
    w.raw_var("", "wavelength", ("wavelength",), np.array(wvls, float),
              dtype=np.float64, attrs={
        "units": "nm", "long_name": "optical wavelength (retrieval products)",
        "comment": (f"union of the retrieval wavelengths; scattering measured at "
                    f"{ch.dry_wvl_sca} nm, absorption at {ch.dry_wvl_abs} nm, "
                    f"humidified scattering constraint at {ch.wet_wvl_sca} nm"
                    + (f", extra calculated output at {ch.val_wvl} nm"
                       if ch.val_wvl else "")
                    + "; variables hold fill at wavelengths they do not cover. "
                    "Groups under /observations define their own wavelength "
                    "axes covering what each instrument family archives")})

    # root georeferencing coordinates (1 Hz aircraft nav), referenced by the
    # CF "coordinates" attribute of every (flight, time) data variable
    for name, suffix, units, std in [
        ("latitude", ch.lat_suffix, "degrees_north", "latitude"),
        ("longitude", ch.lon_suffix, "degrees_east", "longitude"),
        ("altitude", ch.alt_suffix, "m", "altitude"),
    ]:
        col = varmap.resolve(df, suffix, required=False)
        if col is not None:
            w.scatter2d("", name, df[col].to_numpy(float), attrs={
                "units": units, "standard_name": std,
                "long_name": f"aircraft {name} at native cadence"})

    pct = 100.0 * fg.n_dropped / max(len(fg.row_flight), 1)
    w.group_attrs("", {
        "Conventions": "CF-1.8",
        "title": (f"ISARA aerosol retrievals and merged in-situ observations, "
                  f"{cfg.campaign} {cfg.year} ({cfg.psd.variant_name} variant)"),
        "institution": "NASA GSFC / processed with ASCENT-ACP",
        "source": ("Airborne in-situ measurements merged from ICARTT files, "
                   "clock-aligned, QC-filtered and window-averaged; ISARA/MOPSMAP "
                   "retrieval"),
        "history": (f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}: "
                    "created by ASCENT_ACP.netcdf_export"),
        "references": ("Kacenelenbogen et al. (2022), doi:10.5194/acp-22-3713-2022 "
                       "(QA method); Gasteiger & Wiegner (2018), "
                       "doi:10.5194/gmd-11-2739-2018 (MOPSMAP single-particle "
                       "optics); Schlosser et al. (2025) (ISARA method; LARGE "
                       "submicron-scattering impactor D50 = 1.0 um aerodynamic)"),
        "comment": ("All data on a (flight, time) grid: time is seconds since UTC "
                    "midnight of each flight's takeoff day. Groups: /observations "
                    "(native-cadence raw passthrough by instrument family), "
                    "/windowed (60 s statistics repeated at native cadence; "
                    "/observations = window QC + QC-valid means, /retrievals = "
                    "ISARA outputs), /clock_alignment (time-base provenance). "
                    "All optical coefficients in Mm-1. Per-variable "
                    "measurement_conditions attributes record the STP-vs-ambient "
                    "basis from the source ICARTT headers."),
        "flight_segmentation": fg.source,
        "n_rows_dropped_outside_flights": int(fg.n_dropped),
        "dropped_rows_note": (
            f"{fg.n_dropped} merged rows ({pct:.1f}%) fell outside every flight "
            "envelope and were excluded. The merge engine fills values up to "
            "~72 min beyond each instrument's real coverage (nearest/linear "
            "fill), so rows between two same-day flights or beyond the flight "
            "envelope are synthetic, not measurements."),
        "source_merged_pickle": str(cfg.paths.input_pkl),
        "window_seconds": int(cfg.window.window_s),
        "psd_max_um": float(cfg.psd.psd_max_um),
        "psd_truncation_note": (
            "The retrieval PSD is weighted by the impactor penetration curve "
            "in /windowed/observations/impactor_penetration (V6+; replaces "
            "the V5 sharp cut, which misrepresented the ~50% transmission at "
            "the impactor D50 and under-predicted red scattering). psd_max_um "
            "is only a hard grid cap. Absorption channels are bulk (no "
            "impactor) in all years; reported dndlogdp is unweighted."),
        "impactor_d50_aero_um": float(cfg.psd.impactor_d50_aero_um),
        "impactor_gsd": float(cfg.psd.impactor_gsd),
        "impactor_rho_gcm3": float(cfg.psd.impactor_rho_gcm3),
        "isara_forward_engine": cfg.isara.forward_engine,
        "isara_estimator": cfg.isara.estimator,
        "config_json": cfg.to_json(),
        "ascent_acp_git_sha": _git_sha(here),
        "isara_git_sha": _git_sha(cfg.paths.isara_code_dir),
        "instrument_metadata": _instrument_metadata_text(meta),
    })


def _family_split(df_columns, fammap, titles):
    assigned = families.assign_families(list(df_columns), fammap, titles)
    by_family = {}
    for col, (fam, title) in assigned.items():
        by_family.setdefault(fam, []).append((col, title))
    return by_family


_TAG_STOP = {"NASA", "HU", "THE", "ON", "FROM", "IN", "SITU", "AND", "OF",
             "MEASUREMENTS", "DISTRIBUTIONS", "PARTICLE", "SIZE", "AEROSOL",
             "SUBMICRON", "FALCON", "CLOUD"}


def _title_tag(title):
    """A short, stable instrument tag from an ICARTT title for disambiguation."""
    caps = [t for t in re.split(r"[^A-Za-z0-9]+", title or "")
            if len(t) >= 2 and t.upper() == t and t.upper() not in _TAG_STOP]
    if caps:
        return "".join(caps)
    return re.sub(r"[^A-Za-z0-9]+", "", (title or "x"))[:8] or "x"


def _dedupe_shorts(entries):
    """Ensure output names are unique within a family.

    ``entries`` is [(col, title, short)]; on collision, disambiguate every
    colliding member with its title tag. Returns
    [(col, title, short, unique_name)] — ``short`` still keys header lookups,
    ``unique_name`` is the netCDF variable name.
    """
    from collections import Counter
    counts = Counter(s for _, _, s in entries)
    out = []
    for col, title, short in entries:
        name = f"{short}_{_title_tag(title)}" if counts[short] > 1 else short
        out.append((col, title, short, name))
    return out


def _split_bin_columns(pairs):
    """Split [(col, title)] into ({tag: [(col, short, n)]}, [(col, short)])."""
    bins, scalars = {}, []
    titles = {}
    for col, title in pairs:
        short = _short_name(col, title)
        tag, n = _bin_tag(short)
        if tag is not None:
            bins.setdefault(tag, []).append((col, short, n))
            titles.setdefault(tag, {})[col] = title
        else:
            scalars.append((col, title, short))
    # names caught by the <NNN>nm pattern that ALSO parse as per-wavelength
    # optical quantities (ext_dry_405nm, SSA_dry_700nm, ...) are wavelength
    # copies, not size bins — demote them individually. Before this check,
    # SEAC4RS V2 wrote the CRDS/LARGE *_dry_<wvl>nm family as a bogus
    # 6-"bin" size distribution with 0.405-0.7 um "diameters".
    for tag in list(bins):
        keep = []
        for col, short, n in bins[tag]:
            if short.endswith("nm") and _parse_wvl_short(short):
                scalars.append((col, titles[tag][col], short))
            else:
                keep.append((col, short, n))
        if keep:
            bins[tag] = keep
        else:
            del bins[tag]
    # a "tag" with only a few members is almost certainly a per-wavelength
    # scalar caught by the <NNN>nm pattern, not a size distribution — demote
    for tag in [t for t, e in bins.items() if len(e) < 4]:
        for col, short, _ in bins.pop(tag):
            scalars.append((col, titles[tag][col], short))
    for tag in bins:
        bins[tag].sort(key=lambda t: t[2])
    return bins, scalars


def _bin_table_from_shorts(tag, shorts):
    """BinTable synthesized from diameter-named columns ('<TAG>_<NNN>nm...'),
    edges at log-midpoints (end edges extrapolated). None if not nm-named."""
    centers = varmap.bin_centers_from_names(shorts)
    if centers is None or len(centers) < 2:
        return None
    c = np.asarray(centers, float)
    if np.any(np.diff(c) <= 0):
        return None
    logc = np.log10(c)
    mid = (logc[:-1] + logc[1:]) / 2
    lo = 10 ** np.r_[2 * logc[0] - mid[0], mid]
    up = 10 ** np.r_[mid, 2 * logc[-1] - mid[-1]]
    return icartt_headers.BinTable(list(shorts), c, lo, up,
                                   "bin center diameters parsed from the "
                                   "column names; edges at log-midpoints")


def _write_size_coords(w, gpath, tag, bt, order):
    """Size dimension + radius/diameter coordinate variables for one tag.

    The dimension is named after the bin-center-diameter variable so that
    ``diameter_<tag>`` is a true coordinate variable (plotting tools then put
    diameter on the axis instead of a bare bin index). The dimension lives in
    the instrument-family group itself (v5): size axes are group-local.
    """
    dim = f"diameter_{tag}"
    w.dim(dim, len(order), gpath=gpath)
    center = bt.center_um[order] if bt is not None else np.full(len(order), np.nan)
    lower = bt.lower_um[order] if bt is not None else np.full(len(order), np.nan)
    upper = bt.upper_um[order] if bt is not None else np.full(len(order), np.nan)
    src = bt.source if bt is not None else "sizes not found"
    w.raw_var(gpath, f"diameter_{tag}", (dim,), center, dtype=np.float64, attrs={
        "units": "um", "long_name": f"{tag.upper()} bin center diameter",
        "source": src})
    w.raw_var(gpath, f"radius_{tag}", (dim,), center / 2.0, dtype=np.float64, attrs={
        "units": "um", "long_name": f"{tag.upper()} bin center radius"})
    w.raw_var(gpath, f"radius_lower_{tag}", (dim,), lower / 2.0, dtype=np.float64,
              attrs={"units": "um", "long_name": f"{tag.upper()} bin lower-edge radius"})
    w.raw_var(gpath, f"radius_upper_{tag}", (dim,), upper / 2.0, dtype=np.float64,
              attrs={"units": "um", "long_name": f"{tag.upper()} bin upper-edge radius"})
    return dim


def _match_bin_table(bt, shorts):
    """Index of each present short name in the header bin table (or None)."""
    if bt is None:
        return None
    pos = {n: i for i, n in enumerate(bt.columns)}
    if all(s in pos for s in shorts):
        return np.array([pos[s] for s in shorts])
    # tolerate name-prefix differences: both sides are ascending bin order,
    # so a full-length table maps positionally
    if len(bt.columns) == len(shorts):
        return np.arange(len(shorts))
    return None


_TIME_BOOKKEEPING = re.compile(
    r"(?:^|_)(?:(?:start|stop|end|mid(?:point)?)_?(?:utc|time)"
    r"|(?:utc|time)_?(?:start|stop|end|mid))", re.IGNORECASE)


def _is_time_bookkeeping(short, units):
    """True for start/stop/midpoint time-of-day columns that survived the
    merge as data (their content duplicates the row time base). Units are
    checked when archived; a column whose units the header does not resolve
    (independent-variable lines) is judged on the strict name pattern."""
    if units is not None and not re.search(r"sec|sfm|past_midnight|utc",
                                           str(units), re.IGNORECASE):
        return False
    return bool(_TIME_BOOKKEEPING.search(short))


def _split_ordinal_series(scalars, colmeta, fam):
    """Split ordinal PSD columns ('PSD-dNdlogr1'..'PSD-dNdlogr15') out of
    [(col, title, short)].

    Returns ([(base, rd, [(col, title, short, num, size_um)])], rest).
    A series folds only when it has >= 4 members and every member's header
    description carries its size ('... at r=0.05um'); otherwise the columns
    stay scalars.
    """
    groups, rest = {}, []
    for col, title, short in scalars:
        m = _ORDINAL_PSD.match(short)
        if m:
            groups.setdefault((m.group("base"), m.group("rd").lower()),
                              []).append((col, title, short,
                                          int(m.group("num"))))
        else:
            rest.append((col, title, short))
    series = []
    for (base, rd), members in sorted(groups.items()):
        members.sort(key=lambda t: t[3])
        sized = []
        for col, title, short, num in members:
            a = colmeta.attrs(col, title, fam, short)
            # ICARTT variable lines with only 3 fields land the description
            # in the standardized-name slot (PI-Neph), so search both
            desc = f"{a.get('long_name', '')} {a.get('icartt_standard_name', '')}"
            sm = _SIZE_IN_DESC.search(desc)
            sized.append((col, title, short, num,
                          float(sm.group("val")) if sm else None))
        if len(sized) >= 4 and all(s[4] is not None for s in sized):
            series.append((base, rd, sized))
        else:
            rest.extend((c, t, s) for c, t, s, _, _ in sized)
    return series, rest


def _write_observations(w, df, masks, cfg, meta, colmeta, fammap, bin_tables):
    fg = w.fg
    shift_groups = _shift_group_map(cfg)
    w.group_attrs("/observations", {
        "long_name": "raw merged observations at native cadence",
        "native_sampling_seconds": float(fg.step_s),
    })
    w.scatter2d("/observations", "row_qc_flag", _row_qc_flag(masks),
                dtype=np.int16, fill=-1, attrs={
        "long_name": "row quality-control bitmask (0 = valid)",
        "flag_masks": np.array(sorted(_ROW_QC_MEANINGS), np.int16),
        "flag_meanings": " ".join(_ROW_QC_MEANINGS[k] for k in sorted(_ROW_QC_MEANINGS)),
        "_FillValue_meaning": "no merged data at this second",
        "comment": "Kacenelenbogen et al. (2022) A1.1 row screening; see QA_CRITERIA.md"})

    by_family = _family_split(
        df.columns, fammap,
        list(dict.fromkeys(_meta_titles(meta) + colmeta.titles())))
    title_meta = meta or {}
    for fam in families.family_order(fammap, by_family):
        gpath = f"/observations/{fam}"
        bins, scalars = _split_bin_columns(by_family[fam])
        series, scalars = _split_ordinal_series(scalars, colmeta, fam)

        # start/stop/mid time-of-day bookkeeping duplicates the row time
        # base; omit it (recorded in a group attribute for provenance)
        omitted = [(col, title, short) for col, title, short in scalars
                   if _is_time_bookkeeping(
                       short, colmeta.attrs(col, title, fam, short).get("units"))]
        scalars = [e for e in scalars if e not in omitted]

        # pull per-wavelength copies of one quantity onto a GROUP-LOCAL
        # wavelength axis (v5): each family's axis holds exactly the
        # wavelengths that family archives (PI-Neph 473/532/671 nm, ...)
        wvl_groups, rest = {}, []
        for col, title, short in scalars:
            parsed = _parse_wvl_short(short)
            if parsed:
                name, wvl = parsed
                if wvl in wvl_groups.get(name, {}):  # cross-instrument clash
                    rest.append((col, title, short))
                    continue
                wvl_groups.setdefault(name, {})[wvl] = (col, title, short)
            else:
                rest.append((col, title, short))

        fam_wvls = sorted({x for m in wvl_groups.values() for x in m})
        if len(fam_wvls) < 2:  # a 1-wavelength axis is noise; keep scalars
            rest.extend(v for m in wvl_groups.values() for v in m.values())
            wvl_groups = {}
        if wvl_groups:
            w.dim("wavelength", len(fam_wvls), gpath=gpath)
            w.raw_var(gpath, "wavelength", ("wavelength",),
                      np.array(fam_wvls, float), dtype=np.float64, attrs={
                "units": "nm",
                "long_name": f"optical wavelength ({fam} instruments)",
                "comment": ("group-local axis: the wavelengths archived by "
                            "this instrument family")})
        for name, members in sorted(wvl_groups.items()):
            col0, title0, short0 = members[min(members)]
            attrs = colmeta.attrs(col0, title0, fam, short0)
            if attrs["long_name"] == short0 and "icartt_standard_name" in attrs:
                attrs["long_name"] = attrs["icartt_standard_name"]
            attrs["long_name"] = re.sub(
                r"\s*(?:at\s*)?(?:wavelength=)?\b\d{3}\s?(?:nm|um)\b"
                r"|(?:_at)?_\d{3}nm", "",
                attrs["long_name"]).strip()
            if "icartt_standard_name" in attrs:
                attrs["icartt_standard_name"] = re.sub(
                    r"_(Blue|Green|Red)(?=_)", "", attrs["icartt_standard_name"])
            attrs["source_column"] = ", ".join(
                members[x][0] for x in sorted(members))
            attrs["shift_group"] = shift_groups.get(col0, "none")
            attrs["comment"] = (f"values at {sorted(members)} nm of this "
                                "group's wavelength axis; fill elsewhere")
            w.scatter3d(gpath, name,
                        (df[members[x][0]].to_numpy(float) if x in members
                         else None for x in fam_wvls),
                        "wavelength", attrs=attrs)

        # ordinal PSD series with sizes from the header descriptions
        # (PI-Neph GRASP 15-bin dN/dlogr) -> group-local radius/diameter axis
        for base, rd, members in series:
            dim = {"r": "radius", "d": "diameter"}[rd]
            while dim in w.group(gpath).dimensions:
                dim += "_"
            sizes = np.array([s for _, _, _, _, s in members], float)
            w.dim(dim, len(members), gpath=gpath)
            w.raw_var(gpath, dim, (dim,), sizes, dtype=np.float64, attrs={
                "units": "um",
                "long_name": f"bin center {dim.rstrip('_')}",
                "comment": ("bin centers parsed from the ICARTT variable "
                            "descriptions; edges not archived")})
            col0, title0, short0, _, _ = members[0]
            attrs = colmeta.attrs(col0, title0, fam, short0)
            if attrs["long_name"] == short0 and "icartt_standard_name" in attrs:
                attrs["long_name"] = attrs["icartt_standard_name"]
            attrs["long_name"] = _SIZE_IN_DESC.sub(
                "", attrs["long_name"]).strip(" ,;")
            attrs["source_column"] = (f"{len(members)} columns "
                                      f"{members[0][2]}..{members[-1][2]}")
            attrs["shift_group"] = shift_groups.get(col0, "none")
            name = re.sub(r"[^0-9a-z]+", "_", base.lower()).strip("_")
            w.scatter3d(gpath, name,
                        (df[c].to_numpy(float) for c, _, _, _, _ in members),
                        dim, attrs=attrs)

        for col, title, short, name in _dedupe_shorts(rest):
            attrs = colmeta.attrs(col, title, fam, short)
            attrs["shift_group"] = shift_groups.get(col, "none")
            w.scatter2d(gpath, name, df[col].to_numpy(float), attrs=attrs)

        for tag, entries in bins.items():
            cols = [c for c, _, _ in entries]
            shorts = [s for _, s, _ in entries]
            bt = bin_tables.get(tag)
            bt = bt[0] if isinstance(bt, tuple) else bt
            order = _match_bin_table(bt, shorts)
            if order is None:
                bt2 = _fallback_bin_table(tag, df, cfg)
                order = _match_bin_table(bt2, shorts)
                bt = bt2 if order is not None else None
            if order is None:
                bt2 = _bin_table_from_shorts(tag, shorts)
                if bt2 is not None:
                    bt, order = bt2, np.arange(len(shorts))
            if order is None:
                bt, order = None, np.arange(len(shorts))
            dim = _write_size_coords(w, gpath, tag, bt, order)
            title = next(t for c, t in by_family[fam] if c == cols[0])
            attrs = colmeta.attrs(cols[0], title, fam, shorts[0])
            attrs.update(
                source_column=f"{len(cols)} columns {shorts[0]}..{shorts[-1]}",
                shift_group=shift_groups.get(cols[0], "none"))
            if attrs.get("units") in (None, "cm-3", "#/cm3", "#/cm^3",
                                      "1/cm3", "1/cm^3"):
                # aerosol sizers: one dN/dlogDp convention for the file
                attrs["long_name"] = (f"{tag.upper()} number size "
                                      "distribution dN/dlogDp")
                if attrs.get("units") not in (None, "cm-3"):
                    attrs["icartt_units"] = attrs["units"]
                attrs["units"] = "cm-3"
                name = f"dndlogd_{tag}"
            else:
                # per-bin series in other units (FCDP cbin '#/liter/um',
                # nbin counts): keep the tag name and the archived quantity
                attrs["long_name"] = re.sub(
                    r"\(\s*[0-9.]+\s*-\s*[0-9.]+\s*um\s*\)", "",
                    attrs["long_name"]).strip(" ,;") + " (per size bin)"
                name = tag
            w.scatter3d(gpath, name,
                        (df[c].to_numpy(float) for c in cols), dim, attrs=attrs)

        gattrs = {"long_name": families.family_long_name(fammap, fam) or None}
        if fammap:
            gattrs["comment"] = fammap.get("family_comment", {}).get(fam)
        if omitted:
            gattrs["omitted_time_bookkeeping_columns"] = ", ".join(
                s for _, _, s in omitted)
        titles = sorted({t for _, t in by_family[fam]})
        for fld in ("PI_Info", "Institution_Info", "Uncertainty", "Revision", "Stipulations"):
            vals = [f"{t}: {title_meta.get(fld, {}).get(t)}" for t in titles
                    if title_meta.get(fld, {}).get(t)]
            if vals:
                gattrs[fld] = "\n".join(vals)
        w.group_attrs(gpath, {k: v for k, v in gattrs.items() if v})


def _write_windowed_parent(w, results_df, grid, cfg, win_idx):
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    w.group_attrs("/windowed", {
        "long_name": "60 s window statistics, repeated at native cadence",
        "comment": ("Each native-cadence sample carries the value of the 60 s "
                    "window containing it; seconds with no window are filled. "
                    "/observations = window QC and QC-valid measured means; "
                    "/retrievals = ISARA retrieval outputs.")})
    gp = "/windowed/observations"
    w.group_attrs(gp, {
        "long_name": "window QC and QC-valid-only measured window means"})
    w.group_attrs("/windowed/retrievals", {
        "long_name": "ISARA retrieval outputs (MOPSMAP grid search)"})

    # shared retrieval coordinates (wavelength dim lives at root). The size
    # dimension is named "dp_mid" so the bin-center-diameter variable is a
    # true coordinate variable and plots come out against diameter.
    w.dim("dp_mid", len(grid))
    w.raw_var(gp, "dp_mid", ("dp_mid",), grid.dpg_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin center diameter"})
    w.raw_var(gp, "radius_mid", ("dp_mid",), grid.dpg_um / 2.0,
              dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin center radius"})
    w.raw_var(gp, "dp_lower", ("dp_mid",), grid.dpl_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin lower-bound diameter"})
    w.raw_var(gp, "dp_upper", ("dp_mid",), grid.dpu_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin upper-bound diameter"})
    w.raw_var(gp, "psd_instrument", ("dp_mid",),
              np.array(grid.instrument), dtype=str,
              attrs={"long_name": "source instrument of each retrieval PSD bin"})
    if grid.penetration is not None:
        w.raw_var(gp, "impactor_penetration", ("dp_mid",),
                  np.asarray(grid.penetration, float), dtype=np.float64, attrs={
                      "units": "1",
                      "long_name": ("impactor penetration applied to the PSD "
                                    "fed to the retrieval (reported dndlogdp "
                                    "is NOT weighted)"),
                      "comment": ("log-logistic P(D_a)=1/(1+(D_a/D50)^s), "
                                  f"D50={cfg.psd.impactor_d50_aero_um} um "
                                  "aerodynamic (Schlosser et al. 2025), "
                                  f"16-84% gsd={cfg.psd.impactor_gsd}, "
                                  "D_a=dpg*sqrt("
                                  f"{cfg.psd.impactor_rho_gcm3} g cm-3); "
                                  "all ones when no impactor is configured. "
                                  "Absorption channels are bulk (no "
                                  "impactor) but are forward-modeled with "
                                  "the same weighted PSD; the resulting "
                                  "inconsistency is small (abs carries "
                                  "<10% of the fit chi^2)."),
                  })

    w.scatter2d(gp, "window_index", win_idx.astype(float), dtype=np.int32,
                fill=-1, attrs={
        "units": "1",
        "long_name": ("sequential id of the 60 s window this sample belongs "
                      "to (fill where no window)"),
        "comment": ("Every /windowed* variable repeats its window's value at "
                    "the native cadence; equal window_index = same window. "
                    "Use this to recover window blocks (e.g. one scatter "
                    "point per window) without run-length heuristics. Ids "
                    "index the ISARA results bundle rows, ordered by window "
                    "center time across the whole campaign.")})

    flag = _broadcast(results_df["window_qc_flag"].to_numpy(float), win_idx)
    w.scatter2d(gp, "window_qc_flag", flag, dtype=np.int32, fill=-1, attrs={
        "long_name": "window quality control bitmask (0 = good); gates ISARA retrieval",
        "flag_masks": np.array(sorted(windows_mod.FLAG_MEANINGS), np.int32),
        "flag_meanings": " ".join(windows_mod.FLAG_MEANINGS[k]
                                  for k in sorted(windows_mod.FLAG_MEANINGS)),
        "cell_methods": cm,
        "comment": "Kacenelenbogen et al. (2022) A1.2 window screening; see QA_CRITERIA.md"})
    for col, long_name in [
        ("n_valid", "number of QC-valid native samples in window"),
        ("n_cloudy", "samples rejected as cloud-contaminated"),
        ("n_inlet_bad", "samples rejected by inlet flag"),
        ("n_low_signal", "samples rejected by minimum dry Sc450 filter"),
        ("n_low_ssa", "samples rejected by minimum SSA filter"),
        ("n_ambient", "QC-valid samples with usable ambient RH (DLH present, "
                      "below the ambient RH ceiling)"),
    ]:
        if col in results_df:
            vals = _broadcast(results_df[col].fillna(0).to_numpy(float), win_idx)
            w.scatter2d(gp, col, vals, dtype=np.int32, fill=-1,
                        attrs={"units": "1", "long_name": long_name, "cell_methods": cm})


def _write_retrievals(w, results_df, grid, cfg, win_idx):
    ch = cfg.channels
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    go = "/windowed/observations"
    gp = "/windowed/retrievals"

    wvls = wavelength_union(ch)

    def col_rows(col, scale=1.0):
        return _broadcast(results_df[col].to_numpy(float) * scale, win_idx)

    def add_wvl(gpath, name, col_by_wvl, attrs, scale=1.0):
        """Write (flight, time, wavelength); wavelengths without a column
        (or with a column absent from the results) stay fill."""
        w.scatter3d(gpath, name,
                    (col_rows(col_by_wvl[x], scale)
                     if col_by_wvl.get(x) in results_df else None
                     for x in wvls), "wavelength", attrs=attrs)

    add_wvl(go, "scattering_dry_measured",
            {x: f"Sc{x}_dry_mean" for x in ch.dry_wvl_sca},
            {"units": "Mm-1", "cell_methods": cm, "measurement_conditions": "STP",
             "long_name": (f"window-mean dry ({cfg.psd.variant_name}) scattering "
                           f"coefficient, gamma-adjusted to "
                           f"{cfg.filters.dry_ref_rh:.0f}% RH where measured above it")})
    add_wvl(go, "scattering_dry_measured_std",
            {x: f"Sc{x}_dry_std" for x in ch.dry_wvl_sca},
            {"units": "Mm-1", "cell_methods": cm,
             "long_name": "within-window standard deviation of dry scattering"})
    add_wvl(go, "absorption_measured",
            {x: f"Abs{x}_mean" for x in ch.dry_wvl_abs},
            {"units": "Mm-1", "cell_methods": cm, "measurement_conditions": "STP",
             "long_name": "window-mean dry bulk absorption coefficient (PSAP)"})
    add_wvl(go, "absorption_measured_std",
            {x: f"Abs{x}_std" for x in ch.dry_wvl_abs},
            {"units": "Mm-1", "cell_methods": cm,
             "long_name": "within-window standard deviation of absorption"})
    add_wvl(go, "ssa_measured", {x: f"SSA{x}_mean" for x in ch.dry_wvl_sca},
            {"units": "1", "cell_methods": cm,
             "long_name": "window-mean single scattering albedo (LARGE-derived)"})

    wet_w = ch.wet_wvl_sca[0]
    w.scatter2d(go, "scattering_humidified_synthesized",
                col_rows(f"Sc{wet_w}_wet_mean"), attrs={
        "units": "Mm-1", "cell_methods": cm, "measurement_conditions": "STP",
        "long_name": (f"window-mean scattering at {wet_w} nm gamma-adjusted to "
                      f"{cfg.filters.wet_rh:.0f}% RH using the LARGE gamma "
                      "parameterization (synthesized, not directly measured)"),
        "comment": ("SC_calcRH = SC_measRH / exp(gamma*ln((100-calcRH)/(100-measRH))); "
                    "gamma is the LARGE-derived hygroscopic growth exponent. This "
                    "variable is the fitting target of the kappa retrieval (not an "
                    "independent validation). Under kappa_objective='ratio' (see "
                    "/windowed/retrievals) the fitted quantity is the enhancement: "
                    "scattering_wet_calculated/scattering_dry_calculated matches "
                    "this variable divided by scattering_dry_measured; under "
                    "'absolute' the wet_calculated coefficient matched it "
                    "directly.")})

    if f"Sc{wet_w}_amb_mean" in results_df:
        w.scatter2d(go, "scattering_ambient_synthesized",
                    col_rows(f"Sc{wet_w}_amb_mean"), attrs={
            "units": "Mm-1", "cell_methods": cm, "measurement_conditions": "STP",
            "long_name": (f"window-mean scattering at {wet_w} nm gamma-adjusted "
                          "to ambient RH (synthesized, not directly measured)"),
            "comment": ("per-second gamma adjustment of the dry scattering to the "
                        "DLH ambient RH over liquid water, then window-averaged; "
                        f"seconds with RH above {cfg.filters.ambient_rh_max:.0f}% "
                        "or without DLH data are excluded (see rh_ambient, "
                        "n_ambient). Directly comparable to LARGE's "
                        "Sc550_submicron_amb.")})

    for name, col, units, long_name in [
        ("rh_scattering", "RH_Sc_mean", "percent", "window-mean nephelometer sample RH"),
        ("rh_ambient", "RH_amb_mean", "percent",
         "window-mean ambient RH over liquid water (DLH) used for the ambient state"),
        ("rh_ambient_std", "RH_amb_std", "percent",
         "within-window standard deviation of ambient RH"),
        ("gamma550", "gamma_mean", "1", "window-mean scattering hygroscopic growth exponent"),
        ("f_rh_550", "fRH_mean", "1", "window-mean f(RH) 20->80% at 550 nm (LARGE)"),
        ("angstrom_exponent", "AE_mean", "1", "window-mean scattering Angstrom exponent 450-700 nm"),
        ("angstrom_exponent_std", "AE_std", "1", "within-window std of scattering Angstrom exponent"),
        ("latitude", "lat_mean", "degrees_north", "window-mean latitude"),
        ("longitude", "lon_mean", "degrees_east", "window-mean longitude"),
        ("altitude", "alt_mean", "m", "window-mean GPS altitude"),
    ]:
        if col in results_df:
            w.scatter2d(go, name, col_rows(col),
                        attrs={"units": units, "long_name": long_name, "cell_methods": cm})

    w.scatter3d(go, "dndlogdp",
                (col_rows(psd_col_name(d)) for d in grid.dpg_um), "dp_mid", attrs={
        "units": "cm-3", "cell_methods": cm, "measurement_conditions": "STP",
        "long_name": "window-mean dry number size distribution dN/dlogDp (SMPS+LAS)"})

    cri_note = ("the retrieval assumes a spectrally flat refractive index "
                "(one value fit jointly to all channels)")
    mix_note = ("volume-weighted mix of the retrieved dry CRI with water "
                "(1.33+0i) at the kappa-Kohler growth factor; spectrally flat")
    for var, col, long_name in [
        ("refractive_index_real", "dry_RRI_unitless",
         f"ISARA-retrieved real part of the dry complex refractive index; {cri_note}"),
        ("refractive_index_imag", "dry_IRI_unitless",
         f"ISARA-retrieved imaginary part of the dry complex refractive index; {cri_note}"),
        ("kappa", "kappa_unitless",
         "ISARA-retrieved hygroscopicity parameter (kappa-Kohler, single bulk "
         "value; fit objective in group attribute kappa_objective)"),
        ("cri_n_accepted", "dry_CRI_n_accepted_unitless",
         "number of CRI grid candidates matching the measurements within "
         "tolerance (reduced chi^2 <= 1 under the chi2-wmean estimator)"),
        ("refractive_index_real_accepted_std", "dry_RRI_accepted_std_unitless",
         "spread of the real refractive index over the grid (posterior-"
         "weighted std under the chi2-wmean estimator; retrieval-spread "
         "proxy, not a full uncertainty)"),
        ("refractive_index_imag_accepted_std", "dry_IRI_accepted_std_unitless",
         "spread of the imaginary refractive index over the grid (posterior-"
         "weighted std under the chi2-wmean estimator; retrieval-spread "
         "proxy, not a full uncertainty)"),
        ("cri_min_chi2", "dry_CRI_min_chi2_unitless",
         "minimum reduced chi^2 over the CRI grid (sigma: 20% scattering, "
         "1 Mm-1 absorption; success requires <= 1)"),
        ("kappa_min_chi2", "kappa_min_chi2_unitless",
         "minimum reduced chi^2 over the kappa grid (sigma: 1% humidified "
         "scattering; success requires <= 1)"),
        ("kappa_std", "kappa_std_unitless",
         "posterior-weighted standard deviation of kappa over the grid "
         "(retrieval-spread proxy, not a full uncertainty)"),
        ("refractive_index_real_wet", "wet_RRI_unitless",
         f"real refractive index of the wet state; {mix_note}"),
        ("refractive_index_imag_wet", "wet_IRI_unitless",
         f"imaginary refractive index of the wet state; {mix_note}"),
        ("growth_factor_wet", "wet_gf_unitless",
         "kappa-Kohler diameter growth factor of the wet state"),
        ("refractive_index_real_ambient", "amb_RRI_unitless",
         f"real refractive index of the ambient state; {mix_note}"),
        ("refractive_index_imag_ambient", "amb_IRI_unitless",
         f"imaginary refractive index of the ambient state; {mix_note}"),
        ("growth_factor_ambient", "amb_gf_unitless",
         "kappa-Kohler diameter growth factor of the ambient state (at rh_ambient)"),
    ]:
        if col in results_df:
            attrs = {"units": "1", "long_name": long_name, "cell_methods": cm}
            if var == "kappa":
                attrs["comment"] = (
                    "the search grid extends slightly negative (floor "
                    "isara.kappa_min, default -0.1): negative values are "
                    "EFFECTIVE, capturing windows whose synthesized wet/dry "
                    "enhancement target is below 1 (gamma-parameterization "
                    "noise around f~1, particle restructuring); they are "
                    "consistent with zero hygroscopic growth, not literal "
                    "water loss. Humidified-state products (dndlogdp_wet/"
                    "ambient, wet/ambient CRI and growth factors) are only "
                    "computed for kappa >= 0.")
            w.scatter2d(gp, var, col_rows(col), attrs=attrs)

    for col, var in [("attempt_flag_CRI_unitless", "attempt_flag_cri"),
                     ("attempt_flag_kappa_unitless", "attempt_flag_kappa")]:
        if col in results_df:
            vals = _broadcast(results_df[col].fillna(0).to_numpy(float), win_idx)
            w.scatter2d(gp, var, vals, dtype=np.int32, fill=-1, attrs={
                "units": "1",
                "long_name": f"ISARA {var.split('_')[-1].upper()} retrieval attempt flag",
                "flag_values": np.array([0, 1, 2], np.int32),
                "flag_meanings": "not_attempted attempted_but_failed success",
                "cell_methods": cm})

    # humidified PSDs, remapped onto the dry bin grid (surface-conserving);
    # negative (effective) kappa is not literal water uptake -> no grown PSD
    if "kappa_unitless" in results_df:
        kappa = results_df["kappa_unitless"].to_numpy(float)
        kappa = np.where(kappa < 0, np.nan, kappa)
        psd = results_df[[psd_col_name(d) for d in grid.dpg_um]].to_numpy(float)
        psd_states = [("wet", np.full(len(results_df), float(cfg.filters.wet_rh)))]
        if "RH_amb_mean" in results_df:
            psd_states.append(("ambient", results_df["RH_amb_mean"].to_numpy(float)))
        remap_note = (
            "dry PSD grown by the bulk kappa-Kohler growth factor (uniform log-"
            "diameter shift; one gf for all sizes) and remapped back onto the "
            "dry bin grid conserving SURFACE AREA exactly; number and volume "
            "are conserved only approximately. Surface grown past the last "
            "bin edge is in the companion surface_beyond_grid variable. NaN "
            "where kappa (or ambient RH) is unavailable.")
        for tag, rh_state in psd_states:
            gf = humidify.growth_factor(kappa, rh_state)
            dnd = np.full_like(psd, np.nan)
            spill = np.full(len(results_df), np.nan)
            for i in range(len(results_df)):
                if np.isfinite(gf[i]):
                    dnd[i], spill[i] = humidify.humidified_psd(
                        psd[i], grid.dpg_um, grid.dpl_um, grid.dpu_um, gf[i])
            w.scatter3d(gp, f"dndlogdp_{tag}",
                        (_broadcast(dnd[:, k], win_idx) for k in range(psd.shape[1])),
                        "dp_mid", attrs={
                "units": "cm-3", "cell_methods": cm,
                "long_name": (f"{tag}-state number size distribution dN/dlogDp "
                              "on the dry bin grid"),
                "comment": remap_note})
            w.scatter2d(gp, f"surface_beyond_grid_{tag}",
                        _broadcast(spill, win_idx), attrs={
                "units": "um2 cm-3", "cell_methods": cm,
                "long_name": (f"{tag}-state surface concentration grown past "
                              "the largest PSD bin edge"),
                "comment": ("companion to dndlogdp_" + tag + "; 0 when the "
                            "grown distribution fits the grid")})

    if "retrieval_qc_flag" in results_df:
        vals = _broadcast(results_df["retrieval_qc_flag"].to_numpy(float), win_idx)
        w.scatter2d(gp, "retrieval_qc_flag", vals, dtype=np.int32, fill=-1, attrs={
            "units": "1", "long_name": "why a window has or lacks an ISARA retrieval",
            "flag_values": np.array(sorted(results_mod.RETRIEVAL_QC_MEANINGS), np.int32),
            "flag_meanings": " ".join(results_mod.RETRIEVAL_QC_MEANINGS[k]
                                      for k in sorted(results_mod.RETRIEVAL_QC_MEANINGS)),
            "cell_methods": cm})

    # forward-calculated optical properties, one variable per (quantity, state)
    # on the shared wavelength dimension
    quant_name = {"sca_coef": "scattering", "abs_coef": "absorption",
                  "ext_coef": "extinction", "SSA": "ssa"}
    calc = {}  # (state, quant) -> {wvl: column}
    for col in results_df.columns:
        m = _RETR_KEY.match(str(col))
        if not m or m.group("kind") == "meas":
            continue
        calc.setdefault((m.group("state"), m.group("quant")), {})[
            int(m.group("wvl"))] = str(col)
    for (state, quant), col_by_wvl in calc.items():
        is_coef = quant != "SSA"
        add_wvl(gp, f"{quant_name[quant]}_{_STATE_NAME[state]}_calculated", col_by_wvl,
                {"units": "Mm-1" if is_coef else "1", "cell_methods": cm,
                 "long_name": (f"MOPSMAP-calculated {_STATE_NAME[state]} "
                               f"{quant_name[quant]}"
                               + (" coefficient" if is_coef else "")
                               + " for the retrieved refractive index"
                               + (" and kappa" if state != "dry" else "")),
                 "comment": (f"{_STATE_RH_NOTE[state]}; values only at wavelengths "
                             f"ISARA computed ({sorted(col_by_wvl)} nm), fill "
                             "elsewhere"
                             + ("; humidified absorption is model-derived "
                                "(absorption is only measured dry)"
                                if state != "dry" and quant == "abs_coef" else ""))},
                scale=_MM_PER_M if is_coef else 1.0)

    # dry scattering closure ratio: forward-modeled (retrieved CRI, sizing-
    # corrected PSD) over measured, per dry scattering channel. Diagnoses the
    # PSD amplitude error the optical closure sees (sizer under/overcounting).
    def closure_rows(x):
        cal = results_df[f"dry_cal_sca_coef_{x}_m-1"].to_numpy(float) * _MM_PER_M
        meas = results_df[f"Sc{x}_dry_mean"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(meas > 0, cal / meas, np.nan)
        return _broadcast(ratio, win_idx)

    kappa_objective = ("ratio" if any(str(c).startswith("kappa_dry_closure_")
                                      for c in results_df.columns) else "absolute")
    w.scatter3d(gp, "scattering_dry_closure_ratio",
                (closure_rows(x)
                 if (x in ch.dry_wvl_sca
                     and f"dry_cal_sca_coef_{x}_m-1" in results_df
                     and f"Sc{x}_dry_mean" in results_df) else None
                 for x in wvls), "wavelength", attrs={
        "units": "1", "cell_methods": cm,
        "long_name": ("dry scattering closure ratio "
                      "(MOPSMAP-calculated / measured)"),
        "comment": ("QC diagnostic: the forward-modeled dry scattering from "
                    "the (sizing-corrected) PSD at the retrieved CRI, divided "
                    "by the nephelometer measurement. Values far from 1 "
                    "indicate PSD amplitude errors (optical-sizer counting/"
                    "sizing biases) that the bounded CRI grid cannot absorb. "
                    f"kappa_objective='{kappa_objective}': "
                    + ("kappa fits the wet/dry scattering ENHANCEMENT, so "
                       "this amplitude error cancels out of kappa"
                       if kappa_objective == "ratio" else
                       "kappa fit the ABSOLUTE humidified coefficient, so "
                       "this amplitude error leaked directly into kappa "
                       "(closure < 1 inflates kappa, > 1 deflates it)"))})
    w.group_attrs(gp, {"kappa_objective": kappa_objective})


def _write_uncertainty(w, unc_df, grid, cfg, win_idx):
    """/windowed_uncertainty: 1-sigma uncertainties mirroring /windowed.

    Variable names and dims match their /windowed counterparts so users can
    difference the groups directly; values are 1-sigma. Built by
    uncertainty_propagation.py (see UNCERTAINTY_MODULE_PLAN.md).
    """
    ch = cfg.channels
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    go = "/windowed_uncertainty/observations"
    gp = "/windowed_uncertainty/retrievals"
    wvls = wavelength_union(ch)
    um_doc = "ACMAP_Meloe/ISARA/aerosol_insitu_uncertainty_models.md"
    w.group_attrs("/windowed_uncertainty", {
        "comment": (
            "1-sigma uncertainties for the like-named variables under "
            "/windowed. Noise term: chi2-wmean posterior over the CRI grid "
            "(instrument sigma models) through product Jacobians; correlated "
            "nuisances (PSD diameter scale 0.10 lnD, PSD concentration "
            "scale, impactor D50/steepness/density, nephelometer and PSAP "
            "common-mode calibration terms) via the ensemble-gain "
            "linearization; quadrature total. v1 simplifications: kappa "
            "sigma is the grid-posterior std only (PSD-side nuisances "
            "largely cancel in the wet/dry ratio and are not propagated "
            "into kappa); nuisances treated independent; sigma models "
            "evaluated on MEASURED window means. Source error models: "
            + um_doc),
        "sigma_convention": "1-sigma, symmetric",
    })

    def col_rows(col, scale=1.0):
        return _broadcast(unc_df[col].to_numpy(float) * scale, win_idx)

    def add_wvl(gpath, name, col_by_wvl, attrs, scale=1.0):
        w.scatter3d(gpath, name,
                    (col_rows(col_by_wvl[x], scale)
                     if col_by_wvl.get(x) in unc_df else None
                     for x in wvls), "wavelength", attrs=attrs)

    sig = "1-sigma uncertainty of "
    add_wvl(go, "scattering_dry_measured",
            {x: f"Sc{x}_dry_sigma" for x in ch.dry_wvl_sca},
            {"units": "Mm-1", "cell_methods": cm,
             "long_name": sig + "the window-mean dry scattering coefficient "
             "(TSI 3563 model)"})
    add_wvl(go, "absorption_measured",
            {x: f"Abs{x}_sigma" for x in ch.dry_wvl_abs},
            {"units": "Mm-1", "cell_methods": cm,
             "long_name": sig + "the window-mean absorption coefficient "
             "(PSAP model incl. the 0.016*b_sp scattering-subtraction term "
             "with the measured scattering)"})
    if "Sc_wet_sigma" in unc_df:
        w.scatter2d(go, "scattering_humidified_synthesized",
                    col_rows("Sc_wet_sigma"), attrs={
            "units": "Mm-1", "cell_methods": cm,
            "long_name": sig + "the synthesized humidified scattering "
            "(v1: nephelometer model only; gamma-parameterization term "
            "not yet included)"})
    psd_cols = [f"psd_sigma_{psd_col_name(d)}" for d in grid.dpg_um]
    if all(c in unc_df for c in psd_cols):
        w.scatter3d(go, "dndlogdp",
                    (col_rows(c) for c in psd_cols), "dp_mid", attrs={
            "units": "cm-3", "cell_methods": cm,
            "long_name": sig + "the window-mean dN/dlogDp (per-bin diagonal "
            "term: Poisson + relative; the correlated 0.10 lnD diameter-"
            "scale term is NOT representable per bin and is instead "
            "propagated into the retrieval uncertainties)"})

    for var, col, long_name in [
        ("refractive_index_real", "dry_RRI_unitless",
         sig + "the retrieved dry real refractive index (full budget)"),
        ("refractive_index_imag", "dry_IRI_unitless",
         sig + "the retrieved dry imaginary refractive index (full budget)"),
        ("kappa", "kappa_unitless",
         sig + "kappa (v1: grid-posterior std only)"),
        ("refractive_index_real_wet", "wet_RRI_unitless", sig + "the wet-state RRI"),
        ("refractive_index_imag_wet", "wet_IRI_unitless", sig + "the wet-state IRI"),
        ("growth_factor_wet", "wet_gf_unitless", sig + "the wet growth factor"),
        ("refractive_index_real_ambient", "amb_RRI_unitless", sig + "the ambient RRI"),
        ("refractive_index_imag_ambient", "amb_IRI_unitless", sig + "the ambient IRI"),
        ("growth_factor_ambient", "amb_gf_unitless", sig + "the ambient growth factor"),
        ("angstrom_exponent_dry_calculated", "dry_AE_unitless",
         sig + "the dry calculated 450-700 nm scattering Angstrom exponent "
         "(correlations between wavelengths included)"),
        ("sizing_scale_shift", "sizing_lnD_shift_unitless",
         "posterior-mean log-diameter scale shift of the PSD inferred from "
         "the fit residual (diagnostic of the correlated LAS sizing "
         "nuisance; positive = the fit prefers larger true sizes than the "
         "AmmSO4-calibrated labels; PSDs and products are NOT corrected by "
         "this shift — it is counted as uncertainty)"),
        ("angstrom_exponent_ambient_calculated", "amb_AE_unitless",
         sig + "the ambient calculated 450-700 nm scattering Angstrom exponent "
         "(correlations between wavelengths included)"),
    ]:
        if col in unc_df:
            w.scatter2d(gp, var, col_rows(col),
                        attrs={"units": "1", "long_name": long_name,
                               "cell_methods": cm})

    # MAP-fit PSD point estimates -> /windowed/retrievals (they are fit
    # diagnostics, not sigmas; sourced from the uncertainty stage's
    # nuisance-MAP solve)
    grt = "/windowed/retrievals"
    fit_note = (
        "first-order forward state at the MAP nuisance amplitudes: the "
        "measured PSD adjusted by the concentration-scale factor "
        "(psd_scale_factor_fit), the lnD shift (/windowed_uncertainty/"
        "retrievals/sizing_scale_shift) and the impactor-parameter terms, "
        "each conditioned on the fit residual under its prior "
        f"(n_scale_sigma={cfg.isara.n_scale_sigma:g}, "
        f"lnD sigma={cfg.isara.sizing_residual_lnd if cfg.isara.sizing_correction else 0.10:g}). "
        "This is the coefficient the retrieval attributes to the aerosol; "
        "its residual vs the measured coefficient should be at instrument "
        "level when the nuisance model is adequate. The archived dndlogdp "
        "is NOT modified.")
    if "psd_scale_factor_unitless" in unc_df:
        w.scatter2d(grt, "psd_scale_factor_fit",
                    col_rows("psd_scale_factor_unitless"), attrs={
            "units": "1", "cell_methods": cm,
            "long_name": ("MAP multiplicative concentration-scale factor of "
                          "the PSD inferred from the fit residual"),
            "comment": ("1 + theta*sigma_N with prior sigma_N = "
                        f"{cfg.isara.n_scale_sigma:g} (isara.n_scale_sigma); "
                        "values below 1 mean the sizer counts more particles "
                        "than the optical closure supports (and vice versa). "
                        "Diagnostic only -- the archived PSD is not rescaled.")})
    for name, col_by_wvl, quant in [
            ("scattering_dry_fit", {x: f"Sc{x}_dry_fit" for x in ch.dry_wvl_sca},
             "scattering"),
            ("absorption_dry_fit", {x: f"Abs{x}_fit" for x in ch.dry_wvl_abs},
             "absorption")]:
        if any(c in unc_df for c in col_by_wvl.values()):
            w.scatter3d(grt, name,
                        (col_rows(col_by_wvl[x])
                         if col_by_wvl.get(x) in unc_df else None
                         for x in wvls), "wavelength", attrs={
                "units": "Mm-1", "cell_methods": cm,
                "long_name": (f"forward-modeled dry {quant} coefficient at "
                              "the retrieved CRI and the MAP-adjusted (fit) "
                              "PSD"),
                "comment": fit_note})

    if "uncertainty_flag" in unc_df:
        vals = _broadcast(unc_df["uncertainty_flag"].to_numpy(float), win_idx)
        w.scatter2d(gp, "uncertainty_flag", vals, dtype=np.int32, fill=-1, attrs={
            "units": "1",
            "long_name": ("linearization-stress flags for the gain-based "
                          "sigmas (bitmask)"),
            "flag_masks": np.array([1, 2, 4, 8], np.int32),
            "flag_meanings": ("rri_near_grid_edge iri_near_grid_edge "
                              "min_chi2_near_gate large_ambient_growth"),
            "cell_methods": cm})

    quant_name = {"sca_coef": "scattering", "abs_coef": "absorption",
                  "ext_coef": "extinction", "SSA": "ssa"}
    calc = {}
    for col in unc_df.columns:
        m = _RETR_KEY.match(str(col))
        if not m or m.group("kind") == "meas":
            continue
        calc.setdefault((m.group("state"), m.group("quant")), {})[
            int(m.group("wvl"))] = str(col)
    for (state, quant), col_by_wvl in calc.items():
        is_coef = quant != "SSA"
        add_wvl(gp, f"{quant_name[quant]}_{_STATE_NAME[state]}_calculated",
                col_by_wvl,
                {"units": "Mm-1" if is_coef else "1", "cell_methods": cm,
                 "long_name": (sig + f"the calculated {_STATE_NAME[state]} "
                               f"{quant_name[quant]}"
                               + (" coefficient" if is_coef else ""))},
                scale=_MM_PER_M if is_coef else 1.0)


# --------------------------------------------------------------------------- #
# /clock_alignment  (date x shift_group) - unchanged small group via xarray
# --------------------------------------------------------------------------- #
def _clock_alignment_ds(cfg):
    path = cfg.paths.shift_diagnostics_csv
    if not path or not Path(path).exists():
        return None
    tbl = pd.read_csv(path)
    if "date" not in tbl or "shift_group" not in tbl:
        return None
    dates = sorted(tbl["date"].dropna().unique())
    groups = sorted(tbl["shift_group"].dropna().unique())
    di = {d: i for i, d in enumerate(dates)}
    gi = {g: i for i, g in enumerate(groups)}
    shape = (len(dates), len(groups))

    applied = np.full(shape, np.nan)
    peak_r = np.full(shape, np.nan)
    n_valid = np.zeros(shape, int)
    halfwidth = np.full(shape, np.nan)
    dcode = np.zeros(shape, np.int8)
    decision = np.full(shape, "", dtype=object)
    reason = np.full(shape, "", dtype=object)

    for _, r in tbl.iterrows():
        if pd.isna(r["date"]) or pd.isna(r["shift_group"]):
            continue
        i, j = di[r["date"]], gi[r["shift_group"]]
        # apply_clock_alignment records an applied shift as decision "SHIFT"
        # (non-applied dates are "SKIP"); accept either "SHIFT" or "APPLY".
        is_apply = str(r.get("decision", "")).upper() in ("SHIFT", "APPLY")
        dcode[i, j] = 1 if is_apply else 0
        opt = r.get("optimal_shift_s")
        applied[i, j] = float(opt) if (is_apply and pd.notna(opt)) else 0.0
        for arr, key in [(peak_r, "peak_r"), (halfwidth, "monotonic_halfwidth_s")]:
            if pd.notna(r.get(key)):
                arr[i, j] = float(r[key])
        if pd.notna(r.get("n_valid")):
            n_valid[i, j] = int(r["n_valid"])
        decision[i, j] = str(r.get("decision", ""))
        reason[i, j] = str(r.get("reason", "") or "")

    ds = xr.Dataset(coords={
        "flight_date": ("flight_date", pd.to_datetime(dates).to_numpy()),
        "shift_group": ("shift_group", np.array(groups, dtype=object)),
    })
    dims = ("flight_date", "shift_group")
    ds["applied_shift_s"] = (dims, applied)
    ds["applied_shift_s"].attrs.update(
        units="s", long_name="clock shift applied to this group on this date (0 if not applied)")
    ds["decision_code"] = (dims, dcode)
    ds["decision_code"].attrs.update(
        long_name="alignment decision (1 = clock shift applied)",
        flag_values=np.array([0, 1], np.int8), flag_meanings="skip shift")
    ds["peak_r"] = (dims, peak_r)
    ds["peak_r"].attrs.update(units="1", long_name="smoothed cross-correlation peak vs LAS reference")
    ds["n_valid"] = (dims, n_valid)
    ds["n_valid"].attrs.update(units="1", long_name="overlapping valid points behind the correlation")
    ds["monotonic_halfwidth_s"] = (dims, halfwidth)
    ds["monotonic_halfwidth_s"].attrs.update(units="s", long_name="monotonic half-width of the correlation peak")
    ds["decision"] = (dims, decision.astype(str))
    ds["reason"] = (dims, reason.astype(str))
    ds.attrs.update(
        description="Per flight-date, per shift-group clock-alignment provenance. "
                    "This is NOT aerosol QA (see /windowed/window_qc_flag); it records "
                    "the time-base correction applied before merging.",
        source_csv=str(Path(path).name))
    return ds


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def export(df, masks, results_df, grid, cfg, meta=None, path=None, uncertainty=None):
    """Write the full grouped v3 netCDF; returns the output Path."""
    if path is None:
        path = Path(cfg.paths.output_dir) / output_filename(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fgrid = flights.build(df, cfg)
    if fgrid.n_flights == 0:
        raise ValueError("no flights found in the merged frame")

    date_lo = min(fgrid.date).replace("-", "")
    date_hi = max(fgrid.date).replace("-", "")
    headers = icartt_headers.scan_headers(
        cfg.merge.icartt_dir, cfg.merge.filename_regex,
        cfg.merge.instruments or None, date_range=(date_lo, date_hi))
    fammap = families.load_family_map(cfg.campaign, cfg.paths.family_map_json or None)
    colmeta = ColumnMeta(headers, fammap)
    bin_tables = _bin_tables_from_headers(headers)

    win_idx = _window_row_index(df.index, results_df.index, cfg.window.window_s)

    w = _Writer(path, cfg, fgrid)
    try:
        _write_root(w, df, cfg, fgrid, meta)
        if cfg.output.emit_observations:
            _write_observations(w, df, masks, cfg, meta, colmeta, fammap, bin_tables)
        _write_windowed_parent(w, results_df, grid, cfg, win_idx)
        _write_retrievals(w, results_df, grid, cfg, win_idx)
        if uncertainty is not None and len(uncertainty):
            _write_uncertainty(w, uncertainty, grid, cfg, win_idx)
    finally:
        w.close()

    ca = _clock_alignment_ds(cfg)
    if ca is not None:
        ca.to_netcdf(path, mode="a", group="/clock_alignment")
    return path


def output_filename(cfg):
    return (f"ISARA_{cfg.campaign}_{cfg.year}_{cfg.psd.variant_name}_"
            f"{cfg.window.window_s}s_{cfg.output.version}.nc")
