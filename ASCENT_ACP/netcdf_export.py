"""Export the ASCENT-ACP pipeline products to a grouped CF-style netCDF (v3).

One file per campaign year, organized into netCDF-4 groups (see
NETCDF_OUTPUT_SPEC.md):

  /observations            every raw merged-pickle column at native cadence,
                           split into instrument families, + row_qc_flag
  /windowed                60 s window QC flag + reject counts
  /windowed/retrievals     ISARA retrievals and QC-valid-only measured means
  /windowed/raw/<family>   unconditional 60 s mean/std of every raw column
  /clock_alignment         per (flight_date x shift_group) applied clock shifts

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

Optical coefficients fed to ISARA are stored in m-1 (converted from the
Mm-1 of the ICARTT sources). Retrievals are filled (NaN) and flagged where
window QA failed.
"""

import datetime
import re
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from . import families, flights, icartt_headers, varmap
from . import results as results_mod, windows as windows_mod, windowing_raw
from .windows import psd_col_name

_M_PER_MM = 1.0e-6  # Mm-1 -> m-1

# Retr_PSD output keys that carry a wavelength, e.g. dry_cal_sca_coef_550_m-1
_RETR_KEY = re.compile(
    r"^(?P<state>dry|wet)_(?P<kind>cal|meas)_(?P<quant>sca_coef|abs_coef|ext_coef|SSA)"
    r"_(?P<wvl>\d+)_(?P<unit>m-1|unitless)$"
)

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

# size-distribution bin columns: '<TAG>_BinNN' or 'dNdlogD_NNN_<TAG>'
_BIN_SHORT = re.compile(r"(?:^|_)([A-Za-z0-9]+)_Bin(\d+)$|^dNdlogD_0*(\d+)_([A-Za-z0-9]+)$")


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
    """Column name with its instrument-title prefix removed."""
    return col[len(title) + 1:] if title and col.startswith(title + "_") else col


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
    return m.group(4).lower(), int(m.group(3))


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

    def attrs(self, col, title, fam, short=None):
        short = short if short is not None else _short_name(col, title)
        out = {"long_name": short, "source_column": col}
        hdr = self._by_title.get(title)
        vi = hdr.var(short) if hdr else None
        if vi is None:
            # titles can drift between the header and the merged prefix
            # (e.g. DLH), and AMS/AMS-CVI share one title; fall back to the
            # header that actually defines this variable name
            for h in self._headers:
                v2 = h.var(short)
                if v2 is not None:
                    hdr, vi = h, v2
                    break
        if vi is not None:
            if vi.units:
                out["units"] = vi.units
            if vi.description:
                out["long_name"] = vi.description
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
    """{tag: (BinTable, Header)} for every header that has bin columns."""
    out = {}
    for hdr in headers.values():
        bt = icartt_headers.bin_table(hdr)
        if bt is None:
            continue
        tag, _ = _bin_tag(bt.columns[0])
        if tag:
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

    def dim(self, name, size):
        if name not in self.nc.dimensions:
            self.nc.createDimension(name, size)

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
        for k, val in (attrs or {}).items():
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
            v[:, :, k] = flights.scatter(self.fg, per_row, dtype=dtype)
        for kk, val in (attrs or {}).items():
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
def _write_root(w, cfg, fgrid, meta):
    here = Path(__file__).resolve().parent.parent
    fg = fgrid
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
                       "doi:10.5194/gmd-11-2739-2018 (MOPSMAP)"),
        "comment": ("All data on a (flight, time) grid: time is seconds since UTC "
                    "midnight of each flight's takeoff day. Groups: /observations "
                    "(native-cadence raw passthrough by instrument family), "
                    "/windowed (60 s statistics repeated at native cadence; "
                    "/retrievals = ISARA + QC-valid means, /raw = all-row means), "
                    "/clock_alignment (time-base provenance). Optical coefficients "
                    "in m-1. Per-variable measurement_conditions attributes record "
                    "the STP-vs-ambient basis from the source ICARTT headers."),
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
    for col, title in pairs:
        short = _short_name(col, title)
        tag, n = _bin_tag(short)
        if tag is not None:
            bins.setdefault(tag, []).append((col, short, n))
        else:
            scalars.append((col, title, short))
    for tag in bins:
        bins[tag].sort(key=lambda t: t[2])
    return bins, scalars


def _write_size_coords(w, gpath, tag, bt, order):
    """Size dimension + radius/diameter coordinate variables for one tag."""
    dim = f"size_{tag}"
    w.dim(dim, len(order))
    center = bt.center_um[order] if bt is not None else np.full(len(order), np.nan)
    lower = bt.lower_um[order] if bt is not None else np.full(len(order), np.nan)
    upper = bt.upper_um[order] if bt is not None else np.full(len(order), np.nan)
    src = bt.source if bt is not None else "sizes not found"
    w.raw_var(gpath, f"radius_{tag}", (dim,), center / 2.0, dtype=np.float64, attrs={
        "units": "um", "long_name": f"{tag.upper()} bin center radius",
        "source": src})
    w.raw_var(gpath, f"diameter_{tag}", (dim,), center, dtype=np.float64, attrs={
        "units": "um", "long_name": f"{tag.upper()} bin center diameter"})
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

    by_family = _family_split(df.columns, fammap, _meta_titles(meta))
    title_meta = meta or {}
    for fam in families.family_order(fammap, by_family):
        gpath = f"/observations/{fam}"
        bins, scalars = _split_bin_columns(by_family[fam])

        for col, title, short, name in _dedupe_shorts(scalars):
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
                bt, order = None, np.arange(len(shorts))
            dim = _write_size_coords(w, gpath, tag, bt, order)
            title = next(t for c, t in by_family[fam] if c == cols[0])
            attrs = colmeta.attrs(cols[0], title, fam, shorts[0])
            attrs.update(
                long_name=f"{tag.upper()} number size distribution dN/dlogDp",
                source_column=f"{len(cols)} columns {shorts[0]}..{shorts[-1]}",
                shift_group=shift_groups.get(cols[0], "none"))
            w.scatter3d(gpath, f"dndlogd_{tag}",
                        (df[c].to_numpy(float) for c in cols), dim, attrs=attrs)

        gattrs = {"long_name": families.family_long_name(fammap, fam) or None}
        titles = sorted({t for _, t in by_family[fam]})
        for fld in ("PI_Info", "Institution_Info", "Uncertainty", "Revision", "Stipulations"):
            vals = [f"{t}: {title_meta.get(fld, {}).get(t)}" for t in titles
                    if title_meta.get(fld, {}).get(t)]
            if vals:
                gattrs[fld] = "\n".join(vals)
        w.group_attrs(gpath, {k: v for k, v in gattrs.items() if v})


def _write_windowed_parent(w, results_df, grid, cfg, win_idx):
    ch = cfg.channels
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    w.group_attrs("/windowed", {
        "long_name": "60 s window statistics, repeated at native cadence",
        "comment": ("Each native-cadence sample carries the value of the 60 s "
                    "window containing it; seconds with no window are filled.")})

    # shared retrieval coordinates
    w.dim("wavelength_sca", len(ch.dry_wvl_sca))
    w.dim("wavelength_abs", len(ch.dry_wvl_abs))
    w.dim("psd_bin", len(grid))
    w.raw_var("/windowed", "wavelength_sca", ("wavelength_sca",),
              np.array(ch.dry_wvl_sca, float), dtype=np.float64,
              attrs={"units": "nm", "long_name": "nephelometer scattering wavelengths"})
    w.raw_var("/windowed", "wavelength_abs", ("wavelength_abs",),
              np.array(ch.dry_wvl_abs, float), dtype=np.float64,
              attrs={"units": "nm", "long_name": "PSAP absorption wavelengths"})
    w.raw_var("/windowed", "dp_mid", ("psd_bin",), grid.dpg_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin center diameter"})
    w.raw_var("/windowed", "radius_mid", ("psd_bin",), grid.dpg_um / 2.0,
              dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin center radius"})
    w.raw_var("/windowed", "dp_lower", ("psd_bin",), grid.dpl_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin lower-bound diameter"})
    w.raw_var("/windowed", "dp_upper", ("psd_bin",), grid.dpu_um, dtype=np.float64,
              attrs={"units": "um", "long_name": "retrieval PSD bin upper-bound diameter"})
    w.raw_var("/windowed", "psd_instrument", ("psd_bin",),
              np.array(grid.instrument), dtype=str,
              attrs={"long_name": "source instrument of each retrieval PSD bin"})

    flag = _broadcast(results_df["window_qc_flag"].to_numpy(float), win_idx)
    w.scatter2d("/windowed", "window_qc_flag", flag, dtype=np.int32, fill=-1, attrs={
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
    ]:
        if col in results_df:
            vals = _broadcast(results_df[col].fillna(0).to_numpy(float), win_idx)
            w.scatter2d("/windowed", col, vals, dtype=np.int32, fill=-1,
                        attrs={"units": "1", "long_name": long_name, "cell_methods": cm})


def _write_retrievals(w, results_df, grid, cfg, win_idx):
    ch = cfg.channels
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    gp = "/windowed/retrievals"

    def col_rows(col, scale=1.0):
        return _broadcast(results_df[col].to_numpy(float) * scale, win_idx)

    def add_wvl(name, cols, dim, scale, attrs):
        w.scatter3d(gp, name, (col_rows(c, scale) for c in cols), dim, attrs=attrs)

    add_wvl("scattering_dry_measured",
            [f"Sc{x}_dry_mean" for x in ch.dry_wvl_sca], "wavelength_sca", _M_PER_MM,
            {"units": "m-1", "cell_methods": cm, "measurement_conditions": "STP",
             "long_name": (f"window-mean dry ({cfg.psd.variant_name}) scattering "
                           f"coefficient, gamma-adjusted to "
                           f"{cfg.filters.dry_ref_rh:.0f}% RH where measured above it")})
    add_wvl("scattering_dry_measured_std",
            [f"Sc{x}_dry_std" for x in ch.dry_wvl_sca], "wavelength_sca", _M_PER_MM,
            {"units": "m-1", "cell_methods": cm,
             "long_name": "within-window standard deviation of dry scattering"})
    add_wvl("absorption_measured",
            [f"Abs{x}_mean" for x in ch.dry_wvl_abs], "wavelength_abs", _M_PER_MM,
            {"units": "m-1", "cell_methods": cm, "measurement_conditions": "STP",
             "long_name": "window-mean dry bulk absorption coefficient (PSAP)"})
    add_wvl("absorption_measured_std",
            [f"Abs{x}_std" for x in ch.dry_wvl_abs], "wavelength_abs", _M_PER_MM,
            {"units": "m-1", "cell_methods": cm,
             "long_name": "within-window standard deviation of absorption"})
    add_wvl("ssa_measured", [f"SSA{x}_mean" for x in ch.dry_wvl_sca],
            "wavelength_sca", 1.0,
            {"units": "1", "cell_methods": cm,
             "long_name": "window-mean single scattering albedo (LARGE-derived)"})

    wet_w = ch.wet_wvl_sca[0]
    w.scatter2d(gp, "scattering_humidified_synthesized",
                col_rows(f"Sc{wet_w}_wet_mean", _M_PER_MM), attrs={
        "units": "m-1", "cell_methods": cm, "measurement_conditions": "STP",
        "long_name": (f"window-mean scattering at {wet_w} nm gamma-adjusted to "
                      f"{cfg.filters.wet_rh:.0f}% RH (synthesized, not directly measured)"),
        "comment": "SC_calcRH = SC_measRH / exp(gamma*ln((100-calcRH)/(100-measRH)))"})

    for name, col, units, long_name in [
        ("rh_scattering", "RH_Sc_mean", "percent", "window-mean nephelometer sample RH"),
        ("gamma550", "gamma_mean", "1", "window-mean scattering hygroscopic growth exponent"),
        ("f_rh_550", "fRH_mean", "1", "window-mean f(RH) 20->80% at 550 nm (LARGE)"),
        ("angstrom_exponent", "AE_mean", "1", "window-mean scattering Angstrom exponent 450-700 nm"),
        ("angstrom_exponent_std", "AE_std", "1", "within-window std of scattering Angstrom exponent"),
        ("latitude", "lat_mean", "degrees_north", "window-mean latitude"),
        ("longitude", "lon_mean", "degrees_east", "window-mean longitude"),
        ("altitude", "alt_mean", "m", "window-mean GPS altitude"),
    ]:
        if col in results_df:
            w.scatter2d(gp, name, col_rows(col),
                        attrs={"units": units, "long_name": long_name, "cell_methods": cm})

    w.scatter3d(gp, "dndlogdp",
                (col_rows(psd_col_name(d)) for d in grid.dpg_um), "psd_bin", attrs={
        "units": "cm-3", "cell_methods": cm, "measurement_conditions": "STP",
        "long_name": "window-mean dry number size distribution dN/dlogDp (SMPS+LAS)"})

    for var, col, long_name in [
        ("refractive_index_real", "dry_RRI_unitless",
         "ISARA-retrieved real part of the dry complex refractive index"),
        ("refractive_index_imag", "dry_IRI_unitless",
         "ISARA-retrieved imaginary part of the dry complex refractive index"),
        ("kappa", "kappa_unitless",
         "ISARA-retrieved hygroscopicity parameter (kappa-Kohler, single bulk value)"),
    ]:
        if col in results_df:
            w.scatter2d(gp, var, col_rows(col),
                        attrs={"units": "1", "long_name": long_name, "cell_methods": cm})

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

    if "retrieval_qc_flag" in results_df:
        vals = _broadcast(results_df["retrieval_qc_flag"].to_numpy(float), win_idx)
        w.scatter2d(gp, "retrieval_qc_flag", vals, dtype=np.int32, fill=-1, attrs={
            "units": "1", "long_name": "why a window has or lacks an ISARA retrieval",
            "flag_values": np.array(sorted(results_mod.RETRIEVAL_QC_MEANINGS), np.int32),
            "flag_meanings": " ".join(results_mod.RETRIEVAL_QC_MEANINGS[k]
                                      for k in sorted(results_mod.RETRIEVAL_QC_MEANINGS)),
            "cell_methods": cm})

    for col in results_df.columns:
        m = _RETR_KEY.match(str(col))
        if not m or m.group("kind") == "meas":
            continue
        var = f"{m.group('state')}_calculated_{m.group('quant').lower()}_{m.group('wvl')}nm"
        units = "m-1" if m.group("unit") == "m-1" else "1"
        w.scatter2d(gp, var, col_rows(col), attrs={
            "units": units, "cell_methods": cm,
            "long_name": (f"MOPSMAP-calculated {m.group('state')} "
                          f"{m.group('quant').replace('_', ' ')} at {m.group('wvl')} nm "
                          "for the retrieved refractive index"
                          + (" and kappa" if m.group("state") == "wet" else ""))})


_STAT_SUFFIXES = ("_scalar_mean", "_mean", "_std")


def _split_stat(name):
    for suf in _STAT_SUFFIXES:
        if name.endswith(suf):
            return name[:-len(suf)], suf[1:]
    return None, None


def _write_windowed_raw(w, raw_windowed, df, cfg, meta, colmeta, fammap,
                        bin_tables, win_idx, window_index):
    cm = _WINDOW_CM.format(w=cfg.window.window_s)
    rw = raw_windowed.reindex(window_index)
    gp0 = "/windowed/raw"
    w.group_attrs(gp0, {
        "long_name": ("unconditional 60 s statistics of every raw column "
                      "(all rows, no QC), repeated at native cadence")})
    if "n_points" in rw:
        vals = _broadcast(rw["n_points"].fillna(0).to_numpy(float), win_idx)
        w.scatter2d(gp0, "n_points", vals, dtype=np.int32, fill=-1,
                    attrs={"units": "1", "cell_methods": cm,
                           "long_name": "number of native rows in window (all rows)"})

    ws_col, wd_col = windowing_raw.find_wind_pair(df.columns, cfg.channels)
    by_family = _family_split(df.columns, fammap, _meta_titles(meta))

    for fam in families.family_order(fammap, by_family):
        gpath = f"{gp0}/{fam}"
        bins, scalars = _split_bin_columns(by_family[fam])

        for col, title, short, name in _dedupe_shorts(scalars):
            for stat in ("mean", "std", "scalar_mean"):
                src = f"{col}_{stat}"
                if src not in rw:
                    continue
                attrs = colmeta.attrs(col, title, fam, short)
                attrs["long_name"] = f"60 s window {stat} of {short} (all rows)"
                attrs["cell_methods"] = cm
                attrs["source_column"] = col
                if col in (ws_col, wd_col) and stat == "mean":
                    attrs["cell_methods"] = cm.replace("mean", "vector mean")
                    attrs["comment"] = "vector-averaged from u/v wind components"
                if col == wd_col and stat == "std":
                    attrs["comment"] = "Yamartino (1984) circular standard deviation"
                vals = _broadcast(rw[src].to_numpy(float), win_idx)
                w.scatter2d(gpath, f"{name}_{stat}", vals, attrs=attrs)

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
                bt, order = None, np.arange(len(shorts))
            dim = _write_size_coords(w, gpath, tag, bt, order)
            for stat in ("mean", "std"):
                srcs = [f"{c}_{stat}" for c in cols]
                if not all(s in rw for s in srcs):
                    continue
                w.scatter3d(gpath, f"dndlogd_{tag}_{stat}",
                            (_broadcast(rw[s].to_numpy(float), win_idx) for s in srcs),
                            dim, attrs={
                    "long_name": f"60 s window {stat} of {tag.upper()} dN/dlogDp (all rows)",
                    "cell_methods": cm})

        ln = families.family_long_name(fammap, fam)
        if ln:
            w.group_attrs(gpath, {"long_name": f"{ln} - 60 s window statistics"})


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
def export(df, masks, results_df, grid, cfg, meta=None, raw_windowed=None,
           path=None):
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
        _write_root(w, cfg, fgrid, meta)
        if cfg.output.emit_observations:
            _write_observations(w, df, masks, cfg, meta, colmeta, fammap, bin_tables)
        _write_windowed_parent(w, results_df, grid, cfg, win_idx)
        _write_retrievals(w, results_df, grid, cfg, win_idx)
        if cfg.output.emit_windowed_raw:
            if raw_windowed is None:
                raw_windowed = windowing_raw.aggregate_raw(df, cfg)
            _write_windowed_raw(w, raw_windowed, df, cfg, meta, colmeta, fammap,
                                bin_tables, win_idx, results_df.index)
    finally:
        w.close()

    ca = _clock_alignment_ds(cfg)
    if ca is not None:
        ca.to_netcdf(path, mode="a", group="/clock_alignment")
    return path


def output_filename(cfg):
    return (f"ISARA_{cfg.campaign}_{cfg.year}_{cfg.psd.variant_name}_"
            f"{cfg.window.window_s}s_{cfg.output.version}.nc")
