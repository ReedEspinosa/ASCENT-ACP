"""Export the ASCENT-ACP pipeline products to a grouped CF-1.8 netCDF (v2).

One file per campaign year, organized into netCDF-4 groups (see
NETCDF_OUTPUT_SPEC.md):

  /observations            every raw merged-pickle column at native cadence,
                           split into instrument families, + 1 Hz row_qc_flag
  /windowed                shared 60 s window grid + window_qc_flag + QC counts
  /windowed/retrievals     ISARA retrievals and QC-valid-only measured means
  /windowed/raw/<family>   unconditional 60 s mean/std of every raw column
  /clock_alignment         per (flight_date x shift_group) applied clock shifts

Optical coefficients are stored in m-1 (converted from the Mm-1 of the ICARTT
sources). Retrievals are filled (NaN) and flagged where window QA failed.
"""

import datetime
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import families, results as results_mod, windows as windows_mod
from . import windowing_raw
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

# Best-effort units for raw passthrough variables, by column short-name token.
_UNIT_HINTS = {
    "Latitude": "degrees_north", "Longitude": "degrees_east",
    "GPS_altitude": "m", "Pressure_altitude": "m",
}
_UNIT_SUFFIX = [
    ("_ppm", "ppm"), ("_ppb", "ppb"), ("_ppt", "ppt"),
    ("_cm3", "cm-3"), ("_percent", "percent"), ("_degC", "degC"),
]


# --------------------------------------------------------------------------- #
# provenance helpers (unchanged from v1)
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


def _strip_tz(index):
    return index.tz_localize(None) if getattr(index, "tz", None) else index


# --------------------------------------------------------------------------- #
# /observations  (native cadence, by instrument family)
# --------------------------------------------------------------------------- #
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


def _meta_titles(meta):
    return list((meta or {}).get("Data_Info", {}))


def _observations_tree(df, masks, cfg, meta):
    """Return {group_path: Dataset} for /observations and its family subgroups."""
    fammap = families.load_family_map(cfg.campaign, cfg.paths.family_map_json or None)
    assigned = families.assign_families(list(df.columns), fammap, _meta_titles(meta))
    shift_groups = _shift_group_map(cfg)
    title_meta = meta or {}

    time_obs = _strip_tz(df.index).to_numpy()
    parent = xr.Dataset(coords={"time_obs": ("time_obs", time_obs)})
    parent["time_obs"].attrs["long_name"] = "measurement time at native cadence (UTC)"
    parent["row_qc_flag"] = ("time_obs", _row_qc_flag(masks))
    parent["row_qc_flag"].attrs.update(
        long_name="1 Hz row quality-control bitmask (0 = valid)",
        flag_masks=np.array(sorted(_ROW_QC_MEANINGS), np.int16),
        flag_meanings=" ".join(_ROW_QC_MEANINGS[k] for k in sorted(_ROW_QC_MEANINGS)),
        comment="Kacenelenbogen et al. (2022) A1.1 row screening; see QA_CRITERIA.md",
    )
    # native sampling cadence (don't assume 1 Hz)
    if len(df.index) > 1:
        parent.attrs["native_sampling_seconds"] = float(
            np.median(np.diff(df.index.view("int64"))) / 1e9
        )

    tree = {"/observations": parent}
    by_family = {}
    for col, (fam, title) in assigned.items():
        by_family.setdefault(fam, []).append((col, title))

    for fam in families.family_order(fammap, by_family):
        ds = xr.Dataset()
        for col, title in by_family[fam]:
            short = _short_name(col, title)
            ds[short] = ("time_obs", df[col].to_numpy())
            attrs = {"long_name": short}
            units = _guess_units(short)
            if units:
                attrs["units"] = units
            if col in shift_groups:
                attrs["shift_group"] = shift_groups[col]
            else:
                attrs["shift_group"] = "none"
            attrs["source_column"] = col
            ds[short].attrs.update(attrs)
        ln = families.family_long_name(fammap, fam)
        if ln:
            ds.attrs["long_name"] = ln
        # per-instrument provenance from meta (titles in this family)
        titles = sorted({t for _, t in by_family[fam]})
        for fld in ("PI_Info", "Institution_Info", "Uncertainty", "Revision", "Stipulations"):
            vals = [f"{t}: {title_meta.get(fld, {}).get(t)}" for t in titles
                    if title_meta.get(fld, {}).get(t)]
            if vals:
                ds.attrs[fld] = "\n".join(vals)
        tree[f"/observations/{fam}"] = ds
    return tree


# --------------------------------------------------------------------------- #
# /windowed  (shared 60 s grid) + /windowed/retrievals
# --------------------------------------------------------------------------- #
def _windowed_parent(results_df, grid, cfg):
    ch = cfg.channels
    time = _strip_tz(results_df.index)
    ds = xr.Dataset(coords={"time": ("time", time.to_numpy())})
    ds["time"].attrs.update(long_name="window center time (UTC)", bounds="time_bnds")
    bnds = np.column_stack([
        _strip_tz(pd.DatetimeIndex(results_df["time_start"])),
        _strip_tz(pd.DatetimeIndex(results_df["time_end"])),
    ])
    ds["time_bnds"] = (("time", "nv"), bnds)

    # coordinates shared by retrieval children
    ds = ds.assign_coords(
        wavelength_sca=("wavelength_sca", np.array(ch.dry_wvl_sca, float)),
        wavelength_abs=("wavelength_abs", np.array(ch.dry_wvl_abs, float)),
        dp_mid=("psd_bin", grid.dpg_um),
    )
    ds["wavelength_sca"].attrs.update(units="nm", long_name="nephelometer scattering wavelengths")
    ds["wavelength_abs"].attrs.update(units="nm", long_name="PSAP absorption wavelengths")
    ds["dp_mid"].attrs.update(units="um", long_name="PSD bin center diameter")
    ds["dp_lower"] = ("psd_bin", grid.dpl_um)
    ds["dp_upper"] = ("psd_bin", grid.dpu_um)
    ds["dp_lower"].attrs.update(units="um", long_name="PSD bin lower-bound diameter")
    ds["dp_upper"].attrs.update(units="um", long_name="PSD bin upper-bound diameter")
    ds["psd_instrument"] = ("psd_bin", np.array(grid.instrument, dtype="U8"))
    ds["psd_instrument"].attrs["long_name"] = "source instrument of each PSD bin"

    # window QC flag (the ISARA gate) + reject-reason counts
    flag = results_df["window_qc_flag"].to_numpy(int)
    ds["window_qc_flag"] = ("time", flag)
    ds["window_qc_flag"].attrs.update(
        long_name="window quality control bitmask (0 = good); gates ISARA retrieval",
        flag_masks=np.array(sorted(windows_mod.FLAG_MEANINGS), np.int32),
        flag_meanings=" ".join(windows_mod.FLAG_MEANINGS[k] for k in sorted(windows_mod.FLAG_MEANINGS)),
        comment="Kacenelenbogen et al. (2022) A1.2 window screening; see QA_CRITERIA.md",
    )
    for col, long_name in [
        ("n_valid", "number of valid 1 Hz samples in window"),
        ("n_cloudy", "1 Hz samples rejected as cloud-contaminated"),
        ("n_inlet_bad", "1 Hz samples rejected by inlet flag"),
        ("n_low_signal", "1 Hz samples rejected by minimum dry Sc450 filter"),
        ("n_low_ssa", "1 Hz samples rejected by minimum SSA filter"),
    ]:
        if col in results_df:
            ds[col] = ("time", results_df[col].fillna(0).to_numpy(int))
            ds[col].attrs.update(units="1", long_name=long_name)
    return ds


def _retrievals_ds(results_df, grid, cfg):
    """ISARA retrievals + QC-valid-only measured window means (dim: time)."""
    ch = cfg.channels
    ds = xr.Dataset()

    def stack(cols):
        return np.column_stack([results_df[c].to_numpy(float) for c in cols])

    ds["scattering_dry_measured"] = (
        ("time", "wavelength_sca"),
        stack([f"Sc{w}_dry_mean" for w in ch.dry_wvl_sca]) * _M_PER_MM,
    )
    ds["scattering_dry_measured"].attrs.update(
        units="m-1", cell_methods="time: mean over QC-valid 1 Hz samples",
        long_name=(f"window-mean dry ({cfg.psd.variant_name}) scattering coefficient, "
                   f"gamma-adjusted to {cfg.filters.dry_ref_rh:.0f}% RH where measured above it"),
    )
    ds["scattering_dry_measured_std"] = (
        ("time", "wavelength_sca"),
        stack([f"Sc{w}_dry_std" for w in ch.dry_wvl_sca]) * _M_PER_MM,
    )
    ds["scattering_dry_measured_std"].attrs.update(
        units="m-1", long_name="within-window standard deviation of dry scattering")
    ds["absorption_measured"] = (
        ("time", "wavelength_abs"),
        stack([f"Abs{w}_mean" for w in ch.dry_wvl_abs]) * _M_PER_MM,
    )
    ds["absorption_measured"].attrs.update(
        units="m-1", cell_methods="time: mean over QC-valid 1 Hz samples",
        long_name="window-mean dry bulk absorption coefficient (PSAP)")
    ds["absorption_measured_std"] = (
        ("time", "wavelength_abs"),
        stack([f"Abs{w}_std" for w in ch.dry_wvl_abs]) * _M_PER_MM,
    )
    ds["absorption_measured_std"].attrs.update(
        units="m-1", long_name="within-window standard deviation of absorption")
    wet_w = ch.wet_wvl_sca[0]
    ds["scattering_humidified_synthesized"] = (
        "time", results_df[f"Sc{wet_w}_wet_mean"].to_numpy(float) * _M_PER_MM)
    ds["scattering_humidified_synthesized"].attrs.update(
        units="m-1",
        long_name=(f"window-mean scattering at {wet_w} nm gamma-adjusted to "
                   f"{cfg.filters.wet_rh:.0f}% RH (synthesized, not directly measured)"),
        comment="SC_calcRH = SC_measRH / exp(gamma*ln((100-calcRH)/(100-measRH)))")
    ds["ssa_measured"] = (
        ("time", "wavelength_sca"), stack([f"SSA{w}_mean" for w in ch.dry_wvl_sca]))
    ds["ssa_measured"].attrs.update(
        units="1", long_name="window-mean single scattering albedo (LARGE-derived)")
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
            ds[name] = ("time", results_df[col].to_numpy(float))
            ds[name].attrs.update(units=units, long_name=long_name)

    psd = np.column_stack([results_df[psd_col_name(d)].to_numpy(float) for d in grid.dpg_um])
    ds["dndlogdp"] = (("time", "psd_bin"), psd)
    ds["dndlogdp"].attrs.update(
        units="cm-3", long_name="window-mean dry number size distribution dN/dlogDp (SMPS+LAS, STP)")

    def add_scalar(var, col, units, long_name):
        if col in results_df:
            ds[var] = ("time", results_df[col].to_numpy(float))
            ds[var].attrs.update(units=units, long_name=long_name)

    add_scalar("refractive_index_real", "dry_RRI_unitless", "1",
               "ISARA-retrieved real part of the dry complex refractive index")
    add_scalar("refractive_index_imag", "dry_IRI_unitless", "1",
               "ISARA-retrieved imaginary part of the dry complex refractive index")
    add_scalar("kappa", "kappa_unitless", "1",
               "ISARA-retrieved hygroscopicity parameter (kappa-Kohler, single bulk value)")

    for col, var in [("attempt_flag_CRI_unitless", "attempt_flag_cri"),
                     ("attempt_flag_kappa_unitless", "attempt_flag_kappa")]:
        if col in results_df:
            ds[var] = ("time", results_df[col].fillna(0).to_numpy(int))
            ds[var].attrs.update(
                units="1", long_name=f"ISARA {var.split('_')[-1].upper()} retrieval attempt flag",
                flag_values=np.array([0, 1, 2], np.int32),
                flag_meanings="not_attempted attempted_but_failed success")

    if "retrieval_qc_flag" in results_df:
        ds["retrieval_qc_flag"] = ("time", results_df["retrieval_qc_flag"].to_numpy(int))
        ds["retrieval_qc_flag"].attrs.update(
            units="1", long_name="why a window has or lacks an ISARA retrieval",
            flag_values=np.array(sorted(results_mod.RETRIEVAL_QC_MEANINGS), np.int32),
            flag_meanings=" ".join(results_mod.RETRIEVAL_QC_MEANINGS[k]
                                   for k in sorted(results_mod.RETRIEVAL_QC_MEANINGS)))

    # MOPSMAP-calculated optical properties (one variable per Retr_PSD key)
    for col in results_df.columns:
        m = _RETR_KEY.match(str(col))
        if not m or m.group("kind") == "meas":
            continue
        var = f"{m.group('state')}_calculated_{m.group('quant').lower()}_{m.group('wvl')}nm"
        units = "m-1" if m.group("unit") == "m-1" else "1"
        ds[var] = ("time", results_df[col].to_numpy(float))
        ds[var].attrs.update(
            units=units,
            long_name=(f"MOPSMAP-calculated {m.group('state')} "
                       f"{m.group('quant').replace('_', ' ')} at {m.group('wvl')} nm "
                       "for the retrieved refractive index"
                       + (" and kappa" if m.group("state") == "wet" else "")))
    return ds


# --------------------------------------------------------------------------- #
# /windowed/raw  (unconditional 60 s means of every raw column, by family)
# --------------------------------------------------------------------------- #
def _windowed_raw_tree(raw_windowed, window_index, df_columns, cfg, meta):
    """Return {group_path: Dataset} for /windowed/raw and family subgroups."""
    fammap = families.load_family_map(cfg.campaign, cfg.paths.family_map_json or None)
    assigned = families.assign_families(list(df_columns), fammap, _meta_titles(meta))
    # align onto the retrieval window grid so the shared 'time' coord matches
    rw = raw_windowed.reindex(_strip_tz(window_index))

    tree = {"/windowed/raw": xr.Dataset()}
    if "n_points" in rw:
        npts = xr.Dataset()
        npts["n_points"] = ("time", rw["n_points"].fillna(0).to_numpy(int))
        npts["n_points"].attrs.update(units="1", long_name="number of 1 Hz rows in window (all rows)")
        tree["/windowed/raw"] = npts

    by_family = {}
    for col, (fam, title) in assigned.items():
        by_family.setdefault(fam, []).append((col, title))

    for fam in families.family_order(fammap, by_family):
        ds = xr.Dataset()
        for col, title in by_family[fam]:
            short = _short_name(col, title)
            for stat in ("mean", "std"):
                src = f"{col}_{stat}"
                if src not in rw:
                    continue
                ds[f"{short}_{stat}"] = ("time", rw[src].to_numpy(float))
                ds[f"{short}_{stat}"].attrs.update(
                    long_name=f"60 s window {stat} of {short} (all rows)",
                    cell_methods=f"time: {stat} (all 1 Hz rows in window)",
                    source_column=col)
                u = _guess_units(short)
                if u:
                    ds[f"{short}_{stat}"].attrs["units"] = u
        ln = families.family_long_name(fammap, fam)
        if ln:
            ds.attrs["long_name"] = f"{ln} - 60 s window statistics"
        tree[f"/windowed/raw/{fam}"] = ds
    return tree


# --------------------------------------------------------------------------- #
# /clock_alignment  (date x shift_group)
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
def build_datatree(df, masks, results_df, grid, cfg, meta=None, raw_windowed=None):
    """Assemble the full grouped output as an xr.DataTree."""
    here = Path(__file__).resolve().parent.parent
    root = xr.Dataset()
    root.attrs.update(
        Conventions="CF-1.8",
        title=(f"ISARA aerosol retrievals and merged in-situ observations, "
               f"{cfg.campaign} {cfg.year} ({cfg.psd.variant_name} variant)"),
        institution="NASA GSFC / processed with ASCENT-ACP",
        source=("Airborne in-situ measurements merged from ICARTT files, clock-aligned, "
                "QC-filtered and window-averaged; ISARA/MOPSMAP retrieval"),
        history=(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}: "
                 "created by ASCENT_ACP.netcdf_export"),
        references=("Kacenelenbogen et al. (2022), doi:10.5194/acp-22-3713-2022 (QA method); "
                    "Gasteiger & Wiegner (2018), doi:10.5194/gmd-11-2739-2018 (MOPSMAP)"),
        comment=("Groups: /observations (native-cadence raw passthrough by instrument family), "
                 "/windowed (60 s grid; /retrievals = ISARA + QC-valid means, /raw = all-row means), "
                 "/clock_alignment (time-base provenance). Optical coefficients in m-1. "
                 "All aerosol quantities at STP."),
        source_merged_pickle=str(cfg.paths.input_pkl),
        window_seconds=int(cfg.window.window_s),
        config_json=cfg.to_json(),
        ascent_acp_git_sha=_git_sha(here),
        isara_git_sha=_git_sha(cfg.paths.isara_code_dir),
        instrument_metadata=_instrument_metadata_text(meta),
    )

    nodes = {"/": root}
    if cfg.output.emit_observations:
        nodes.update(_observations_tree(df, masks, cfg, meta))

    nodes["/windowed"] = _windowed_parent(results_df, grid, cfg)
    nodes["/windowed/retrievals"] = _retrievals_ds(results_df, grid, cfg)

    if cfg.output.emit_windowed_raw:
        if raw_windowed is None:
            raw_windowed = windowing_raw.aggregate_raw(df, cfg)
        nodes.update(_windowed_raw_tree(raw_windowed, results_df.index, df.columns, cfg, meta))

    ca = _clock_alignment_ds(cfg)
    if ca is not None:
        nodes["/clock_alignment"] = ca

    return xr.DataTree.from_dict(nodes)


_TIME_ENC = {"units": "seconds since 1970-01-01T00:00:00"}


def _apply_encoding(dt, cfg):
    """Set per-variable .encoding in place (float32 + zlib, time units)."""
    comp = cfg.output.compression_level
    for path in dt.groups:
        ds = dt[path].dataset
        for name, da in ds.data_vars.items():
            if cfg.output.float32 and np.issubdtype(da.dtype, np.floating):
                da.encoding.update(dtype="float32", zlib=comp > 0, complevel=comp)
            elif np.issubdtype(da.dtype, np.integer):
                da.encoding.update(zlib=comp > 0, complevel=comp)
        for tname in ("time", "time_obs"):
            if tname in ds.coords and tname in ds.variables:
                dt[path][tname].encoding.update(_TIME_ENC)


def write(dt, path, cfg):
    """Write the DataTree to a grouped netCDF-4 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _apply_encoding(dt, cfg)
    dt.to_netcdf(path)
    return path


def to_dataset(results_df, grid, cfg, meta=None):
    """Back-compat shim: the retrievals-only flat dataset (no groups)."""
    parent = _windowed_parent(results_df, grid, cfg)
    retr = _retrievals_ds(results_df, grid, cfg)
    return xr.merge([parent, retr])


def output_filename(cfg):
    return (f"ISARA_{cfg.campaign}_{cfg.year}_{cfg.psd.variant_name}_"
            f"{cfg.window.window_s}s_{cfg.output.version}.nc")
