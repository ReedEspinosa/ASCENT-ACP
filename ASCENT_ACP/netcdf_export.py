"""Export assembled ISARA results to a CF-1.8-style netCDF file.

One file per campaign year. Measured window means, their standard deviations
and QC bookkeeping sit alongside the retrieved refractive index, kappa and
MOPSMAP-calculated optical properties. All optical coefficients are in m-1
(converted from the Mm-1 of the ICARTT sources).
"""

import datetime
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import windows as windows_mod
from .windows import psd_col_name

_M_PER_MM = 1.0e-6  # Mm-1 -> m-1

# Retr_PSD output keys that carry a wavelength, e.g. dry_cal_sca_coef_550_m-1
_RETR_KEY = re.compile(
    r"^(?P<state>dry|wet)_(?P<kind>cal|meas)_(?P<quant>sca_coef|abs_coef|ext_coef|SSA)"
    r"_(?P<wvl>\d+)_(?P<unit>m-1|unitless)$"
)


def _git_sha(repo_dir):
    try:
        return (
            subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
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


def to_dataset(results_df, grid, cfg, meta=None):
    ch = cfg.channels
    time = results_df.index.tz_localize(None) if results_df.index.tz else results_df.index
    ds = xr.Dataset(coords={"time": ("time", time.to_numpy())})
    ds["time"].attrs["long_name"] = "window center time (UTC)"

    bnds = np.column_stack(
        [
            pd.DatetimeIndex(results_df["time_start"]).tz_localize(None),
            pd.DatetimeIndex(results_df["time_end"]).tz_localize(None),
        ]
    )
    ds["time_bnds"] = (("time", "nv"), bnds)
    ds["time"].attrs["bounds"] = "time_bnds"

    wvl_sca = np.array(ch.dry_wvl_sca, float)
    wvl_abs = np.array(ch.dry_wvl_abs, float)
    ds = ds.assign_coords(
        wavelength_sca=("wavelength_sca", wvl_sca),
        wavelength_abs=("wavelength_abs", wvl_abs),
        dp_mid=("psd_bin", grid.dpg_um),
    )
    ds["wavelength_sca"].attrs.update(units="nm", long_name="nephelometer scattering wavelengths")
    ds["wavelength_abs"].attrs.update(units="nm", long_name="PSAP absorption wavelengths")
    ds["dp_mid"].attrs.update(units="um", long_name="PSD bin center diameter")
    ds["dp_lower"] = ("psd_bin", grid.dpl_um)
    ds["dp_upper"] = ("psd_bin", grid.dpu_um)
    ds["dp_lower"].attrs.update(units="um", long_name="PSD bin lower-bound diameter")
    ds["dp_upper"].attrs.update(units="um", long_name="PSD bin upper-bound diameter")
    ds["psd_instrument"] = (
        "psd_bin",
        np.array(grid.instrument, dtype="U8"),
    )
    ds["psd_instrument"].attrs["long_name"] = "source instrument of each PSD bin"

    def stack(cols):
        return np.column_stack([results_df[c].to_numpy(float) for c in cols])

    # --- measured optical properties (window means / stds) ---
    ds["scattering_dry_measured"] = (
        ("time", "wavelength_sca"),
        stack([f"Sc{w}_dry_mean" for w in ch.dry_wvl_sca]) * _M_PER_MM,
    )
    ds["scattering_dry_measured"].attrs.update(
        units="m-1",
        long_name=(
            f"window-mean dry ({cfg.psd.variant_name}) scattering coefficient, "
            f"gamma-adjusted to {cfg.filters.dry_ref_rh:.0f}% RH where measured above it"
        ),
    )
    ds["scattering_dry_measured_std"] = (
        ("time", "wavelength_sca"),
        stack([f"Sc{w}_dry_std" for w in ch.dry_wvl_sca]) * _M_PER_MM,
    )
    ds["scattering_dry_measured_std"].attrs.update(
        units="m-1", long_name="within-window standard deviation of dry scattering"
    )
    ds["absorption_measured"] = (
        ("time", "wavelength_abs"),
        stack([f"Abs{w}_mean" for w in ch.dry_wvl_abs]) * _M_PER_MM,
    )
    ds["absorption_measured"].attrs.update(
        units="m-1", long_name="window-mean dry bulk absorption coefficient (PSAP)"
    )
    ds["absorption_measured_std"] = (
        ("time", "wavelength_abs"),
        stack([f"Abs{w}_std" for w in ch.dry_wvl_abs]) * _M_PER_MM,
    )
    ds["absorption_measured_std"].attrs.update(
        units="m-1", long_name="within-window standard deviation of absorption"
    )
    wet_w = ch.wet_wvl_sca[0]
    ds["scattering_humidified_synthesized"] = (
        "time",
        results_df[f"Sc{wet_w}_wet_mean"].to_numpy(float) * _M_PER_MM,
    )
    ds["scattering_humidified_synthesized"].attrs.update(
        units="m-1",
        long_name=(
            f"window-mean scattering at {wet_w} nm gamma-adjusted to "
            f"{cfg.filters.wet_rh:.0f}% RH (synthesized, not directly measured)"
        ),
        comment="SC_calcRH = SC_measRH / exp(gamma*ln((100-calcRH)/(100-measRH)))",
    )
    ds["ssa_measured"] = (
        ("time", "wavelength_sca"),
        stack([f"SSA{w}_mean" for w in ch.dry_wvl_sca]),
    )
    ds["ssa_measured"].attrs.update(
        units="1", long_name="window-mean single scattering albedo (LARGE-derived)"
    )
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
        ds[name] = ("time", results_df[col].to_numpy(float))
        ds[name].attrs.update(units=units, long_name=long_name)

    # --- PSD ---
    psd = np.column_stack([results_df[psd_col_name(d)].to_numpy(float) for d in grid.dpg_um])
    ds["dndlogdp"] = (("time", "psd_bin"), psd)
    ds["dndlogdp"].attrs.update(
        units="cm-3",
        long_name="window-mean dry number size distribution dN/dlogDp (SMPS+LAS, STP)",
    )

    # --- QC bookkeeping ---
    flag = results_df["window_qc_flag"].to_numpy(int)
    ds["window_qc_flag"] = ("time", flag)
    ds["window_qc_flag"].attrs.update(
        long_name="window quality control bitmask (0 = good)",
        flag_masks=np.array(sorted(windows_mod.FLAG_MEANINGS), np.int32),
        flag_meanings=" ".join(
            windows_mod.FLAG_MEANINGS[k] for k in sorted(windows_mod.FLAG_MEANINGS)
        ),
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

    # --- retrieved quantities ---
    def add_scalar(var, col, units, long_name):
        if col in results_df:
            ds[var] = ("time", results_df[col].to_numpy(float))
            ds[var].attrs.update(units=units, long_name=long_name)

    add_scalar(
        "refractive_index_real", "dry_RRI_unitless", "1",
        "ISARA-retrieved real part of the dry complex refractive index",
    )
    add_scalar(
        "refractive_index_imag", "dry_IRI_unitless", "1",
        "ISARA-retrieved imaginary part of the dry complex refractive index",
    )
    add_scalar(
        "kappa", "kappa_unitless", "1",
        "ISARA-retrieved hygroscopicity parameter (kappa-Kohler, single bulk value)",
    )
    for col, var in [
        ("attempt_flag_CRI_unitless", "attempt_flag_cri"),
        ("attempt_flag_kappa_unitless", "attempt_flag_kappa"),
    ]:
        if col in results_df:
            ds[var] = ("time", results_df[col].fillna(0).to_numpy(int))
            ds[var].attrs.update(
                units="1",
                long_name=f"ISARA {var.split('_')[-1].upper()} retrieval attempt flag",
                flag_values=np.array([0, 1, 2], np.int32),
                flag_meanings="not_attempted attempted_but_failed success",
            )

    # MOPSMAP-calculated optical properties (one variable per Retr_PSD key)
    for col in results_df.columns:
        m = _RETR_KEY.match(str(col))
        if not m or m.group("kind") == "meas":
            continue  # measured echoes already covered above
        var = f"{m.group('state')}_calculated_{m.group('quant').lower()}_{m.group('wvl')}nm"
        units = "m-1" if m.group("unit") == "m-1" else "1"
        ds[var] = ("time", results_df[col].to_numpy(float))
        ds[var].attrs.update(
            units=units,
            long_name=(
                f"MOPSMAP-calculated {m.group('state')} "
                f"{m.group('quant').replace('_', ' ')} at {m.group('wvl')} nm "
                "for the retrieved refractive index"
                + (" and kappa" if m.group("state") == "wet" else "")
            ),
        )

    # --- global attributes ---
    here = Path(__file__).resolve().parent.parent
    ds.attrs.update(
        Conventions="CF-1.8",
        title=(
            f"ISARA aerosol refractive index and hygroscopicity retrievals, "
            f"{cfg.campaign} {cfg.year} ({cfg.psd.variant_name} variant)"
        ),
        institution="NASA GSFC / processed with ASCENT-ACP",
        source=(
            "Airborne in-situ measurements (NASA LARGE: TSI-3563 nephelometer, "
            "PSAP, SMPS, TSI-3340 LAS) merged from ICARTT files, clock-aligned, "
            "QC-filtered and window-averaged; ISARA/MOPSMAP retrieval"
        ),
        history=(
            f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}: "
            "created by ASCENT_ACP.pipeline"
        ),
        references=(
            "Kacenelenbogen et al. (2022), doi:10.5194/acp-22-3713-2022 (filtering method); "
            "Gasteiger & Wiegner (2018), doi:10.5194/gmd-11-2739-2018 (MOPSMAP)"
        ),
        comment=(
            "Optical coefficients in m-1 (ICARTT sources are Mm-1). "
            "Humidified scattering is synthesized from gamma550, not directly measured. "
            "PSAP absorption is bulk; supermicron absorption assumed negligible. "
            f"PSD truncated at {min(cfg.psd.psd_max_um, cfg.psd.inlet_cutoff_um)} um; "
            "the 1 um cyclone cut is aerodynamic while LAS bins are optical diameter - "
            "treated as equivalent. All aerosol quantities at STP."
        ),
        source_merged_pickle=str(cfg.paths.input_pkl),
        window_seconds=int(cfg.window.window_s),
        config_json=cfg.to_json(),
        ascent_acp_git_sha=_git_sha(here),
        isara_git_sha=_git_sha(cfg.paths.isara_code_dir),
        instrument_metadata=_instrument_metadata_text(meta),
    )
    return ds


def write(ds, path):
    """Write with float32 + zlib compression for all float data variables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[name] = {"dtype": "float32", "zlib": True, "complevel": 4}
        elif np.issubdtype(da.dtype, np.integer):
            encoding[name] = {"zlib": True, "complevel": 4}
    if "time" in ds.coords:
        encoding["time"] = {"units": "seconds since 1970-01-01T00:00:00"}
    ds.to_netcdf(path, encoding=encoding)
    return path


def output_filename(cfg):
    return (
        f"ISARA_{cfg.campaign}_{cfg.year}_{cfg.psd.variant_name}_"
        f"{cfg.window.window_s}s_V1.nc"
    )
