"""
Test script to check instrument clock alignment by calculating cross-correlation
between LAS and SMPS bin sums with time shifts.

This script:
1. Loads the merged ICARTT data
2. Sums all LAS bins and all SMPS bins separately
3. Calculates cross-correlation between LAS sum and SMPS sum with shifts of ±15 time steps
4. Plots autocorrelation as a function of shift for each date separately
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the icartt_read_and_merge package to path
sys.path.insert(0, '/Users/wrespino/Synced/Local_Code_MacBook/icartt_read_and_merge')

from ASCENT_ACP.ascent_acp import run_ascent_acp_merge
from icartt_read_and_merge.ancillary_utils import read_size_distribution_radii


def identify_bin_columns(df):
    """Identify LAS and SMPS bin columns in the dataframe."""
    las_bins = [col for col in df.columns if 'LAS' in col.upper() and 'BIN' in col.upper()]
    smps_bins = [col for col in df.columns if 'SMPS' in col.upper() and 'BIN' in col.upper()]

    # Also check for patterns like "Bin01", "Bin02", etc. with LAS/SMPS prefix
    if not las_bins:
        las_bins = [col for col in df.columns if col.startswith('LAS_') and 'Bin' in col]
    if not smps_bins:
        smps_bins = [col for col in df.columns if col.startswith('SMPS_') and 'Bin' in col]

    # Check for patterns without underscore
    if not las_bins:
        las_bins = [col for col in df.columns if 'LAS' in col and any(f'Bin{i:02d}' in col for i in range(1, 100))]
    if not smps_bins:
        smps_bins = [col for col in df.columns if 'SMPS' in col and any(f'Bin{i:02d}' in col for i in range(1, 100))]

    return las_bins, smps_bins


def wavelength_to_rgb(wavelength_nm):
    """
    Convert wavelength (nm) to RGB color tuple.

    Uses approximate visible spectrum mapping:
    - 400 nm → blue (0, 0, 1)
    - 550 nm → green (0, 1, 0)
    - 700 nm → red (1, 0, 0)
    - Interpolates linearly between these anchor points

    Returns RGB tuple with values in [0, 1].
    """
    if wavelength_nm <= 400:
        return (0.0, 0.0, 1.0)  # Blue
    elif wavelength_nm <= 550:
        # Interpolate from blue to green
        t = (wavelength_nm - 400) / (550 - 400)
        r = 0.0
        g = t
        b = 1.0 - t
        return (r, g, b)
    elif wavelength_nm <= 700:
        # Interpolate from green to red
        t = (wavelength_nm - 550) / (700 - 550)
        r = t
        g = 1.0 - t
        b = 0.0
        return (r, g, b)
    else:
        return (1.0, 0.0, 0.0)  # Red


def angstrom_extrapolate(value_at_wl, from_wavelength, to_wavelength, angstrom_exponent=1.0):
    """
    Extrapolate optical property to new wavelength using Ångström exponent.

    value(λ) = value(λ₀) × (λ/λ₀)^(-α)

    Parameters
    ----------
    value_at_wl : np.ndarray or pd.Series
        Optical property value at reference wavelength
    from_wavelength : float
        Reference wavelength (nm)
    to_wavelength : float
        Target wavelength (nm)
    angstrom_exponent : float
        Ångström exponent (default 1.0)

    Returns
    -------
    np.ndarray or pd.Series
        Extrapolated values at target wavelength
    """
    wavelength_ratio = to_wavelength / from_wavelength
    return value_at_wl * (wavelength_ratio ** (-angstrom_exponent))


def find_matching_optical_columns(df, scattering_columns, property_name='Ext'):
    """
    Find extinction or absorption columns that match scattering column types.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing optical measurements
    scattering_columns : dict
        {wavelength_nm: column_name} for scattering
    property_name : str
        'Ext' for extinction or 'Abs' for absorption

    Returns
    -------
    dict
        {wavelength_nm: {matched_wavelength: column_name}} for each scattering wavelength
    """
    # Find all columns for this property
    all_cols = [col for col in df.columns if 'optical_aerosol' in col and property_name in col]

    # Extract wavelength and type from column names
    property_data = {}  # {wavelength: {type: column_name}}
    for col in all_cols:
        # Extract wavelength (e.g., "Ext532" or "Abs470")
        import re
        match = re.search(rf'{property_name}(\d+)', col)
        if match:
            wl = int(match.group(1))
            # Determine type (total, submicron, ambient)
            if 'total_amb' in col.lower():
                col_type = 'total_amb'
            elif 'submicron_amb' in col.lower():
                col_type = 'submicron_amb'
            elif 'total' in col.lower():
                col_type = 'total'
            elif 'submicron' in col.lower():
                col_type = 'submicron'
            else:
                col_type = 'unknown'

            if wl not in property_data:
                property_data[wl] = {}
            property_data[wl][col_type] = col

    # Match each scattering wavelength
    matched = {}
    for sc_wl, sc_col in scattering_columns.items():
        # Determine scattering type
        if 'total_amb' in sc_col.lower():
            sc_type = 'total_amb'
        elif 'submicron_amb' in sc_col.lower():
            sc_type = 'submicron_amb'
        elif 'total' in sc_col.lower():
            sc_type = 'total'
        elif 'submicron' in sc_col.lower():
            sc_type = 'submicron'
        else:
            sc_type = 'total'  # default

        matched[sc_wl] = {}

        # Find matching wavelengths within 50 nm
        for prop_wl in property_data.keys():
            if abs(prop_wl - sc_wl) <= 50:
                # Try to match type, fall back to total
                if sc_type in property_data[prop_wl]:
                    matched[sc_wl][prop_wl] = property_data[prop_wl][sc_type]
                elif 'total' in property_data[prop_wl]:
                    matched[sc_wl][prop_wl] = property_data[prop_wl]['total']
                elif property_data[prop_wl]:  # Any available type
                    matched[sc_wl][prop_wl] = list(property_data[prop_wl].values())[0]

    return matched


def compute_ssa(df, sc_columns, ext_matched, abs_matched):
    """
    Compute single scattering albedo (SSA) two ways.

    SSA_method1 = scattering / extinction (dashed line)
    SSA_method2 = scattering / (absorption + scattering) (solid line)

    Uses Ångström exponent interpolation/extrapolation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with optical measurements
    sc_columns : dict
        {wavelength_nm: column_name} for scattering
    ext_matched : dict
        {sc_wl: {ext_wl: column_name}}
    abs_matched : dict
        {sc_wl: {abs_wl: column_name}}

    Returns
    -------
    dict
        {wavelength_nm: {'method1': pd.Series, 'method2': pd.Series}}
    """
    ssa_dict = {}

    for sc_wl, sc_col in sorted(sc_columns.items()):
        sc_data = df[sc_col].values

        # Get extinction at this wavelength
        ext_at_wl = None
        if sc_wl in ext_matched and ext_matched[sc_wl]:
            ext_wls = list(ext_matched[sc_wl].keys())
            if ext_wls:
                # Use closest wavelength
                closest_ext_wl = min(ext_wls, key=lambda x: abs(x - sc_wl))
                ext_col = ext_matched[sc_wl][closest_ext_wl]
                ext_data = df[ext_col].values
                # Extrapolate to scattering wavelength
                ext_at_wl = angstrom_extrapolate(ext_data, closest_ext_wl, sc_wl, angstrom_exponent=1.0)

        # Get absorption at this wavelength
        abs_at_wl = None
        if sc_wl in abs_matched and abs_matched[sc_wl]:
            abs_wls = list(abs_matched[sc_wl].keys())
            if abs_wls:
                # Use closest wavelength
                closest_abs_wl = min(abs_wls, key=lambda x: abs(x - sc_wl))
                abs_col = abs_matched[sc_wl][closest_abs_wl]
                abs_data = df[abs_col].values
                # Extrapolate to scattering wavelength
                abs_at_wl = angstrom_extrapolate(abs_data, closest_abs_wl, sc_wl, angstrom_exponent=1.0)

        # Compute SSA
        ssa_dict[sc_wl] = {}

        # Method 1: scattering / extinction
        if ext_at_wl is not None:
            ssa_method1 = sc_data / ext_at_wl
            ssa_method1[ext_at_wl <= 0] = np.nan
            ssa_dict[sc_wl]['method1'] = pd.Series(ssa_method1, index=df.index)
        else:
            ssa_dict[sc_wl]['method1'] = pd.Series(np.nan, index=df.index)

        # Method 2: scattering / (absorption + scattering)
        if abs_at_wl is not None:
            denom = abs_at_wl + sc_data
            ssa_method2 = sc_data / denom
            ssa_method2[denom <= 0] = np.nan
            ssa_dict[sc_wl]['method2'] = pd.Series(ssa_method2, index=df.index)
        else:
            ssa_dict[sc_wl]['method2'] = pd.Series(np.nan, index=df.index)

    return ssa_dict


def compute_geometric_cross_section(df, las_bins, smps_bins, radii_dict,
                                     sc550_label, bin_data_is_dNdlogDp=True):
    """
    Compute total geometric cross-section σ_geo from the particle size distribution.

    σ_geo = π × Σ r_i² × N_i, converted to Mm⁻¹.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with bin columns.
    las_bins : list of str
        LAS bin column names.
    smps_bins : list of str
        SMPS bin column names (may be empty).
    radii_dict : dict
        {'LAS': [r1, r2, ...], 'SMPS': [r1, r2, ...]} in nm.
    sc550_label : str
        Label used for Sc550 — checked for 'submicron' to apply truncation.
    bin_data_is_dNdlogDp : bool
        If True, convert dN/dlogDp to N per bin before computing.

    Returns
    -------
    np.ndarray
        σ_geo in Mm⁻¹, same length as df.
    """
    # Build (radius_nm, column_name) pairs for each instrument
    # NOTE: radii_dict contains DIAMETERS in nm (from ICARTT "Mid Points" headers);
    # convert to radii by dividing by 2.
    las_diameters = radii_dict.get('LAS', [])
    smps_diameters = radii_dict.get('SMPS', [])
    las_radii = [d / 2.0 for d in las_diameters]
    smps_radii = [d / 2.0 for d in smps_diameters]

    assert len(las_radii) == len(las_bins), (
        f"LAS radii count ({len(las_radii)}) != LAS bin count ({len(las_bins)})")
    if smps_bins:
        assert len(smps_radii) == len(smps_bins), (
            f"SMPS radii count ({len(smps_radii)}) != SMPS bin count ({len(smps_bins)})")

    las_pairs = list(zip(las_radii, las_bins))
    smps_pairs = list(zip(smps_radii, smps_bins)) if smps_bins and smps_radii else []

    # Handle SMPS/LAS overlap: keep all SMPS, keep LAS bins above max SMPS radius
    if smps_pairs:
        max_smps_r = max(r for r, _ in smps_pairs)
        las_pairs = [(r, col) for r, col in las_pairs if r > max_smps_r]
        combined = smps_pairs + las_pairs
    else:
        combined = las_pairs

    # Submicron truncation
    if 'submicron' in sc550_label.lower():
        combined = [(r, col) for r, col in combined if r <= 500.0]

    # Sort by radius
    combined.sort(key=lambda x: x[0])

    # Extrapolation when SMPS absent: add synthetic bins below smallest LAS radius
    if not smps_pairs and combined:
        r_min = combined[0][0]
        if r_min > 1.0:
            synth_radii = np.geomspace(1.0, r_min * 0.99, num=5)
            smallest_col = combined[0][1]
            # We'll handle synthetic bin concentrations below after extracting N_matrix
            synth_cols = [f'__synth_{i}' for i in range(len(synth_radii))]
            synth_pairs = list(zip(synth_radii.tolist(), synth_cols))
            combined = synth_pairs + combined
        else:
            synth_cols = []
    else:
        synth_cols = []

    if not combined:
        return np.full(len(df), np.nan)

    radii_nm = np.array([r for r, _ in combined])
    bin_columns = [col for _, col in combined]

    # Build N_matrix
    real_cols = [c for c in bin_columns if not c.startswith('__synth_')]
    synth_mask = np.array([c.startswith('__synth_') for c in bin_columns])

    N_matrix = np.full((len(df), len(bin_columns)), np.nan)

    # Fill real columns
    real_indices = [i for i, c in enumerate(bin_columns) if not c.startswith('__synth_')]
    for j, idx in enumerate(real_indices):
        N_matrix[:, idx] = df[real_cols[j]].values

    # Fill synthetic columns via linear interpolation from smallest real bin to 0
    if synth_cols:
        smallest_real_idx = real_indices[0]
        smallest_real_vals = N_matrix[:, smallest_real_idx]
        n_synth = synth_mask.sum()
        for k in range(n_synth):
            frac = radii_nm[k] / radii_nm[smallest_real_idx]  # 0 at r=0, 1 at smallest real
            N_matrix[:, k] = smallest_real_vals * frac

    # Convert dN/dlogDp to N per bin if needed
    if bin_data_is_dNdlogDp:
        diameters = 2.0 * radii_nm  # nm
        n_bins = len(diameters)
        # Compute bin edges as geometric means of adjacent mid-points
        bin_edges = np.zeros(n_bins + 1)
        for i in range(1, n_bins):
            bin_edges[i] = np.sqrt(diameters[i - 1] * diameters[i])
        # Extrapolate outer edges
        if n_bins > 1:
            bin_edges[0] = diameters[0] ** 2 / bin_edges[1]
            bin_edges[-1] = diameters[-1] ** 2 / bin_edges[-2]
        else:
            bin_edges[0] = diameters[0] * 0.5
            bin_edges[-1] = diameters[0] * 2.0
        delta_logDp = np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        # N_i = (dN/dlogDp)_i * delta_logDp_i
        N_matrix = N_matrix * delta_logDp[np.newaxis, :]

    # Compute σ_geo in Mm⁻¹ (vectorized)
    # r in nm -> cm: x1e-7, r² in cm²: x1e-14; N in #/cm³; product cm⁻¹; x1e8 for Mm⁻¹; net x1e-6
    r_squared = radii_nm ** 2  # nm²

    # NaN handling: use nansum but flag rows with >50% NaN
    nan_frac = np.isnan(N_matrix).sum(axis=1) / N_matrix.shape[1]
    sigma_geo = np.pi * 1e-6 * np.nansum(N_matrix * r_squared[np.newaxis, :], axis=1)
    sigma_geo[nan_frac > 0.5] = np.nan

    return sigma_geo


def compute_scattering_efficiency(sc550, sigma_geo):
    """
    Compute empirical scattering efficiency SE = σ_measured / σ_geo.

    Parameters
    ----------
    sc550 : np.ndarray
        Measured scattering in Mm⁻¹.
    sigma_geo : np.ndarray
        Geometric cross-section in Mm⁻¹.

    Returns
    -------
    np.ndarray
        Dimensionless scattering efficiency.
    """
    sc550_vals = np.asarray(sc550, dtype=float)
    sigma_geo_vals = np.asarray(sigma_geo, dtype=float)
    se = sc550_vals / sigma_geo_vals
    se[sigma_geo_vals <= 0] = np.nan
    return se


def calculate_cross_correlation(las_sum, sc550, max_shift=30):
    """
    Calculate cross-correlation between LAS sum and Sc550 scattering with time shifts.

    Uses a consistent window for all shifts to avoid edge effects.

    Parameters:
    -----------
    las_sum : pd.Series
        Sum of LAS bins
    sc550 : pd.Series
        Sc550 aerosol scattering at 550nm
    max_shift : int
        Maximum shift in time steps (default: 30)

    Returns:
    --------
    shifts : np.array
        Array of shift values (-max_shift to +max_shift)
    correlations : np.array
        Array of correlation coefficients for each shift
    dot_products : np.array
        Array of normalized dot products for each shift
    n_valid_points : np.array
        Array of number of valid data points used for each correlation

    Notes:
    ------
    Edge effect handling:
    - To ensure fair comparison, we exclude the first and last max_shift points
      from ALL calculations, so every shift uses the same middle portion
    - This prevents artificial correlation improvements at non-zero shifts
    """
    # Align indices (use intersection of valid data)
    common_idx = las_sum.index.intersection(sc550.index)
    las_full = las_sum.loc[common_idx].values
    sc550_full = sc550.loc[common_idx].values
    n_total = len(las_full)

    if n_total < max_shift * 3:
        print(f"  Warning: Not enough data points ({n_total}) for max_shift={max_shift}")
        return np.array([]), np.array([]), np.array([])

    # LAS reference window: always the same middle portion
    las_ref = las_full[max_shift : n_total - max_shift]
    window_size = len(las_ref)
    print(f"  Fixed LAS window: {window_size} points (indices {max_shift} to {n_total - max_shift})")

    shifts = np.arange(-max_shift, max_shift + 1)
    correlations = np.zeros(len(shifts))
    dot_products = np.zeros(len(shifts))
    n_valid_points = np.zeros(len(shifts), dtype=int)

    for i, shift in enumerate(shifts):
        # Sc550 window slides: positive shift means Sc550 taken from later in time
        sc550_window = sc550_full[max_shift + shift : n_total - max_shift + shift]

        # Both windows are always the same length
        both_valid = ~np.isnan(las_ref) & ~np.isnan(sc550_window)
        las_vals = las_ref[both_valid]
        sc550_vals = sc550_window[both_valid]

        n_valid_points[i] = len(las_vals)

        if len(las_vals) > 1:
            if np.std(las_vals) == 0 or np.std(sc550_vals) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(las_vals, sc550_vals)[0, 1]
            dot_prod = np.dot(las_vals, sc550_vals) / len(las_vals)
        else:
            corr = np.nan
            dot_prod = np.nan

        correlations[i] = corr
        dot_products[i] = dot_prod

    # Print summary
    print(f"  Correlation range: {np.nanmin(correlations):.6f} to {np.nanmax(correlations):.6f}")
    print(f"  Shift at max correlation: {shifts[np.nanargmax(correlations)]}")
    print(f"  Shift at max dot product: {shifts[np.nanargmax(dot_products)]}")
    print(f"  n_valid range: {n_valid_points.min()} to {n_valid_points.max()}")

    return shifts, correlations, dot_products, n_valid_points


def plot_time_series(date, las_sum, smps_sum, sc_data=None,
                     output_dir=None, zoom_label='', scattering_efficiency_dict=None,
                     ssa_dict=None, abs_data=None):
    """
    Plot time series of LAS sum, SMPS sum, and optionally scattering with multiple y-axes.
    When scattering_efficiency_dict is provided, adds a panel showing SE for all wavelengths.
    When ssa_dict is provided, adds a panel showing SSA for all wavelengths.
    When abs_data is provided, adds a panel showing absorption for all wavelengths.

    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    las_sum : pd.Series
        Time series of LAS bin sum
    smps_sum : pd.Series
        Time series of SMPS bin sum
    sc_data : dict, optional
        Dictionary mapping wavelength (nm) to scattering time series
    output_dir : str, optional
        Directory to save plot
    zoom_label : str, optional
        Additional label for zoomed plots (e.g., '_zoom')
    scattering_efficiency_dict : dict, optional
        Dictionary mapping wavelength (nm) to SE time series (dimensionless)
    ssa_dict : dict, optional
        Dictionary mapping wavelength (nm) to {'method1': Series, 'method2': Series}
    abs_data : dict, optional
        Dictionary mapping wavelength (nm) to absorption time series (Mm⁻¹)
    """
    # Determine number of panels
    n_panels = 1
    if scattering_efficiency_dict is not None:
        n_panels += 1
    if ssa_dict is not None:
        n_panels += 1
    if abs_data is not None:
        n_panels += 1

    if n_panels == 1:
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax_se = None
        ax_ssa = None
        ax_abs = None
    elif n_panels == 2:
        fig, (ax1, ax_se) = plt.subplots(2, 1, figsize=(14, 9),
                                          height_ratios=[2, 1], sharex=True)
        ax_ssa = None
        ax_abs = None
    elif n_panels == 3:
        fig, (ax1, ax_se, ax_ssa) = plt.subplots(3, 1, figsize=(14, 12),
                                                   height_ratios=[2, 1, 1], sharex=True)
        ax_abs = None
    else:  # n_panels == 4
        fig, (ax1, ax_se, ax_ssa, ax_abs) = plt.subplots(4, 1, figsize=(14, 15),
                                                           height_ratios=[2, 1, 1, 1], sharex=True)

    # Left y-axis: LAS sum
    color1 = 'tab:blue'
    ax1.set_ylabel('LAS Sum (#/cm³)', color=color1, fontsize=12)
    line1 = ax1.plot(las_sum.index, las_sum.values, '-', color=color1,
                     linewidth=1.5, alpha=0.7, label='LAS Sum')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    # Set minimum to ~5% below zero for visual spacing
    las_max = las_sum.max() if las_sum.notna().any() else 1
    ax1.set_ylim(bottom=-0.05 * las_max, top=las_max * 1.05)

    # Right y-axis: SMPS sum
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('SMPS Sum (#/cm³)', color=color2, fontsize=12)
    line2 = ax2.plot(smps_sum.index, smps_sum.values, '-', color=color2,
                     linewidth=1.5, alpha=0.7, label='SMPS Sum')
    ax2.tick_params(axis='y', labelcolor=color2)
    # Set minimum to ~5% below zero for visual spacing
    smps_max = smps_sum.max() if smps_sum.notna().any() else 1
    ax2.set_ylim(bottom=-0.05 * smps_max, top=smps_max * 1.05)

    lines = line1 + line2

    # Third y-axis: Scattering at middle wavelength only (if provided)
    if sc_data is not None and len(sc_data) > 0:
        ax3 = ax1.twinx()
        # Offset the third axis to the right
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.set_ylabel('Scattering (Mm⁻¹)', fontsize=12)

        # Use 550 nm (green) as the reference wavelength, or middle wavelength if not available
        wavelengths = sorted(sc_data.keys())
        if 550 in wavelengths:
            ref_wavelength = 550
        else:
            ref_wavelength = wavelengths[len(wavelengths) // 2]

        sc_ref = sc_data[ref_wavelength]
        sc_max = sc_ref.max() if sc_ref.notna().any() else 1
        ax3.set_ylim(bottom=-0.05 * sc_max, top=sc_max * 1.05)

        # Plot only the reference wavelength
        color = wavelength_to_rgb(ref_wavelength)
        line3 = ax3.plot(sc_ref.index, sc_ref.values, '-', color=color,
                       linewidth=1.5, alpha=0.7, label=f'Sc{ref_wavelength}')
        lines = lines + line3

        ax3.tick_params(axis='y', labelcolor='black')

    # Title
    title = f'LAS, SMPS, and Scattering Time Series: {date}'
    if zoom_label:
        title += ' (Zoomed)'
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Middle panel: Scattering Efficiency
    if scattering_efficiency_dict is not None and len(scattering_efficiency_dict) > 0 and ax_se is not None:
        # Plot SE for each wavelength with wavelength-based colors
        for wavelength in sorted(scattering_efficiency_dict.keys()):
            se_series = scattering_efficiency_dict[wavelength]
            color = wavelength_to_rgb(wavelength)

            if isinstance(se_series, pd.Series):
                se_index = se_series.index
                se_values = se_series.values
            else:
                se_index = las_sum.index
                se_values = se_series

            ax_se.plot(se_index, se_values, '-', color=color,
                      linewidth=1.5, alpha=0.7, label=f'SE @ {wavelength} nm')

        ax_se.set_ylabel('Scattering Efficiency', fontsize=12)
        if ax_ssa is None:
            ax_se.set_xlabel('Time (UTC)', fontsize=12)
        ax_se.set_ylim(-0.3, 6.0)  # Fixed y-limits as requested
        ax_se.grid(True, alpha=0.3)
        ax_se.legend(loc='upper left', fontsize=9)

    # SSA panel: Single Scattering Albedo
    if ssa_dict is not None and len(ssa_dict) > 0 and ax_ssa is not None:
        # Plot SSA for each wavelength with both methods
        for wavelength in sorted(ssa_dict.keys()):
            color = wavelength_to_rgb(wavelength)

            # Method 1: scattering / extinction (dashed)
            if 'method1' in ssa_dict[wavelength]:
                ssa_method1 = ssa_dict[wavelength]['method1']
                if isinstance(ssa_method1, pd.Series) and ssa_method1.notna().sum() > 0:
                    ax_ssa.plot(ssa_method1.index, ssa_method1.values, '--', color=color,
                               linewidth=1.5, alpha=0.7, label=f'SSA₁ @ {wavelength} nm')

            # Method 2: scattering / (absorption + scattering) (solid)
            if 'method2' in ssa_dict[wavelength]:
                ssa_method2 = ssa_dict[wavelength]['method2']
                if isinstance(ssa_method2, pd.Series) and ssa_method2.notna().sum() > 0:
                    ax_ssa.plot(ssa_method2.index, ssa_method2.values, '-', color=color,
                               linewidth=1.5, alpha=0.7, label=f'SSA₂ @ {wavelength} nm')

        ax_ssa.set_ylabel('Single Scattering Albedo', fontsize=12)
        if ax_abs is None:
            ax_ssa.set_xlabel('Time (UTC)', fontsize=12)
        ax_ssa.set_ylim(0.7, 1.1)  # Tightened range around typical values
        ax_ssa.grid(True, alpha=0.3)
        ax_ssa.legend(loc='upper left', fontsize=8, ncol=2)

    # Bottom panel: Absorption
    if abs_data is not None and len(abs_data) > 0 and ax_abs is not None:
        # Plot absorption for each wavelength with wavelength-based colors
        for wavelength in sorted(abs_data.keys()):
            abs_series = abs_data[wavelength]
            color = wavelength_to_rgb(wavelength)

            if isinstance(abs_series, pd.Series) and abs_series.notna().sum() > 0:
                ax_abs.plot(abs_series.index, abs_series.values, '-', color=color,
                           linewidth=1.5, alpha=0.7, label=f'Abs{wavelength}')

        ax_abs.set_ylabel('Absorption (Mm⁻¹)', fontsize=12)
        ax_abs.set_xlabel('Time (UTC)', fontsize=12)
        ax_abs.set_yscale('log')
        ax_abs.set_ylim(0.001, 100)  # Log scale from 0.001 to 100 Mm⁻¹
        ax_abs.grid(True, alpha=0.3, which='both')
        ax_abs.legend(loc='upper left', fontsize=9)

    if ax_se is None and ax_ssa is None and ax_abs is None:
        ax1.set_xlabel('Time (UTC)', fontsize=12)

    # Format x-axis for time
    fig.autofmt_xdate()

    # Adjust layout to accommodate third axis
    if sc_data is not None and len(sc_data) > 0:
        plt.subplots_adjust(right=0.85)
    else:
        plt.tight_layout()

    # Save the plot
    if output_dir is None:
        output_dir = 'clock_alignment_plots'
    os.makedirs(output_dir, exist_ok=True)
    date_str = date.strftime('%Y%m%d')
    filename = os.path.join(output_dir, f'time_series{zoom_label}_{date_str}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved time series plot to {filename}")

    plt.close()  # Close the figure to free memory


def plot_time_series_zoom(date, las_sum, smps_sum, sc_data,
                          output_dir=None, scattering_efficiency_dict=None, ssa_dict=None,
                          abs_data=None):
    """
    Plot zoomed time series around a high-quality ±60 second window with complete data.

    Selection priority:
    1. Window with no NaNs in LAS, SMPS, and scattering, midpoint scattering > 50th percentile
    2. Window with no NaNs in LAS and scattering (drop SMPS requirement), midpoint scattering > 50th percentile
    3. Window with no NaNs in scattering (drop LAS requirement), midpoint scattering > 50th percentile

    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    las_sum : pd.Series
        Time series of LAS bin sum
    smps_sum : pd.Series
        Time series of SMPS bin sum
    sc_data : dict
        Dictionary mapping wavelength (nm) to scattering time series
    output_dir : str, optional
        Directory to save plot
    scattering_efficiency_dict : dict, optional
        Dictionary mapping wavelength (nm) to SE time series
    ssa_dict : dict, optional
        Dictionary mapping wavelength (nm) to {'method1': Series, 'method2': Series}
    abs_data : dict, optional
        Dictionary mapping wavelength (nm) to absorption time series
    """
    if sc_data is None or len(sc_data) == 0:
        print(f"  Warning: No valid scattering data for {date}, skipping zoom plot")
        return

    # Use 550 nm as reference wavelength for zoom window selection
    # (fall back to first available wavelength if 550 not present)
    ref_wavelength = 550 if 550 in sc_data else sorted(sc_data.keys())[0]
    sc_ref = sc_data[ref_wavelength]

    if sc_ref.notna().sum() == 0:
        print(f"  Warning: No valid Sc{ref_wavelength} data for {date}, skipping zoom plot")
        return

    # Calculate 50th percentile of reference scattering for the day
    sc_median = sc_ref.quantile(0.5)
    print(f"  Sc{ref_wavelength} 50th percentile: {sc_median:.2f} Mm⁻¹")

    # Window half-width
    window = pd.Timedelta(seconds=60)

    def check_window(center_time, require_las=True, require_smps=True):
        """Check if a window centered at center_time has complete data."""
        start_time = center_time - window
        end_time = center_time + window

        # Get window data
        sc_window = sc_ref[(sc_ref.index >= start_time) & (sc_ref.index <= end_time)]
        las_window = las_sum[(las_sum.index >= start_time) & (las_sum.index <= end_time)]
        smps_window = smps_sum[(smps_sum.index >= start_time) & (smps_sum.index <= end_time)]

        # Check if window has data and no NaNs
        sc_complete = len(sc_window) > 0 and sc_window.notna().all()

        if require_las:
            las_complete = len(las_window) > 0 and las_window.notna().all()
        else:
            las_complete = True

        if require_smps:
            smps_complete = len(smps_window) > 0 and smps_window.notna().all()
        else:
            smps_complete = True

        return sc_complete and las_complete and smps_complete

    def find_best_window(require_las=True, require_smps=True):
        """Find the window with highest scattering at midpoint meeting criteria."""
        valid_centers = []

        for center_time in sc_ref.index:
            # Skip if center point has NaN
            if pd.isna(sc_ref[center_time]):
                continue

            # Skip if below 50th percentile
            if sc_ref[center_time] < sc_median:
                continue

            # Check if window meets criteria
            if check_window(center_time, require_las, require_smps):
                valid_centers.append((center_time, sc_ref[center_time]))

        if valid_centers:
            # Return center with maximum scattering
            best_center = max(valid_centers, key=lambda x: x[1])
            return best_center[0], best_center[1]
        else:
            return None, None

    # Try each priority level
    print(f"  Searching for optimal zoom window...")

    # Priority 1: All three instruments
    center_time, center_sc = find_best_window(require_las=True, require_smps=True)
    if center_time is not None:
        print(f"  Found window with complete LAS, SMPS, and scattering data")
        print(f"  Center time: {center_time}, Sc{ref_wavelength}: {center_sc:.2f} Mm⁻¹")
    else:
        # Priority 2: LAS and scattering only
        center_time, center_sc = find_best_window(require_las=True, require_smps=False)
        if center_time is not None:
            print(f"  Found window with complete LAS and scattering data (SMPS has gaps)")
            print(f"  Center time: {center_time}, Sc{ref_wavelength}: {center_sc:.2f} Mm⁻¹")
        else:
            # Priority 3: Scattering only
            center_time, center_sc = find_best_window(require_las=False, require_smps=False)
            if center_time is not None:
                print(f"  Found window with complete scattering data (LAS and SMPS have gaps)")
                print(f"  Center time: {center_time}, Sc{ref_wavelength}: {center_sc:.2f} Mm⁻¹")
            else:
                print(f"  Warning: No suitable zoom window found above 50th percentile")
                return

    # Extract zoom window data
    start_time = center_time - window
    end_time = center_time + window

    las_zoom = las_sum[(las_sum.index >= start_time) & (las_sum.index <= end_time)]
    smps_zoom = smps_sum[(smps_sum.index >= start_time) & (smps_sum.index <= end_time)]

    # Slice scattering data to zoom window
    sc_zoom = {}
    for wavelength, sc_series in sc_data.items():
        sc_zoom[wavelength] = sc_series[(sc_series.index >= start_time) & (sc_series.index <= end_time)]

    # Slice scattering efficiency to zoom window
    se_zoom_dict = None
    if scattering_efficiency_dict is not None:
        se_zoom_dict = {}
        for wavelength, se_series in scattering_efficiency_dict.items():
            se_zoom_dict[wavelength] = se_series[(se_series.index >= start_time) &
                                                 (se_series.index <= end_time)]

    # Slice SSA to zoom window
    ssa_zoom_dict = None
    if ssa_dict is not None:
        ssa_zoom_dict = {}
        for wavelength, methods in ssa_dict.items():
            ssa_zoom_dict[wavelength] = {}
            for method_name, ssa_series in methods.items():
                ssa_zoom_dict[wavelength][method_name] = ssa_series[(ssa_series.index >= start_time) &
                                                                     (ssa_series.index <= end_time)]

    # Slice absorption to zoom window
    abs_zoom_dict = None
    if abs_data is not None:
        abs_zoom_dict = {}
        for wavelength, abs_series in abs_data.items():
            abs_zoom_dict[wavelength] = abs_series[(abs_series.index >= start_time) &
                                                    (abs_series.index <= end_time)]

    # Plot using the same function with zoom_label
    plot_time_series(date, las_zoom, smps_zoom, sc_data=sc_zoom,
                     output_dir=output_dir, zoom_label='_zoom',
                     scattering_efficiency_dict=se_zoom_dict,
                     ssa_dict=ssa_zoom_dict,
                     abs_data=abs_zoom_dict)


def plot_cross_correlation(date, shifts, correlations, dot_products, sc550_label='Sc550', n_valid_points=None, output_dir=None):
    """
    Plot cross-correlation and normalized dot product as a function of shift for a given date.
    Uses two y-axes: left for correlation coefficient, right for normalized dot product.

    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    shifts : np.array
        Array of shift values
    correlations : np.array
        Array of correlation coefficients
    dot_products : np.array
        Array of normalized dot products (divided by number of valid points)
    sc550_label : str, optional
        Label for Sc550 data (e.g., 'Sc550_total', 'Sc550_submicron')
    n_valid_points : np.array, optional
        Array of number of valid points used for each correlation
    output_dir : str, optional
        Directory to save plot
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left y-axis: Correlation coefficient
    color1 = 'tab:blue'
    ax1.set_ylabel('Pearson Correlation Coefficient', color=color1, fontsize=12)
    line1 = ax1.plot(shifts, correlations, 'o-', color=color1, linewidth=2,
                     markersize=4, label='Correlation')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Auto-scale correlation axis to show variation
    valid_corr_mask = ~np.isnan(correlations)
    if valid_corr_mask.any():
        corr_min = np.nanmin(correlations)
        corr_max = np.nanmax(correlations)
        corr_range = corr_max - corr_min
        if corr_range > 0:
            # Set y-axis limits with some padding to show variation
            ax1.set_ylim(corr_min - 0.05*corr_range, corr_max + 0.05*corr_range)
        else:
            # If range is zero (all identical), use a small range around the value
            center_val = corr_min
            ax1.set_ylim(center_val - 0.01, center_val + 0.01)
    
    # Right y-axis: Normalized dot product
    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel('Normalized Dot Product (Mean)', color=color2, fontsize=12)
    
    # Filter out NaN values for plotting
    valid_dot_mask = ~np.isnan(dot_products)
    if valid_dot_mask.any():
        line2 = ax1_twin.plot(shifts[valid_dot_mask], dot_products[valid_dot_mask], 
                             's-', color=color2, linewidth=2, markersize=4, 
                             alpha=0.7, label='Dot Product')
    else:
        line2 = []
    
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.grid(False)  # Don't show grid on second axis to avoid clutter
    
    # Format y-axis to show variation better if values are very close
    if valid_dot_mask.any():
        dot_min = np.nanmin(dot_products)
        dot_max = np.nanmax(dot_products)
        dot_range = dot_max - dot_min
        if dot_range > 0:
            # Set y-axis limits with some padding to show variation
            ax1_twin.set_ylim(dot_min - 0.05*dot_range, dot_max + 0.05*dot_range)
        else:
            # If range is zero (all identical), use a small range around the value
            # to make it clear they're identical
            center_val = dot_min
            ax1_twin.set_ylim(center_val * 0.9999, center_val * 1.0001)
            print(f"  WARNING: All dot products are identical ({center_val:.2e})")
    
    # Find maximum correlation
    valid_corr_mask = ~np.isnan(correlations)
    if valid_corr_mask.any():
        max_idx = np.nanargmax(correlations)
        max_shift = shifts[max_idx]
        max_corr = correlations[max_idx]
        max_dot = dot_products[max_idx]
        
        ax1.plot(max_shift, max_corr, 'r*', markersize=15, 
                label=f'Max correlation at shift={max_shift}')
        if not np.isnan(max_dot):
            ax1_twin.plot(max_shift, max_dot, 'r*', markersize=15, alpha=0.7)
        
        # Add text box with max correlation info
        if n_valid_points is not None:
            n_valid_at_max = n_valid_points[max_idx]
            textstr = (f'Max correlation: {max_corr:.4f}\n'
                      f'Optimal shift: {max_shift} steps\n'
                      f'Dot product at max: {max_dot:.2e}\n'
                      f'Valid points at max: {n_valid_at_max}')
        else:
            textstr = (f'Max correlation: {max_corr:.4f}\n'
                      f'Optimal shift: {max_shift} steps\n'
                      f'Dot product at max: {max_dot:.2e}')
    else:
        textstr = 'No valid correlations'
    
    ax1.set_title(f'LAS-{sc550_label} Cross-Correlation and Normalized Dot Product: {date}',
                 fontsize=14, fontweight='bold')
    
    # Combine legends (handle case where line2 might be empty)
    lines = list(line1)
    if len(line2) > 0:
        lines.extend(line2)
    labels = [l.get_label() for l in lines]
    if len(labels) > 0:
        ax1.legend(lines, labels, loc='upper left')
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Set x-axis label
    ax1.set_xlabel('Time Shift (steps)', fontsize=12)

    plt.tight_layout()
    
    # Always save the plot, never display it
    if output_dir is None:
        output_dir = 'clock_alignment_plots'
    os.makedirs(output_dir, exist_ok=True)
    date_str = date.strftime('%Y%m%d')
    filename = os.path.join(output_dir, f'cross_correlation_{date_str}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {filename}")
    
    plt.close()  # Close the figure to free memory


def main():
    """Main function to run clock alignment test."""
    
    
    print("=" * 70)
    print("Instrument Clock Alignment Test")
    print("=" * 70)
    
    # Load merged data
    print("\nLoading merged ICARTT data...")
#     df, meta = run_ascent_acp_merge(prefix_instr_name=False, output_directory=None)
    df, meta = run_ascent_acp_merge(mode_input='Load_Pickle', pickle_directory='/Users/wrespino/Downloads/ACTIVATE_TEST', pickle_filename='merged1sec_LAS-SMPS-Optical_2020Feb_V1')
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    
    # Identify bin columns
    print("\nIdentifying LAS and SMPS bin columns...")
    las_bins, smps_bins = identify_bin_columns(df)
    
    print(f"  Found {len(las_bins)} LAS bin columns")
    if las_bins:
        print(f"    Sample: {las_bins[:5]}")
    else:
        print("    WARNING: No LAS bin columns found!")
    
    print(f"  Found {len(smps_bins)} SMPS bin columns")
    if smps_bins:
        print(f"    Sample: {smps_bins[:5]}")
    else:
        print("    WARNING: No SMPS bin columns found!")
    
    if not las_bins:
        print("\nERROR: Cannot proceed without LAS bin columns.")
        return

    if not smps_bins:
        print("\nWARNING: No SMPS bin columns found. SMPS data will not be available for time series plots.")
    
    # Sum bins
    print("\nCalculating bin sums...")
    # Use min_count=1 to preserve NaNs (otherwise all-NaN rows sum to 0)
    las_sum = df[las_bins].sum(axis=1, min_count=1)
    smps_sum = df[smps_bins].sum(axis=1, min_count=1)

    print(f"  LAS sum: {las_sum.notna().sum()} valid points")
    print(f"  SMPS sum: {smps_sum.notna().sum()} valid points")

    # ICARTT directory for reading radii from headers
    icartt_directory = '/Users/wrespino/Downloads/ACTIVATE_TEST'

    # Find all scattering columns
    print("\nIdentifying aerosol scattering columns...")
    sc_columns = {}  # {wavelength_nm: column_name}

    # Find all optical aerosol scattering columns
    all_sc_cols = [col for col in df.columns if 'optical_aerosol' in col and
                   col.startswith('In-situ') and any(f'Sc{wl}' in col for wl in [450, 550, 700])]

    if all_sc_cols:
        print(f"  Found {len(all_sc_cols)} scattering-related column(s):")
        for col in all_sc_cols:
            print(f"    - {col}")

        # Extract wavelength and prefer non-ambient, total versions
        for wavelength in [450, 550, 700]:
            wl_cols = [col for col in all_sc_cols if f'Sc{wavelength}' in col]
            if wl_cols:
                # Prefer total over submicron, non-ambient over ambient
                total_cols = [col for col in wl_cols if 'total' in col.lower()]
                if total_cols:
                    non_amb = [col for col in total_cols if 'amb' not in col.lower()]
                    sc_columns[wavelength] = non_amb[0] if non_amb else total_cols[0]
                else:
                    submicron_cols = [col for col in wl_cols if 'submicron' in col.lower()]
                    if submicron_cols:
                        non_amb = [col for col in submicron_cols if 'amb' not in col.lower()]
                        sc_columns[wavelength] = non_amb[0] if non_amb else submicron_cols[0]
                    else:
                        sc_columns[wavelength] = wl_cols[0]

        print(f"\n  Selected scattering columns:")
        for wavelength, col in sorted(sc_columns.items()):
            n_valid = df[col].notna().sum()
            print(f"    Sc{wavelength}: {n_valid} valid points")
    else:
        print("  WARNING: No scattering columns found!")

    # Determine if we're using submicron or total scattering
    # (affects PSD truncation in SE calculation)
    if sc_columns and 550 in sc_columns:
        ref_label = 'submicron' if 'submicron' in sc_columns[550].lower() else 'total'
    else:
        ref_label = 'total'
    
    # Compute scattering efficiency
    print("\nComputing empirical scattering efficiency...")
    se_series_dict = {}  # {wavelength_nm: pd.Series}
    try:
        radii_dict = read_size_distribution_radii(df, icartt_directory)
        if radii_dict and 'LAS' in radii_dict:
            sigma_geo = compute_geometric_cross_section(
                df, las_bins, smps_bins, radii_dict, f'Sc_{ref_label}')
            print(f"  σ_geo range: {np.nanmin(sigma_geo):.2f} to {np.nanmax(sigma_geo):.2f} Mm⁻¹")

            if sc_columns:
                for wavelength, col_name in sorted(sc_columns.items()):
                    sc_data = df[col_name]
                    se_values = compute_scattering_efficiency(sc_data.values, sigma_geo)
                    se_series_dict[wavelength] = pd.Series(se_values, index=df.index)
                    valid_se = se_series_dict[wavelength].dropna()
                    print(f"  SE @ {wavelength} nm: range={valid_se.min():.2f} to {valid_se.max():.2f}, median={valid_se.median():.2f}")
            else:
                print("  Skipping SE: no scattering data available")
        else:
            print("  Skipping SE: no LAS radii found")
    except Exception as e:
        print(f"  Warning: Could not compute scattering efficiency: {e}")

    # Compute single scattering albedo (SSA) and collect absorption data
    print("\nComputing single scattering albedo (SSA)...")
    ssa_dict = {}
    abs_columns = {}  # {wavelength_nm: column_name} for absorption
    try:
        if sc_columns:
            ext_matched = find_matching_optical_columns(df, sc_columns, property_name='Ext')
            abs_matched = find_matching_optical_columns(df, sc_columns, property_name='Abs')

            # Print what was found
            print("  Matched extinction columns:")
            for sc_wl in sorted(ext_matched.keys()):
                if ext_matched[sc_wl]:
                    for ext_wl, ext_col in ext_matched[sc_wl].items():
                        print(f"    Sc{sc_wl} -> Ext{ext_wl}")
                else:
                    print(f"    Sc{sc_wl} -> no match found")

            print("  Matched absorption columns:")
            for sc_wl in sorted(abs_matched.keys()):
                if abs_matched[sc_wl]:
                    for abs_wl, abs_col in abs_matched[sc_wl].items():
                        print(f"    Sc{sc_wl} -> Abs{abs_wl}")
                        # Store absorption column for plotting (use native wavelength)
                        if abs_wl not in abs_columns:
                            abs_columns[abs_wl] = abs_col
                else:
                    print(f"    Sc{sc_wl} -> no match found")

            ssa_dict = compute_ssa(df, sc_columns, ext_matched, abs_matched)

            # Print SSA statistics
            for wavelength in sorted(ssa_dict.keys()):
                if 'method1' in ssa_dict[wavelength]:
                    valid_ssa1 = ssa_dict[wavelength]['method1'].dropna()
                    if len(valid_ssa1) > 0:
                        print(f"  SSA₁ @ {wavelength} nm: median={valid_ssa1.median():.3f}, range={valid_ssa1.min():.3f} to {valid_ssa1.max():.3f}")
                if 'method2' in ssa_dict[wavelength]:
                    valid_ssa2 = ssa_dict[wavelength]['method2'].dropna()
                    if len(valid_ssa2) > 0:
                        print(f"  SSA₂ @ {wavelength} nm: median={valid_ssa2.median():.3f}, range={valid_ssa2.min():.3f} to {valid_ssa2.max():.3f}")
        else:
            print("  Skipping SSA: no scattering data available")
    except Exception as e:
        print(f"  Warning: Could not compute SSA: {e}")
        import traceback
        traceback.print_exc()

    # Group by date
    print("\nGrouping data by date...")
    df['date'] = df.index.date
    dates = sorted(df['date'].unique())
    print(f"  Found {len(dates)} unique dates")
    
    # Calculate cross-correlation for each date
    print("\nCalculating cross-correlations for each date...")
    print("=" * 70)
    
    output_dir = 'clock_alignment_plots'
    
    for date in dates:
        print(f"\nProcessing date: {date}")
        date_mask = df['date'] == date
        df_date = df[date_mask]

        las_sum_date = las_sum[date_mask]
        smps_sum_date = smps_sum[date_mask]

        # Slice scattering data by date
        sc_date = {}
        for wavelength, col_name in sc_columns.items():
            sc_date[wavelength] = df[col_name][date_mask]

        # Slice absorption data by date
        abs_date = {}
        for wavelength, col_name in abs_columns.items():
            abs_date[wavelength] = df[col_name][date_mask]

        # Slice SE data by date
        se_date_dict = {}
        for wavelength in se_series_dict.keys():
            se_date_dict[wavelength] = se_series_dict[wavelength][date_mask]

        # Slice SSA data by date
        ssa_date_dict = {}
        for wavelength in ssa_dict.keys():
            ssa_date_dict[wavelength] = {}
            for method_name in ssa_dict[wavelength].keys():
                ssa_date_dict[wavelength][method_name] = ssa_dict[wavelength][method_name][date_mask]

        # Use 550 nm for cross-correlation (or first available wavelength)
        ref_wavelength = 550 if 550 in sc_date else sorted(sc_date.keys())[0] if sc_date else None

        # Check if we have enough data
        valid_las = las_sum_date.notna().sum()
        valid_smps = smps_sum_date.notna().sum()
        valid_sc_ref = sc_date[ref_wavelength].notna().sum() if ref_wavelength else 0
        print(f"  Valid LAS points: {valid_las}")
        print(f"  Valid SMPS points: {valid_smps}")
        print(f"  Valid Sc{ref_wavelength} points: {valid_sc_ref}")

        if valid_las < 90 or valid_sc_ref < 90:  # Need at least 90 points for ±30 shift with trimming
            print(f"  Skipping {date}: insufficient LAS or Sc{ref_wavelength} data (need at least 90 points)")
            continue

        # Calculate cross-correlation between LAS and reference scattering wavelength
        shifts, correlations, dot_products, n_valid_points = calculate_cross_correlation(
            las_sum_date, sc_date[ref_wavelength], max_shift=30
        )
        
        if len(shifts) == 0:
            print(f"  Skipping {date}: could not calculate correlations")
            continue
        
        # Find optimal shift
        valid_corr_mask = ~np.isnan(correlations)
        if valid_corr_mask.any():
            max_idx = np.nanargmax(correlations)
            optimal_shift = shifts[max_idx]
            max_corr = correlations[max_idx]
            max_dot = dot_products[max_idx]
            n_valid_at_max = n_valid_points[max_idx] if n_valid_points is not None else None
            
            # Debug: Check if dot products are varying
            valid_dot_mask = ~np.isnan(dot_products)
            if valid_dot_mask.any():
                dot_min = np.nanmin(dot_products)
                dot_max = np.nanmax(dot_products)
                dot_std = np.nanstd(dot_products)
                dot_range = dot_max - dot_min
                print(f"  Optimal shift: {optimal_shift} steps")
                print(f"  Maximum correlation: {max_corr:.4f}")
                print(f"  Dot product at optimal shift: {max_dot:.2e}")
                print(f"  Dot product range: {dot_min:.2e} to {dot_max:.2e}")
                print(f"  Dot product std: {dot_std:.2e}, range: {dot_range:.2e}")
                if dot_range > 0:
                    rel_variation = (dot_range / dot_max) * 100
                    print(f"  Relative variation: {rel_variation:.6f}%")
                else:
                    print(f"  WARNING: All dot products are identical!")
                if n_valid_at_max is not None:
                    print(f"  Valid points at optimal shift: {n_valid_at_max}")
            else:
                print(f"  Optimal shift: {optimal_shift} steps")
                print(f"  Maximum correlation: {max_corr:.4f}")
                print(f"  Warning: All dot products are NaN")
        else:
            print(f"  Warning: No valid correlations found for {date}")
            optimal_shift = None
            max_corr = np.nan
        
        # Plot full time series (with scattering if available)
        plot_time_series(date, las_sum_date, smps_sum_date, sc_data=sc_date,
                        output_dir=output_dir,
                        scattering_efficiency_dict=se_date_dict,
                        ssa_dict=ssa_date_dict,
                        abs_data=abs_date)

        # Plot zoomed time series around max scattering (if scattering available)
        if sc_date:
            plot_time_series_zoom(date, las_sum_date, smps_sum_date, sc_data=sc_date,
                                 output_dir=output_dir,
                                 scattering_efficiency_dict=se_date_dict,
                                 ssa_dict=ssa_date_dict,
                                 abs_data=abs_date)

        # Plot cross-correlation
        plot_cross_correlation(date, shifts, correlations, dot_products,
                              sc550_label=f'Sc{ref_wavelength}', n_valid_points=n_valid_points,
                              output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("Clock alignment test complete!")
    print(f"Plots saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

