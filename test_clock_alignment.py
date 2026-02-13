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


def plot_time_series(date, las_sum, smps_sum, sc550=None, output_dir=None, zoom_label=''):
    """
    Plot time series of LAS sum, SMPS sum, and optionally Sc550 with multiple y-axes.

    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    las_sum : pd.Series
        Time series of LAS bin sum
    smps_sum : pd.Series
        Time series of SMPS bin sum
    sc550 : pd.Series, optional
        Time series of aerosol scattering at 550nm
    output_dir : str, optional
        Directory to save plot
    zoom_label : str, optional
        Additional label for zoomed plots (e.g., '_zoom')
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left y-axis: LAS sum
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (UTC)', fontsize=12)
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

    # Third y-axis: Sc550 (if provided)
    if sc550 is not None:
        ax3 = ax1.twinx()
        # Offset the third axis to the right
        ax3.spines['right'].set_position(('axes', 1.15))
        color3 = 'tab:green'
        ax3.set_ylabel('Sc550 Total (Mm⁻¹)', color=color3, fontsize=12)
        line3 = ax3.plot(sc550.index, sc550.values, '-', color=color3,
                         linewidth=1.5, alpha=0.7, label='Sc550 Total')
        ax3.tick_params(axis='y', labelcolor=color3)
        # Set minimum to ~5% below zero for visual spacing
        sc550_max = sc550.max() if sc550.notna().any() else 1
        ax3.set_ylim(bottom=-0.05 * sc550_max, top=sc550_max * 1.05)
        lines = lines + line3

    # Title
    title = f'LAS, SMPS, and Sc550 Time Series: {date}'
    if zoom_label:
        title += ' (Zoomed)'
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Format x-axis for time
    fig.autofmt_xdate()

    # Adjust layout to accommodate third axis
    if sc550 is not None:
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


def plot_time_series_zoom(date, las_sum, smps_sum, sc550, output_dir=None):
    """
    Plot zoomed time series around a high-quality ±60 second window with complete data.

    Selection priority:
    1. Window with no NaNs in LAS, SMPS, and Sc550, midpoint Sc550 > 50th percentile
    2. Window with no NaNs in LAS and Sc550 (drop SMPS requirement), midpoint Sc550 > 50th percentile
    3. Window with no NaNs in Sc550 (drop LAS requirement), midpoint Sc550 > 50th percentile

    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    las_sum : pd.Series
        Time series of LAS bin sum
    smps_sum : pd.Series
        Time series of SMPS bin sum
    sc550 : pd.Series
        Time series of aerosol scattering at 550nm
    output_dir : str, optional
        Directory to save plot
    """
    if sc550.notna().sum() == 0:
        print(f"  Warning: No valid Sc550 data for {date}, skipping zoom plot")
        return

    # Calculate 50th percentile of Sc550 for the day
    sc550_median = sc550.quantile(0.5)
    print(f"  Sc550 50th percentile: {sc550_median:.2f} Mm⁻¹")

    # Window half-width
    window = pd.Timedelta(seconds=60)

    def check_window(center_time, require_las=True, require_smps=True):
        """Check if a window centered at center_time has complete data."""
        start_time = center_time - window
        end_time = center_time + window

        # Get window data
        sc550_window = sc550[(sc550.index >= start_time) & (sc550.index <= end_time)]
        las_window = las_sum[(las_sum.index >= start_time) & (las_sum.index <= end_time)]
        smps_window = smps_sum[(smps_sum.index >= start_time) & (smps_sum.index <= end_time)]

        # Check if window has data and no NaNs
        sc550_complete = len(sc550_window) > 0 and sc550_window.notna().all()

        if require_las:
            las_complete = len(las_window) > 0 and las_window.notna().all()
        else:
            las_complete = True

        if require_smps:
            smps_complete = len(smps_window) > 0 and smps_window.notna().all()
        else:
            smps_complete = True

        return sc550_complete and las_complete and smps_complete

    def find_best_window(require_las=True, require_smps=True):
        """Find the window with highest Sc550 at midpoint meeting criteria."""
        valid_centers = []

        for center_time in sc550.index:
            # Skip if center point has NaN
            if pd.isna(sc550[center_time]):
                continue

            # Skip if below 50th percentile
            if sc550[center_time] < sc550_median:
                continue

            # Check if window meets criteria
            if check_window(center_time, require_las, require_smps):
                valid_centers.append((center_time, sc550[center_time]))

        if valid_centers:
            # Return center with maximum Sc550
            best_center = max(valid_centers, key=lambda x: x[1])
            return best_center[0], best_center[1]
        else:
            return None, None

    # Try each priority level
    print(f"  Searching for optimal zoom window...")

    # Priority 1: All three instruments
    center_time, center_sc550 = find_best_window(require_las=True, require_smps=True)
    if center_time is not None:
        print(f"  Found window with complete LAS, SMPS, and Sc550 data")
        print(f"  Center time: {center_time}, Sc550: {center_sc550:.2f} Mm⁻¹")
    else:
        # Priority 2: LAS and Sc550 only
        center_time, center_sc550 = find_best_window(require_las=True, require_smps=False)
        if center_time is not None:
            print(f"  Found window with complete LAS and Sc550 data (SMPS has gaps)")
            print(f"  Center time: {center_time}, Sc550: {center_sc550:.2f} Mm⁻¹")
        else:
            # Priority 3: Sc550 only
            center_time, center_sc550 = find_best_window(require_las=False, require_smps=False)
            if center_time is not None:
                print(f"  Found window with complete Sc550 data (LAS and SMPS have gaps)")
                print(f"  Center time: {center_time}, Sc550: {center_sc550:.2f} Mm⁻¹")
            else:
                print(f"  Warning: No suitable zoom window found above 50th percentile")
                return

    # Extract zoom window data
    start_time = center_time - window
    end_time = center_time + window

    las_zoom = las_sum[(las_sum.index >= start_time) & (las_sum.index <= end_time)]
    smps_zoom = smps_sum[(smps_sum.index >= start_time) & (smps_sum.index <= end_time)]
    sc550_zoom = sc550[(sc550.index >= start_time) & (sc550.index <= end_time)]

    # Plot using the same function with zoom_label
    plot_time_series(date, las_zoom, smps_zoom, sc550_zoom,
                     output_dir=output_dir, zoom_label='_zoom')


def plot_cross_correlation(date, shifts, correlations, dot_products, n_valid_points=None, output_dir=None):
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
    
    ax1.set_title(f'LAS-Sc550 Cross-Correlation and Normalized Dot Product: {date}',
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
    df, meta = run_ascent_acp_merge(mode_input='Load_Pickle', pickle_directory='/Users/wrespino/Downloads/ACTIVATE_TEST', pickle_filename='merged1sec_LAS-SMPS-Optical_2020-2-14_V1')
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

    # Find Sc550 column
    print("\nIdentifying Sc550 aerosol scattering column...")
    sc550_col = None
    pattern = 'Sc550_total'

    # Find all matching columns
    matching_cols = [col for col in df.columns if pattern in col or pattern.lower() in col.lower()]

    if matching_cols:
        print(f"  Found {len(matching_cols)} matching column(s): {matching_cols}")

        # If multiple matches, prefer the one that ends with the pattern
        exact_matches = [col for col in matching_cols if col.endswith(pattern) or col.lower().endswith(pattern.lower())]

        if exact_matches:
            sc550_col = exact_matches[0]
            print(f"  Selected column ending with '{pattern}': {sc550_col}")
        else:
            sc550_col = matching_cols[0]
            print(f"  Selected first match: {sc550_col}")

        sc550 = df[sc550_col]
        print(f"  Sc550: {sc550.notna().sum()} valid points")
    else:
        print("  WARNING: No Sc550 column found!")
        sc550 = None
    
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
        sc550_date = sc550[date_mask] if sc550 is not None else None

        # Check if we have enough data
        valid_las = las_sum_date.notna().sum()
        valid_smps = smps_sum_date.notna().sum()
        valid_sc550 = sc550_date.notna().sum() if sc550_date is not None else 0
        print(f"  Valid LAS points: {valid_las}")
        print(f"  Valid SMPS points: {valid_smps}")
        print(f"  Valid Sc550 points: {valid_sc550}")

        if valid_las < 90 or valid_sc550 < 90:  # Need at least 90 points for ±30 shift with trimming
            print(f"  Skipping {date}: insufficient LAS or Sc550 data (need at least 90 points)")
            continue

        # Calculate cross-correlation between LAS and Sc550
        shifts, correlations, dot_products, n_valid_points = calculate_cross_correlation(
            las_sum_date, sc550_date, max_shift=30
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
        
        # Plot full time series (with Sc550 if available)
        plot_time_series(date, las_sum_date, smps_sum_date, sc550_date, output_dir=output_dir)

        # Plot zoomed time series around max Sc550 (if Sc550 available)
        if sc550_date is not None:
            plot_time_series_zoom(date, las_sum_date, smps_sum_date, sc550_date, output_dir=output_dir)

        # Plot cross-correlation
        plot_cross_correlation(date, shifts, correlations, dot_products,
                              n_valid_points=n_valid_points, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("Clock alignment test complete!")
    print(f"Plots saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

