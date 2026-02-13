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


def calculate_cross_correlation(las_sum, smps_sum, max_shift=15):
    """
    Calculate cross-correlation between LAS sum and SMPS sum with time shifts.
    
    Parameters:
    -----------
    las_sum : pd.Series
        Sum of LAS bins
    smps_sum : pd.Series
        Sum of SMPS bins
    max_shift : int
        Maximum shift in time steps (default: 15)
    
    Returns:
    --------
    shifts : np.array
        Array of shift values (-max_shift to +max_shift)
    correlations : np.array
        Array of correlation coefficients for each shift
    dot_products : np.array
        Array of dot products (sum of element-wise products) for each shift
    n_valid_points : np.array
        Array of number of valid data points used for each correlation
    
    Notes:
    ------
    NaN handling:
    - For each shift, we align the two series and only use time points where
      BOTH LAS and SMPS have valid (non-NaN) data
    - np.corrcoef() automatically handles NaN by only using valid pairs
    - The correlation coefficient is NOT normalized by the number of valid points;
      it's the standard Pearson correlation coefficient, which is normalized by
      the standard deviations of the two series
    - However, fewer valid points means less statistical power, so we track
      n_valid_points to assess reliability
    """
    # Align indices (use intersection of valid data)
    common_idx = las_sum.index.intersection(smps_sum.index)
    las_aligned = las_sum.loc[common_idx]
    smps_aligned = smps_sum.loc[common_idx]
    
    if len(las_aligned) < max_shift * 2 + 1:
        print(f"  Warning: Not enough data points ({len(las_aligned)}) for max_shift={max_shift}")
        return np.array([]), np.array([]), np.array([])
    
    shifts = np.arange(-max_shift, max_shift + 1)
    correlations = np.zeros(len(shifts))
    dot_products = np.zeros(len(shifts))
    n_valid_points = np.zeros(len(shifts), dtype=int)
    
    for i, shift in enumerate(shifts):
        if shift == 0:
            # No shift - direct correlation
            # Find points where both have valid data
            both_valid_mask = las_aligned.notna() & smps_aligned.notna()
            las_vals = las_aligned[both_valid_mask].values
            smps_vals = smps_aligned[both_valid_mask].values
            
            n_valid_points[i] = len(las_vals)
            
            if len(las_vals) > 1:
                # Check for constant values (would cause NaN correlation)
                if np.std(las_vals) == 0 or np.std(smps_vals) == 0:
                    corr = np.nan
                else:
                    corr = np.corrcoef(las_vals, smps_vals)[0, 1]
                # Calculate dot product - zeros are fine, they contribute 0
                dot_prod = np.dot(las_vals, smps_vals)
            else:
                corr = np.nan
                dot_prod = np.nan
        elif shift > 0:
            # Shift SMPS forward (SMPS values appear later)
            # Align: LAS[0:-shift] with SMPS[shift:]
            if len(las_aligned) > shift:
                las_subset = las_aligned.iloc[:-shift]
                smps_subset = smps_aligned.iloc[shift:]
                
                # Find points where both have valid data after alignment
                both_valid_mask = las_subset.notna() & smps_subset.notna()
                las_vals = las_subset[both_valid_mask].values
                smps_vals = smps_subset[both_valid_mask].values
                
                n_valid_points[i] = len(las_vals)
                
                if len(las_vals) > 1:
                    # Check for constant values
                    if np.std(las_vals) == 0 or np.std(smps_vals) == 0:
                        corr = np.nan
                    else:
                        corr = np.corrcoef(las_vals, smps_vals)[0, 1]
                    dot_prod = np.dot(las_vals, smps_vals)
                else:
                    corr = np.nan
                    dot_prod = np.nan
            else:
                corr = np.nan
                dot_prod = np.nan
                n_valid_points[i] = 0
        else:  # shift < 0
            # Shift SMPS backward (SMPS values appear earlier)
            shift_abs = abs(shift)
            if len(las_aligned) > shift_abs:
                las_subset = las_aligned.iloc[shift_abs:]
                smps_subset = smps_aligned.iloc[:-shift_abs]
                
                # Find points where both have valid data after alignment
                both_valid_mask = las_subset.notna() & smps_subset.notna()
                las_vals = las_subset[both_valid_mask].values
                smps_vals = smps_subset[both_valid_mask].values
                
                n_valid_points[i] = len(las_vals)
                
                if len(las_vals) > 1:
                    # Check for constant values
                    if np.std(las_vals) == 0 or np.std(smps_vals) == 0:
                        corr = np.nan
                    else:
                        corr = np.corrcoef(las_vals, smps_vals)[0, 1]
                    dot_prod = np.dot(las_vals, smps_vals)
                else:
                    corr = np.nan
                    dot_prod = np.nan
            else:
                corr = np.nan
                dot_prod = np.nan
                n_valid_points[i] = 0
        
        correlations[i] = corr
        dot_products[i] = dot_prod
    
    return shifts, correlations, dot_products, n_valid_points


def plot_cross_correlation(date, shifts, correlations, dot_products, n_valid_points=None, output_dir=None):
    """
    Plot cross-correlation and dot product as a function of shift for a given date.
    Uses two y-axes: left for correlation coefficient, right for dot product.
    
    Parameters:
    -----------
    date : datetime.date
        Date being plotted
    shifts : np.array
        Array of shift values
    correlations : np.array
        Array of correlation coefficients
    dot_products : np.array
        Array of dot products
    n_valid_points : np.array, optional
        Array of number of valid points used for each correlation
    output_dir : str, optional
        Directory to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1 = axes[0]  # Correlation and dot product plot (2 y-axes)
    ax2 = axes[1]  # Valid points plot
    
    # Left y-axis: Correlation coefficient
    color1 = 'tab:blue'
    ax1.set_ylabel('Pearson Correlation Coefficient', color=color1, fontsize=12)
    line1 = ax1.plot(shifts, correlations, 'o-', color=color1, linewidth=2, 
                     markersize=4, label='Correlation')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Dot product
    ax1_twin = ax1.twinx()
    color2 = 'tab:orange'
    ax1_twin.set_ylabel('Dot Product', color=color2, fontsize=12)
    
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
    
    ax1.set_title(f'LAS-SMPS Cross-Correlation and Dot Product: {date}', 
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
    
    # Plot number of valid points
    if n_valid_points is not None:
        ax2.plot(shifts, n_valid_points, 'g-', linewidth=2, marker='o', markersize=3)
        ax2.set_ylabel('Number of Valid Data Points', fontsize=12)
        ax2.set_xlabel('Time Shift (steps)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Number of Valid Points Used in Correlation', fontsize=11)
    else:
        ax2.set_xlabel('Time Shift (steps)', fontsize=12)
        ax2.text(0.5, 0.5, 'Valid point counts not available', 
                transform=ax2.transAxes, ha='center', va='center')
    
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
    df, meta = run_ascent_acp_merge(prefix_instr_name=False, output_directory=None)
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
    
    if not las_bins or not smps_bins:
        print("\nERROR: Cannot proceed without both LAS and SMPS bin columns.")
        return
    
    # Sum bins
    print("\nCalculating bin sums...")
    las_sum = df[las_bins].sum(axis=1)
    smps_sum = df[smps_bins].sum(axis=1)
    
    print(f"  LAS sum: {las_sum.notna().sum()} valid points")
    print(f"  SMPS sum: {smps_sum.notna().sum()} valid points")
    
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
        
        # Check if we have enough data
        valid_las = las_sum_date.notna().sum()
        valid_smps = smps_sum_date.notna().sum()
        print(f"  Valid LAS points: {valid_las}")
        print(f"  Valid SMPS points: {valid_smps}")
        
        if valid_las < 30 or valid_smps < 30:  # Need at least 30 points for ±15 shift
            print(f"  Skipping {date}: insufficient data (need at least 30 points)")
            continue
        
        # Calculate cross-correlation
        shifts, correlations, dot_products, n_valid_points = calculate_cross_correlation(
            las_sum_date, smps_sum_date, max_shift=15
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
        
        # Plot
        plot_cross_correlation(date, shifts, correlations, dot_products, 
                              n_valid_points=n_valid_points, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("Clock alignment test complete!")
    print(f"Plots saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

