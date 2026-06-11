"""
Apply clock alignment shifts to merged ICARTT DataFrame.

For each UTC date × shift_group defined in variable_shift_table.csv:
  1. Compute cross-correlation between that group's alignment variable and
     the sum of LAS bins (LAS = reference timeline, no shift).
  2. Smooth the correlation-vs-lag curve with a Gaussian kernel and evaluate
     peak quality (n_valid, peak_r, peak_margin, monotonic_halfwidth_s).
  3. If all thresholds pass, shift every variable in that group for that
     date by the optimal lag; otherwise leave unshifted (SKIP).

Writes:
  - <input_pkl>_timeShifted.pkl
  - shift_diagnostics_<basename>.csv (one row per date × shift_group)
  - clock_alignment_plots/crosscorr_all_<date>.png (optional, default on)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# Skip thresholds — starting values; expect to iterate after first run.
MIN_N_VALID = 300
MIN_PEAK_R = 0.2
MIN_MONOTONIC_HALFWIDTH_S = 5

MAX_SHIFT_S = 30          # ± search window for optimal lag
EXCLUSION_RADIUS_S = 5    # exclusion zone for second-peak search
SMOOTH_SIGMA_S = 3.0      # Gaussian σ for lag-curve smoothing
CONCAVITY_WINDOW_S = 10   # ± window for concavity_fraction diagnostic


def identify_las_bin_columns(df):
    return [c for c in df.columns if 'LAS' in c and 'Bin' in c]


def compute_las_sum(df):
    """Row-wise sum of LAS bins; NaN for rows where >50% bins missing."""
    las_cols = identify_las_bin_columns(df)
    if not las_cols:
        raise RuntimeError("No LAS bin columns found in DataFrame")
    sub = df[las_cols]
    total = sub.sum(axis=1, skipna=True)
    total = total.where(sub.isna().mean(axis=1) < 0.5, np.nan)
    return total, las_cols


def cross_correlate(ref_series, target_series, max_shift=MAX_SHIFT_S):
    """
    Shift-lag correlation of target against ref (LAS). Positive lag means
    target samples taken `lag` steps LATER correlate best with ref → target
    clock runs ahead of ref; apply .shift(-lag) to align onto ref.

    Returns (shifts, correlations, n_valid_per_shift). Empty arrays if insufficient data.
    """
    idx = ref_series.index.intersection(target_series.index)
    r = ref_series.loc[idx].values.astype(float)
    t = target_series.loc[idx].values.astype(float)
    n = len(r)

    if n < max_shift * 3:
        return np.array([]), np.array([]), np.array([])

    ref_win = r[max_shift:n - max_shift]
    ref_nan = np.isnan(ref_win)
    shifts = np.arange(-max_shift, max_shift + 1)
    corrs = np.full(len(shifts), np.nan)
    n_valid = np.zeros(len(shifts), dtype=int)

    for i, s in enumerate(shifts):
        tgt_win = t[max_shift + s:n - max_shift + s]
        both = ~ref_nan & ~np.isnan(tgt_win)
        n_valid[i] = int(both.sum())
        if n_valid[i] < 2:
            continue
        rv, tv = ref_win[both], tgt_win[both]
        rstd, tstd = rv.std(), tv.std()
        if rstd == 0 or tstd == 0:
            continue
        corrs[i] = float(((rv - rv.mean()) * (tv - tv.mean())).mean() / (rstd * tstd))

    return shifts, corrs, n_valid


def smooth_correlation(corrs, sigma=SMOOTH_SIGMA_S):
    """Gaussian-smooth correlation curve, preserving NaNs as NaN."""
    y = corrs.copy()
    nan_mask = np.isnan(y)
    if nan_mask.all():
        return y
    # Interpolate NaNs for smoothing, then restore
    idx = np.arange(len(y))
    y[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], y[~nan_mask])
    y = gaussian_filter1d(y, sigma=sigma, mode='reflect')
    y[nan_mask] = np.nan
    return y


def monotonic_halfwidth(shifts, smoothed, peak_idx):
    """
    Walk outward from peak_idx; stop when smoothed R increases.
    Returns (left_hw_s, right_hw_s). Assumes 1-Hz grid (step = 1 s).
    """
    n = len(smoothed)
    # Right
    right = peak_idx
    while right < n - 1 and not np.isnan(smoothed[right + 1]) and smoothed[right + 1] <= smoothed[right]:
        right += 1
    # Left
    left = peak_idx
    while left > 0 and not np.isnan(smoothed[left - 1]) and smoothed[left - 1] <= smoothed[left]:
        left -= 1
    return int(shifts[peak_idx] - shifts[left]), int(shifts[right] - shifts[peak_idx])


def compute_diagnostics(shifts, corrs):
    """Peak metrics on smoothed correlation curve."""
    zero_idx = int(np.where(shifts == 0)[0][0])
    smoothed = smooth_correlation(corrs)
    out = {
        'optimal_shift_s': np.nan,
        'peak_r': np.nan,
        'peak_r_raw': np.nan,
        'zero_lag_r': np.nan,
        'peak_margin': np.nan,
        'second_peak_r': np.nan,
        'monotonic_halfwidth_s': np.nan,
        'left_halfwidth_s': np.nan,
        'right_halfwidth_s': np.nan,
        'concavity_fraction': np.nan,
        'smoothed': smoothed,
    }
    if np.all(np.isnan(smoothed)):
        return out

    max_idx = int(np.nanargmax(smoothed))
    peak_r = float(smoothed[max_idx])
    optimal = int(shifts[max_idx])

    # Second peak outside exclusion zone (on smoothed)
    mask_outside = np.abs(shifts - optimal) > EXCLUSION_RADIUS_S
    if mask_outside.any() and not np.all(np.isnan(smoothed[mask_outside])):
        second = float(np.nanmax(smoothed[mask_outside]))
    else:
        second = np.nan

    left_hw, right_hw = monotonic_halfwidth(shifts, smoothed, max_idx)

    # Concavity fraction on smoothed curve, ± CONCAVITY_WINDOW_S of peak
    d2 = np.gradient(np.gradient(smoothed))
    window_mask = np.abs(shifts - optimal) <= CONCAVITY_WINDOW_S
    d2_win = d2[window_mask]
    d2_win = d2_win[~np.isnan(d2_win)]
    concavity_fraction = float((d2_win < 0).mean()) if len(d2_win) else np.nan

    out.update({
        'optimal_shift_s': optimal,
        'peak_r': peak_r,
        'peak_r_raw': float(corrs[max_idx]) if not np.isnan(corrs[max_idx]) else np.nan,
        'zero_lag_r': float(smoothed[zero_idx]) if not np.isnan(smoothed[zero_idx]) else np.nan,
        'peak_margin': peak_r - second if not np.isnan(second) else np.nan,
        'second_peak_r': second,
        'monotonic_halfwidth_s': min(left_hw, right_hw),
        'left_halfwidth_s': left_hw,
        'right_halfwidth_s': right_hw,
        'concavity_fraction': concavity_fraction,
    })
    return out


def decide(diag, n_valid):
    reasons = []
    if n_valid < MIN_N_VALID:
        reasons.append(f"n_valid={n_valid}<{MIN_N_VALID}")
    if np.isnan(diag['peak_r']) or diag['peak_r'] < MIN_PEAK_R:
        reasons.append(f"peak_r<{MIN_PEAK_R}")
    hw = diag['monotonic_halfwidth_s']
    if np.isnan(hw) or hw < MIN_MONOTONIC_HALFWIDTH_S:
        reasons.append(f"monotonic_halfwidth_s<{MIN_MONOTONIC_HALFWIDTH_S}")
    if reasons:
        return 'SKIP', '; '.join(reasons)
    return 'SHIFT', ''


def load_shift_table(csv_path):
    """alignment_variable may be a '|'-separated list of fallbacks."""
    tbl = pd.read_csv(csv_path)
    yes = tbl[tbl['apply_time_shift'] == 'YES']
    groups = {}
    for g, sub in yes.groupby('shift_group'):
        raw = str(sub['alignment_variable'].iloc[0])
        groups[str(g)] = {
            'alignment_candidates': [s.strip() for s in raw.split('|') if s.strip()],
            'variables': sub['variable'].tolist(),
        }
    return groups


def pick_alignment_variable(candidates, columns):
    """Return first candidate present in columns, else None."""
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def plot_date(date_str, group_results, plot_path):
    """One figure per date; one subplot per group. Each subplot shows raw and smoothed R."""
    import matplotlib.pyplot as plt

    n = len(group_results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (g, res) in zip(axes, group_results.items()):
        shifts, corrs, smoothed = res['shifts'], res['corrs'], res['smoothed']
        diag = res['diag']
        decision = res['decision']
        reason = res['reason']

        if len(shifts) == 0:
            ax.text(0.5, 0.5, f"{g}: insufficient data", transform=ax.transAxes, ha='center')
            ax.set_xlim(-MAX_SHIFT_S, MAX_SHIFT_S)
            ax.set_ylabel('R')
            continue

        ax.plot(shifts, corrs, color='lightgray', lw=0.8, label='raw R')
        if smoothed is not None:
            ax.plot(shifts, smoothed, color='C0', lw=1.5, label=f'smoothed (σ={SMOOTH_SIGMA_S:.0f}s)')
        ax.axhline(0, color='k', lw=0.3)
        ax.axvline(0, color='k', lw=0.3, ls='--')

        color = 'C2' if decision == 'SHIFT' else 'C3'
        if not np.isnan(diag['optimal_shift_s']):
            ax.axvline(diag['optimal_shift_s'], color=color, lw=1.2, ls=':')
            ax.plot(diag['optimal_shift_s'], diag['peak_r'], 'o', color=color, ms=7)

        txt = (f"{g}  |  {decision}"
               f"\nshift={diag['optimal_shift_s']:+}s  "
               f"peak_r={diag['peak_r']:.3f}  "
               f"margin={diag['peak_margin']:.3f}  "
               f"halfwidth=({diag['left_halfwidth_s']},{diag['right_halfwidth_s']})s"
               f"\nconcavity_frac={diag['concavity_fraction']:.2f}"
               + (f"\nreason: {reason}" if reason else ""))
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, va='bottom', fontsize=7.5,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=color, alpha=0.85))
        ax.set_ylabel('R')
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('lag (s)  [positive = target leads LAS]')
    fig.suptitle(f'Cross-correlation vs LAS  —  {date_str}', y=0.995)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def apply_clock_alignment(input_pkl, shift_table_csv, output_pkl, diagnostics_csv, plot_dir):
    print(f"Loading: {input_pkl}")
    with open(input_pkl, 'rb') as f:
        df = pickle.load(f)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    groups = load_shift_table(shift_table_csv)
    print(f"Shift groups to process: {sorted(groups.keys())}")

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    df_out = df.copy()
    rows = []

    for date_ts in sorted(df.index.floor('D').unique()):
        date_str = date_ts.strftime('%Y-%m-%d')
        date_mask = df.index.floor('D') == date_ts
        date_slice = df.loc[date_mask]
        slice_idx = date_slice.index

        try:
            las_sum, _ = compute_las_sum(date_slice)
        except RuntimeError as e:
            print(f"{date_str}: {e}; skipping all groups this date")
            continue

        print(f"\n{date_str}  ({len(date_slice):,} rows)")
        group_results = {}

        for g, info in sorted(groups.items()):
            align_var = pick_alignment_variable(info['alignment_candidates'], date_slice.columns)
            row = {
                'date': date_str,
                'shift_group': g,
                'alignment_variable': align_var,
                'n_valid': 0,
                'optimal_shift_s': None,
                'peak_r': None,
                'peak_r_raw': None,
                'zero_lag_r': None,
                'peak_margin': None,
                'second_peak_r': None,
                'monotonic_halfwidth_s': None,
                'left_halfwidth_s': None,
                'right_halfwidth_s': None,
                'concavity_fraction': None,
                'decision': 'SKIP',
                'reason': '',
                'n_vars_shifted': 0,
            }

            if align_var is None:
                row['alignment_variable'] = '|'.join(info['alignment_candidates'])
                row['reason'] = 'no alignment_variable candidate present in DataFrame'
                rows.append(row)
                print(f"  {g:<12s} SKIP (alignment_variable missing)")
                continue

            shifts_arr, corrs, n_valid_arr = cross_correlate(las_sum, date_slice[align_var])
            if len(shifts_arr) == 0:
                row['reason'] = 'insufficient data for cross-correlation'
                rows.append(row)
                group_results[g] = {'shifts': shifts_arr, 'corrs': corrs, 'smoothed': None,
                                    'diag': {'optimal_shift_s': np.nan, 'peak_r': np.nan,
                                             'peak_margin': np.nan,
                                             'left_halfwidth_s': np.nan, 'right_halfwidth_s': np.nan,
                                             'concavity_fraction': np.nan},
                                    'decision': 'SKIP', 'reason': 'insufficient data'}
                print(f"  {g:<12s} SKIP (insufficient data)")
                continue

            diag = compute_diagnostics(shifts_arr, corrs)
            zero_idx = int(np.where(shifts_arr == 0)[0][0])
            n_valid_center = int(n_valid_arr[zero_idx])
            decision, reason = decide(diag, n_valid_center)

            row.update({k: v for k, v in diag.items() if k != 'smoothed'})
            row['n_valid'] = n_valid_center
            row['decision'] = decision
            row['reason'] = reason

            group_results[g] = {
                'shifts': shifts_arr,
                'corrs': corrs,
                'smoothed': diag['smoothed'],
                'diag': diag,
                'decision': decision,
                'reason': reason,
            }

            if decision == 'SHIFT':
                lag = int(diag['optimal_shift_s'])
                cols_to_shift = [c for c in info['variables'] if c in df_out.columns]
                if lag == 0 or not cols_to_shift:
                    row['n_vars_shifted'] = 0
                    print(f"  {g:<12s} SHIFT 0s (no-op; peak_r={diag['peak_r']:.3f})")
                else:
                    for col in cols_to_shift:
                        df_out.loc[slice_idx, col] = df_out.loc[slice_idx, col].shift(-lag)
                    row['n_vars_shifted'] = len(cols_to_shift)
                    print(f"  {g:<12s} SHIFT {lag:+d}s "
                          f"(peak_r={diag['peak_r']:.3f}, margin={diag['peak_margin']:.3f}, "
                          f"hw=({diag['left_halfwidth_s']},{diag['right_halfwidth_s']})s, "
                          f"n={len(cols_to_shift)} vars)")
            else:
                print(f"  {g:<12s} SKIP ({reason})")

            rows.append(row)

        if plot_dir and group_results:
            plot_path = os.path.join(plot_dir, f'crosscorr_all_{date_str}.png')
            plot_date(date_str, group_results, plot_path)

    diag_df = pd.DataFrame(rows)
    for col in ('peak_r', 'peak_r_raw', 'zero_lag_r', 'peak_margin',
                'second_peak_r', 'concavity_fraction'):
        diag_df[col] = pd.to_numeric(diag_df[col], errors='coerce').round(4)
    diag_df.to_csv(diagnostics_csv, index=False)
    print(f"\nDiagnostics written: {diagnostics_csv}")

    with open(output_pkl, 'wb') as f:
        pickle.dump(df_out, f)
    print(f"Shifted pickle written: {output_pkl}")

    print_summary(diag_df)
    return df_out, diag_df


def print_summary(diag_df):
    print("\n==== summary ====")
    print(f"total entries: {len(diag_df)}")
    print(f"decisions: {diag_df['decision'].value_counts().to_dict()}")
    applied = diag_df[diag_df['decision'] == 'SHIFT']
    if len(applied):
        print("\nshift magnitudes (s) per group:")
        for g, sub in applied.groupby('shift_group'):
            s = pd.to_numeric(sub['optimal_shift_s'], errors='coerce')
            print(f"  {g:<12s} n={len(sub):>3d}  min={s.min():+.0f}  "
                  f"median={s.median():+.0f}  max={s.max():+.0f}")
    skipped = diag_df[diag_df['decision'] == 'SKIP']
    if len(skipped):
        print("\nskip reasons:")
        print(skipped['reason'].value_counts().to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_pkl', help='merged pickle from ascent_acp merge')
    p.add_argument('--shift-table',
                   default='clock_alignment_results/variable_shift_table.csv')
    p.add_argument('--output-pkl', default=None,
                   help='default: <input>_timeShifted.pkl')
    p.add_argument('--diagnostics-csv', default=None,
                   help='default: clock_alignment_results/shift_diagnostics_<basename>.csv')
    p.add_argument('--plot-dir',
                   default='clock_alignment_plots',
                   help='directory for per-date cross-correlation plots; empty string to disable')
    args = p.parse_args()

    in_path = os.path.abspath(args.input_pkl)
    if args.output_pkl is None:
        root, ext = os.path.splitext(in_path)
        args.output_pkl = f"{root}_timeShifted{ext}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.diagnostics_csv is None:
        base = os.path.splitext(os.path.basename(in_path))[0]
        args.diagnostics_csv = os.path.join(
            script_dir, 'clock_alignment_results', f'shift_diagnostics_{base}.csv')

    shift_table = args.shift_table
    if not os.path.isabs(shift_table):
        shift_table = os.path.join(script_dir, shift_table)

    plot_dir = args.plot_dir
    if plot_dir and not os.path.isabs(plot_dir):
        plot_dir = os.path.join(script_dir, plot_dir)
    if plot_dir == '':
        plot_dir = None

    apply_clock_alignment(in_path, shift_table, args.output_pkl, args.diagnostics_csv, plot_dir)


if __name__ == '__main__':
    main()
