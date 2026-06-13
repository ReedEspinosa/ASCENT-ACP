# ASCENT-ACP QA / QC Criteria

Quality control in the ACTIVATE retrieval pipeline operates as **four sequential
gates**: clock-alignment → row-level (1 Hz) screening → window-level (60 s)
screening → retrieval acceptance. Each gate is described below with the
thresholds actually used in the full-run configs (`configs/activate_2020_full.json`,
`configs/activate_2021_full.json`) and source code.

The row- and window-level criteria follow Kacenelenbogen et al. (2022),
*ACP* 22, 3713, Appendix A1.

---

## 1. Clock-alignment QA (per flight-date, per instrument group)

`apply_clock_alignment.py` decides whether a per-group time shift is applied
before merging, based on the cross-correlation lag curve of each instrument
group against the LAS reference. A shift is applied only if the peak passes
**all three** thresholds; otherwise the data is kept at native timing (no shift).

| Criterion | Threshold | Meaning |
|---|---|---|
| `MIN_N_VALID` | ≥ 300 | overlapping valid 1 Hz points behind the correlation |
| `MIN_PEAK_R` | ≥ 0.2 | smoothed peak correlation coefficient |
| `MIN_MONOTONIC_HALFWIDTH_S` | ≥ 5 s | peak is a real, monotonic concave maximum, not noise |

Search parameters: ±30 s lag window (`MAX_SHIFT_S`), Gaussian smoothing
σ = 3 s (`SMOOTH_SIGMA_S`), second-peak exclusion ±5 s (`EXCLUSION_RADIUS_S`),
concavity diagnostic window ±10 s (`CONCAVITY_WINDOW_S`).

> **Note:** these thresholds are commented in code as *"starting values; expect
> to iterate after first run"* — they are the most likely to warrant tuning.

---

## 2. Row-level QC — 1 Hz screening

`ASCENT_ACP/filtering.py::row_qc`. A 1 Hz row is marked **valid** only if it
fails *none* of the following:

| Mask | Condition (True = reject) |
|---|---|
| `cloudy` | any available CDP/FCDP probe with **N > 1.0 cm⁻³** *or* **LWC > 0.001 g m⁻³**, expanded by a **±5 s pad** around any cloud hit |
| `inlet_bad` | `InletFlag_LARGE ≠ 0`, or missing (unknown inlet treated as bad) |
| `low_signal` | dry scattering at 450 nm **not > 10 Mm⁻¹** (NaN also fails) |
| `low_ssa` | SSA at 550 nm **≤ 0.7** (NaN passes — SSA requires Abs > 1 Mm⁻¹) |

RH standardization is performed **per row, before any averaging**, so
intra-window RH variability is handled exactly:
- dry scattering gamma-adjusted to the **40 %** reference RH (`dry_ref_rh`);
- a synthesized humidified channel gamma-adjusted to **80 %** RH (`wet_rh`) for
  the kappa retrieval.

Config keys: `filters.cloud_n_max_cm3`, `cloud_lwc_max_gm3`, `cloud_pad_s`,
`require_inlet_flag_zero`, `min_dry_sc450_Mm`, `min_ssa`, `ssa_filter_wvl`,
`dry_ref_rh`, `wet_rh`, `use_fcdp`.

---

## 3. Window-level QC — 60 s aggregation

`ASCENT_ACP/windows.py`. Valid 1 Hz rows are block-averaged into 60 s windows
(`window.window_s`). Each window receives a `window_qc_flag` bitmask; only
windows with **`window_qc_flag == 0`** proceed to retrieval.

| Bit | Flag | Condition |
|---|---|---|
| 1 | `FLAG_TOO_FEW_POINTS` | valid 1 Hz samples **< 20** (`min_valid_points`) |
| 2 | `FLAG_AE_UNSTABLE` | scattering Ångström exponent **relative** std **> 0.3** (`ae_max_relstd`, `ae_std_mode="relative"`), or AE mean/std missing |
| 4 | `FLAG_MISSING_OPTICS` | any window-mean dry scattering (450/550/700) or absorption (470/532/660) is NaN |

PSD bins additionally require **≥ 10 contributing samples**
(`min_valid_points_per_bin`); otherwise that bin's window mean is set to NaN.

---

## 4. Retrieval acceptance

`ASCENT_ACP/isara_bridge.py` + the ISARA library. Each surviving window
(`window_qc_flag == 0`) is sent to `ISARA.Retr_PSD`. Retrieval success is
gated, not assumed:

- A window with **< 2 valid PSD bins** raises and is recorded with
  `attempt_flag_CRI_unitless = 0` and `attempt_flag_kappa_unitless = 0`.
- CRI and kappa each carry their own `attempt_flag`; failed retrievals are
  flagged and retained, **not dropped**. This is why output counts decrease
  monotonically: QC-pass windows ≥ CRI retrievals ≥ kappa retrievals (kappa
  additionally needs the synthesized humidified channel).
- Closure/validation is anchored at **`val_wvl = 532 nm`** (SSA closure).

---

## Year variants (2020 vs 2021)

The QA thresholds above are **identical** between years; only the optical
channel set and PSD upper bound differ:

| | 2020 ("total") | 2021 ("submicron") |
|---|---|---|
| Scattering channels | `Sc*_total` | `Sc*_submicron` |
| PSD upper bound (`psd.psd_max_um`) | 5.0 µm | 1.0 µm |

---

## Net funnel (illustrative, 2021 full run)

17,371 windows → 463 QC-pass (`window_qc_flag == 0`) → 349 CRI retrievals →
292 kappa retrievals.
