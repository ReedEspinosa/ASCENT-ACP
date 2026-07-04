# ASCENT-ACP netCDF Output v2/v3 + Single-Pass Driver — Design Spec

Status: **implemented (v3)**. Supersedes the flat single-group v1 file formerly
produced by `netcdf_export.py`, and adds a campaign-agnostic ICARTT→netCDF
driver (`ASCENT_ACP.run`). All decisions in §2 are as-built; run
`python -m ASCENT_ACP.run --config configs/activate_2021_full.json`.

## v3 addendum (2026-07) — layout changes on top of the v2 design below

The group tree of §4 is unchanged, but v3 (`…_V3.nc`) restructures the axes
and several variable families:

1. **(flight × time) grid replaces the epoch time axes.** `flight` is the
   flight number in takeoff order within the campaign year (same-day `_L1/_L2`
   ICARTT files are separate flights); `time` is **seconds since UTC midnight
   of that flight's takeoff day** (extends past 86 400 if a flight crosses
   midnight). Root coords: `flight`, `time`, `flight_id`, `flight_date`,
   `midnight_epoch`, `takeoff_time`, `landing_time`. Absolute time of a sample
   = `midnight_epoch(flight) + time`.
   Flight envelopes come from the marker instrument's ICARTT files
   (`merge.flight_marker_instrument`, e.g. `ACTIVATE-SUMMARY`); without them,
   data-presence gap detection (`ASCENT_ACP/flights.py`). Merged rows outside
   every envelope are **dropped**: the merge engine nearest/linear-fills up to
   ~72 min beyond each instrument's real coverage, so those rows (e.g. the
   hours between two same-day flights) are synthetic, not measurements.
2. **No coarse time dimension.** All 60 s products (`/windowed`,
   `/windowed/retrievals`, `/windowed/raw`) are repeated at the native cadence
   on the same (flight, time) grid (`cell_methods` notes the 60 s window).
3. **Per-bin columns are compacted** into one variable per instrument
   (`dndlogd_smps/las/cdp/cas/fcdp[…_mean/_std]`) with a `size_<tag>` dimension
   and **bin-center radius** coordinate (`radius_<tag>`, µm) plus companion
   `diameter_<tag>`, `radius_lower/upper_<tag>`. Bin sizes are parsed from the
   ICARTT headers (per-variable `bin_center_X um` descriptions, FCDP
   `OTHER_COMMENTS` edge lists, SMPS/LAS nm bounds lines), falling back to the
   packaged `data/*_bins.csv`.
4. **Per-variable metadata from the source ICARTT headers**
   (`ASCENT_ACP/icartt_headers.py`): `units`, `long_name` (header description),
   `icartt_standard_name`, and `measurement_conditions` = `STP` | `ambient` |
   `not_applicable` | `unspecified`. Audited for ACTIVATE 2020+2021: **all
   quantitative aerosol variables are STP** (optical, SMPS/LAS, CCN, AMS,
   PILS); cloud probes (CDP/CAS/FCDP) are ambient as physically appropriate;
   mole fractions / ratios / flags / state-nav are `not_applicable`.
5. **Wind is vector-averaged** in `/windowed/raw` (u/v decomposition;
   meteorological FROM convention): `Wind_Speed_mean` and
   `Wind_Direction_mean` are the resultant-vector values,
   `Wind_Speed_scalar_mean` keeps the plain mean, `Wind_Direction_std` is the
   Yamartino (1984) estimator. Column pair set by
   `channels.wind_speed_suffix`/`wind_dir_suffix`.
6. The `composition` family is renamed **`aerosol_composition`** (avoids
   confusion with trace-gas composition).
7. Written by a streaming netCDF4 writer (one variable at a time, per-bin
   slices for 3-D) so memory stays flat despite the dense flight×seconds grid;
   `/clock_alignment` is unchanged (appended via xarray).

## 1. Goals

1. **Carry every variable from the merged pickle through to the netCDF**, at
   full time coverage (every timestep, regardless of QA), even for variables
   ISARA never touches.
2. Provide those variables at **both** their native cadence **and** as 60 s
   window averages, alongside the QA flags and ISARA retrievals.
3. The QA flag (Kacenelenbogen 2022 row + window checks; `QA_CRITERIA.md`
   §2–3) is **recorded** in the file and **gates ISARA** — a skipped retrieval
   becomes a bad-data flag + fill value, no window dropped.
4. Record the clock-alignment shift as provenance (a `(date × shift_group)`
   table + a per-variable `shift_group` attribute). This is **not** the same as
   the row/window QA.
5. **Run in one pass from ICARTT to netCDF for any campaign**, saving the
   intermediate pickles as checkpoints so a mid-run failure is recoverable and
   resumable. Nothing campaign-specific is hardcoded — it all comes from config.

## 2. Resolved decisions

| # | Decision | Choice |
|---|---|---|
| A | Raw-variable time resolution | **Native cadence** (detected at runtime, never assume 1 Hz) |
| B | Also emit 60 s averages of **all** raw vars | **Yes** → `/windowed/raw` group |
| C | Output filename | bump `…_V1.nc` → **`…_V2.nc`** |
| D | Fill for skipped retrievals | **NaN** (CF `_FillValue`) |

## 3. Grounding facts (verified against 2021 data)

- Merged pickle: **1 Hz, 235 columns, ~1.04 M rows/year**, tz-aware UTC index.
  Native cadence is **not guaranteed** 1 Hz — detect from the index.
- All 235 columns map cleanly to **16 ICARTT instrument titles** (0 unmatched),
  collapsing to **8 logical families** (§5).
- `meta` holds **per-instrument** fields (`Data_Info`, `Instrument_Info`,
  `PI_Info`, `Uncertainty`, `Revision`, `Stipulations`, `Institution_Info`),
  keyed by title. **No per-variable units** exist in `meta`.
- Clock shifts apply to **4 shift_groups** only (Optical, AMS, AMS-CVI, CCN per
  `variable_shift_table.csv`); other columns are unshifted/reference. Applied
  shift varies **per flight date** (`shift_diagnostics_*.csv`).
- Toolchain: **xarray 2026.2.0 (`xr.DataTree`)** + **netCDF4 1.7.2** → native
  hierarchical groups via `DataTree.to_netcdf()`.
- The current merge (`run_full_activate_merge.py`) is ACTIVATE-specific: a
  hardcoded instrument list, `*_HU25_YYYYMMDD_R#.ict` filename regex, source
  dir, and per-date master timeline. Generalizing this is the bulk of goal 5.

## 4. Group layout

Three top-level groups. The 60 s products share one `time` coordinate by living
under a common `/windowed` parent (DataTree children inherit parent coords).

```
/  (root: global attrs + provenance only)
│
├── observations/                 dim: time_obs (native cadence, ~1.04M)
│   ├── time_obs                   (coord, UTC)
│   ├── row_qc_flag                bitmask @ native cadence (the 1 Hz QA)
│   ├── optical/                   20 vars
│   ├── microphysical/             1  var
│   ├── aerosol_size_dist/         64 vars (SMPS 34 + LAS 30) + bin-diameter coords
│   ├── aerosol_composition/       38 vars (AMS 29 + PILS 9)
│   ├── ccn/                       2  vars
│   ├── cloud_probes/              90 vars (CDP 34 + CAS 34 + FCDP 22)
│   ├── trace_gas/                 7  vars
│   └── state_nav/                 13 vars
│
├── windowed/                     dim: time (window center, ~17k) + time_bnds  [shared]
│   ├── time, time_bnds            (coords, inherited by children below)
│   ├── window_qc_flag             ← QA flag that gates ISARA (bitmask, 0=good)
│   ├── n_valid, n_cloudy, n_inlet_bad, n_low_signal, n_low_ssa
│   │
│   ├── retrievals/                ISARA + QC-valid-only measured optical means
│   │   ├── wavelength_sca/abs, psd_bin, dp_mid/lower/upper  (coords/labels)
│   │   ├── scattering_dry(+std), absorption(+std), ssa, scattering_humidified, …
│   │   ├── dndlogdp               window-mean PSD
│   │   ├── refractive_index_real/imag, kappa
│   │   ├── <*_calculated_*>       MOPSMAP optics (dry/wet sca/abs/ext/SSA per wvl)
│   │   ├── attempt_flag_cri, attempt_flag_kappa
│   │   └── retrieval_qc_flag      ← NEW: why a retrieval is fill (§7)
│   │
│   └── raw/                       60 s mean+std of EVERY raw column (all-rows), by family
│       ├── optical/  …  state_nav/   (same 8 families as observations)
│       └── each var -> <name>_mean, <name>_std ; + n_points per window
│
└── clock_alignment/              dims: flight_date (~43), shift_group (4+)
    ├── flight_date, shift_group       (coords)
    ├── applied_shift_s                (flight_date × shift_group)
    ├── peak_r, n_valid, monotonic_halfwidth_s, decision_code
    ├── decision, reason               (string, flight_date × shift_group)
    └── attrs: MIN_N_VALID, MIN_PEAK_R, MIN_MONOTONIC_HALFWIDTH_S, MAX_SHIFT_S, …
```

**Two kinds of 60 s mean, on purpose:** `windowed/retrievals` optical means are
over **QC-valid 1 Hz rows only** (the ISARA inputs, unchanged from v1).
`windowed/raw` means are **unconditional** (all rows in the window) with an
`n_points` count, so the passthrough is a faithful straight average. Both are
labelled in their `long_name`/`cell_methods` so they're never confused.

## 5. Instrument → family map (235 cols, exhaustive)

| Family | ICARTT instrument title(s) | n |
|---|---|---|
| `optical` | In-situ_optical_aerosol_measurements | 20 |
| `microphysical` | In-situ_microphysical_aerosol_measurements | 1 |
| `aerosol_size_dist` | SMPS (34) + TSI_LAS (30) | 64 |
| `aerosol_composition` | Aerodyne_HR-ToF_AMS (29) + PILS_IC (9) | 38 |
| `ccn` | DMT_CCN_Counter | 2 |
| `cloud_probes` | DMT_CDP (34) + DMT_CAS (34) + SPEC_FCDP (22) | 90 |
| `trace_gas` | CO2 + CH4 + CO + UV_Ozone + DLH_H2O | 7 |
| `state_nav` | In-situ_state_and_aircraft_measurements | 13 |

Keyed by **longest-prefix match** of column name against `meta['Data_Info']`
titles. Stored as a **per-campaign** data file
`ASCENT_ACP/data/<campaign>_instrument_families.json` (ACTIVATE shipped;
falls back to one-family-per-title if a campaign has no map). Unknown titles →
`other/` group + logged warning.

## 6. Raw variable metadata

- **long_name**: column short-name with the instrument-title prefix stripped.
- **units**: best-effort parse from the instrument's `Data_Info` ICARTT header;
  omitted (not guessed) if unparseable.
- **shift_group**: from `variable_shift_table.csv`; vars not in the table get
  `"none"`. Points into `/clock_alignment`.
- dtype preserved (floats → float32 on write; integer flags kept integer).
- Each `observations/<family>` and `windowed/raw/<family>` subgroup gets
  group-level attrs from `meta` (`PI_Info`, `Institution_Info`, `Uncertainty`,
  `Revision`, `Stipulations`, `Instrument_Info`).

## 7. QA + retrieval bad-data semantics

- **Row QC (native)** → `observations/row_qc_flag` bitmask (1 cloudy, 2
  inlet_bad, 4 low_signal, 8 low_ssa; 0 = valid) with `flag_masks`/
  `flag_meanings`. Retained for every timestep.
- **Window QC (60 s)** → `windowed/window_qc_flag` (bitmask from `windows.py`;
  0 = good). **Gates ISARA.**
- **ISARA gating**: runs only where `window_qc_flag == 0`. Elsewhere
  `refractive_index_*`, `kappa`, `*_calculated_*` are `_FillValue` and
  `retrieval_qc_flag` says why:
  - `0` retrieved OK
  - `1` not attempted — failed window QA
  - `2` attempted but ISARA failed (e.g. < 2 valid PSD bins)
  `attempt_flag_cri`/`attempt_flag_kappa` retained for the CRI-vs-kappa split.
  No window is dropped.

## 8. Clock-alignment provenance group

From `shift_diagnostics_<basename>.csv`, pivoted to `(flight_date, shift_group)`:
`applied_shift_s` (= `optimal_shift_s` where `decision==APPLY`, else 0 with a
`decision_code`/`reason` to disambiguate), `peak_r`, `n_valid`,
`monotonic_halfwidth_s`, `decision_code` (0 SKIP / 1 APPLY), string
`decision`/`reason`. Group attrs record the acceptance thresholds. Per-variable
`shift_group` attr links each raw var to its row here for a given date.

## 9. Single-pass, campaign-agnostic driver

New top-level driver: **`python -m ASCENT_ACP.run --config <campaign>.json
[--years …] [--dates …] [--from-stage merge|align|retrieve|export] [--force]`**
that chains all four stages and **checkpoints to pickle between each**, so a
crash (or `--max-windows` debug run) resumes without redoing finished work.

Stages and checkpoints (all paths derived from config, per year):

| Stage | Input | Output checkpoint | Skip-if-exists key |
|---|---|---|---|
| 1 merge | ICARTT dir | `merged_<campaign>_<year>.pkl` + `_meta.pickle` | both exist |
| 2 align | merged pkl | `merged_<campaign>_<year>_timeShifted.pkl` + `shift_diagnostics_*.csv` | both exist |
| 3 retrieve | shifted pkl | `results_<campaign>_<year>.pkl` (windows+QA+ISARA bundle) | exists |
| 4 export | results + shifted + meta + diag | `ISARA_<campaign>_<year>_<variant>_<win>s_V2.nc` | exists |

- **Resume**: each stage checks for its checkpoint and skips unless `--force` or
  `--from-stage` names it or an earlier one. A stage that partially fails leaves
  no checkpoint, so it re-runs cleanly.
- **Campaign-agnostic merge**: the ACTIVATE-specific bits in
  `run_full_activate_merge.py` (instrument list, `*_HU25_YYYYMMDD_R#.ict` regex,
  per-date memory-safe timeline) move into a config-driven `merge` stage:
  - `MergeConfig`: `icartt_dir`, `instruments` (list), `filename_regex` (with
    named `date`/`instr` groups), `master_timeline_per_date` (bool),
    `merge_mode`, `n_workers`, `exclude_regexes`.
  - Memory-safe per-date merging stays the default (a year-long 1 s grid blows
    25 GB RAM); flights are assumed not to cross UTC midnight (logged if they do).
- **Generality**: channel suffixes (`varmap`), PSD bin CSVs, instrument-family
  map, and shift table are already (or become) config/data files keyed by
  campaign. The existing shell scripts (`run_*_activate*.sh`) become thin
  wrappers that just call `ASCENT_ACP.run` with a campaign config, or are
  retired.

The shell-script staging (`run_full_activate_overnight.sh`,
`run_2021_activate.sh`) is preserved as a fallback but the single driver becomes
the supported path.

### 9.1 Reuse boundary (what is and isn't touched)

The proven path **to the time-aligned pickle is not rewritten**:

- **Unchanged, called as-is:** `icartt_read_and_merge.icartt_merger()` (the
  merge engine, incl. the sub-second CDP fix) and `apply_clock_alignment.py`
  (cross-correlation, thresholds, shift application, diagnostics CSV,
  `_timeShifted.pkl`). The driver invokes alignment verbatim and consumes the
  identical pickle it already produces — output is byte-identical to today.
- **Generalized (wrapper only):** `run_full_activate_merge.py` → config-driven
  `merge.py`. Only its ACTIVATE-hardcoded surface (instrument list, filename
  regex, source dir, per-date timeline) becomes config; the per-date
  memory-safe loop and the `icartt_merger()` call inside are unchanged.
- **New:** everything after the time-aligned pickle (raw passthrough, dual-grid
  windowing, QA carry-through, grouped netCDF export) and the `run.py` driver.

## 10. Code changes

| File | Change |
|---|---|
| `ASCENT_ACP/run.py` | **New** — the single-pass driver/CLI (§9), stage checkpointing + resume. |
| `ASCENT_ACP/merge.py` | **New** — config-driven generalization of `run_full_activate_merge.py` (per-date memory-safe merge). |
| `ASCENT_ACP/netcdf_export.py` | Rewrite `to_dataset` → `build_datatree(...)` returning `xr.DataTree`; `write()` → `DataTree.to_netcdf` with per-node encoding. |
| `ASCENT_ACP/windowing_raw.py` | **New** — unconditional 60 s mean/std/count of all raw columns by family (`windowed/raw`). |
| `ASCENT_ACP/data/ACTIVATE_instrument_families.json` | **New** — title→family map + family order. |
| `ASCENT_ACP/config.py` | Add `MergeConfig`; `PathsConfig`: `icartt_dir`, `shift_diagnostics_csv`, `shift_table_csv`, checkpoint stems. New `OutputConfig`: compression, float32, `emit_windowed_raw` (default true). |
| `ASCENT_ACP/pipeline.py` | Keep `df`+`masks` past windowing; load shift diagnostics; feed exporter. Stays callable as the stage-3/4 core invoked by `run.py`. |
| `ASCENT_ACP/results.py` | Add `retrieval_qc_flag` derivation. |
| `configs/activate_2020_full.json`, `…2021_full.json` | Add merge block, shift CSV paths, output block. |
| `tests/` | `test_netcdf_groups.py` (group tree, native-cadence detection, fill/flag logic, family completeness = every column in exactly one family); `test_merge_config.py`; `test_resume.py` (stage skip-if-exists). |

`windows.py`, `filtering.py`, `isara_bridge.py`, `sizebins.py`, `plots.py`
unchanged. The `results.save_checkpoint` pickle stays, so `plots.py` and the
sanity scripts keep working.

## 11. Size / performance

`observations` ~1.04 M × 235 float32 ≈ 1 GB uncompressed; `windowed/raw` adds
~17 k × 470. With `zlib` level 4 + per-day time chunking, expect a few hundred
MB/year total. Peak RAM at export holds the source `df` + the DataTree; if tight,
write `observations` family-by-family in append mode. The per-date merge already
bounds merge-stage memory.

## 12. Remaining minor confirmations (non-blocking; sensible defaults chosen)

- `windowed/raw` means are **unconditional** (all rows) + `n_points`, distinct
  from the QC-valid-only retrieval means. (Alternative: also emit valid-only —
  doubles raw 60 s vars; default off.)
- Resume granularity is **per stage per year** (not per flight date). Finer
  per-date resume can be added later if a single date's failure proves common.
