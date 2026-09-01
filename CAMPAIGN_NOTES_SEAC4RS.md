# SEAC4RS (2013, NASA DC-8) campaign notes and decisions

First non-ACTIVATE application of the pipeline (V1 files, 2026-08-31).
Config: `configs/seac4rs_2013_full.json`; family map and bin tables in
`ASCENT_ACP/data/SEAC4RS_*`. Every decision below was made to mimic the
ACTIVATE instrument choices where possible and to keep the pipeline
general (no campaign-specific code paths — all differences are config or
data files, plus the generalizations listed at the end).

## Instrument choices

- **Optical**: LARGE-OPTICAL (TSI-3563 + PSAP), submicron scattering
  channels `Sc450/550/700_sub` (mirroring ACTIVATE 2021), bulk absorption.
  **Abs660_total is fill for the entire campaign** — the retrieval runs
  with 2 absorption channels (470, 532); the sca/abs channel counts were
  decoupled in ISARA to support this generally.
- **PSD**: LARGE SMPS-PSL (30 bins, 11-316 nm) + LAS-PSL (29 bins,
  0.1-6.31 um), stitched at 0.095 um (SMPS <= 95 nm, LAS > 95 nm) via the
  new `psd.stitch_dp_um` config — the instruments overlap on SEAC4RS,
  unlike ACTIVATE. **The LAS is PSL-calibrated (RI 1.59)** vs ACTIVATE's
  AmmSO4 calibration; the correlated diameter-scale nuisance already in
  the uncertainty model covers RI-driven sizing error in either
  convention, but expect a different `sizing_scale_shift` climatology.
- **Extra size instruments** (observations only, compacted onto their own
  diameter dims): UHSAS-AmmSO4, APS-PSL.
- **Cloud screening**: SPEC FCDP (unzipped from the archived .zip files;
  note the extracted files are named `Seac4rs-...` — the filename regex is
  case-tolerant). FCDP `conc` is #/L and `lwc` g/m^3; handled by the
  existing FilterConfig scale factors (1e-3, 1.0). No CDP on the DC-8
  (the CDP suffixes simply do not resolve; the filter skips them).
- **Ambient RH**: DC-8 housekeeping `Relative_Humidity_H2O` (no DLH RH
  product; DLH archives only H2O ppmv and is merged for completeness).
- **Inlet flag**: LARGE `Inlet` (0 = normal inlet). Several LARGE files
  carry an identically-named `Inlet` column, so the config uses a
  longer suffix (`optical_..._DC-8_Inlet`) to disambiguate — the
  suffix-resolution mechanism handles this without code changes.
- **Independent retrievals pulled into /observations** (not used by the
  ISARA fit): PI-Neph GRASP products (15-bin PSD, per-wavelength complex
  RI at 473/532/671 nm, SSA, sphericity; own inlet with ~2 um radius
  cut) under `pineph_retrievals`; DASH-SP size-resolved growth factors
  (Dp, RH, GF; no RI archived in R1) under `hygroscopic_growth`.
- **Additional context data merged**: LARGE CNC, AMS 60 s composition,
  SAGA bulk ions, HDSP2 black carbon, NOAA CRDS extinction (independent
  dry extinction at 405/532/662 — useful validation), CCN counter,
  DC-8 housekeeping (nav/met/state).

## Assumptions and caveats

- **Nephelometer sample RH is not archived** for SEAC4RS; the pipeline
  assumes a constant 30% (`channels.rh_sc_assumed_percent`, new config).
  The gamma synthesis of the 80%-RH kappa target inherits this
  assumption (~8% wet-target sensitivity between assuming 30% vs 40% at
  gamma ~ 0.5) — a documented caveat, not fixable from the archive.
- **Impactor**: assumed the same LARGE submicron convention as ACTIVATE
  (D50 = 1.0 um aerodynamic, gsd 1.15, rho 1.77 for the aero->optical
  mapping). Not stated in the SEAC4RS headers; worth confirming with the
  LARGE group alongside the ACTIVATE penetration-curve request.
- **No clock alignment** was performed (ACTIVATE's shift tables are
  campaign-specific). The align stage is skipped by pointing
  `paths.input_pkl` at the merged pickle directly (no `_timeShifted`
  suffix). Cross-instrument timing offsets are unquantified.
- The absorption-vs-scattering size-range inconsistency (bulk PSAP vs
  submicron PSD) carries over from ACTIVATE unchanged.
- 2013 flights are smoke/dust-influenced (BB plumes): expect more
  windows failing the SSA >= 0.7 and spectral-consistency gates, and the
  spherical/flat-RI assumptions are weaker in dust.

## Generalizations added for this campaign (all backward-compatible)

1. `varmap.resolve_bins`: bin columns may be diameter-named
   (`LAS_100nm`) as well as ordinal (`LAS_Bin01`).
2. `psd.stitch_dp_um`: hand-off diameter for overlapping SMPS/LAS.
3. `channels.rh_sc_suffix = ""` + `rh_sc_assumed_percent`: campaigns
   without an archived nephelometer RH.
4. ISARA: independent scattering/absorption channel counts.
5. netCDF export: `<TAG>_<NNN>nm[_<CAL>]` bin recognition with bin tables
   synthesized from the names (log-midpoint edges), a min-4-columns rule
   so per-wavelength scalars are not mistaken for bins, '/' sanitization
   in variable names (SAGA `*_ug/m3`), and leading-underscore stripping
   after title-prefix removal.
6. `scripts/plot_uncertainty_boxplots.py`: any-campaign bundle mode +
   extended extinction bins; `scripts/compare_independent_ri.py`:
   ISARA-vs-PI-Neph RI and ISARA-vs-DASH-SP kappa scatter comparisons.

## V1 results (2026-08-31)

All 21 flights merged (two merge-engine fixes were needed and committed to
icartt_read_and_merge: NaN mean-time-separation guard for 0-1-row files,
and sort/dedupe of non-monotonic source timestamps). Pipeline 243 s end to
end. 3118/9975 windows pass QC; **2073 CRI and 1992 kappa retrievals**.
Medians: dry RRI 1.526, IRI 0.007, SSA550 0.953, kappa 0.41 (SE-US
sulfate/smoke-consistent), min reduced chi^2 0.82.

**Headline diagnostic — `sizing_lnD_shift` = +0.181 (IQR +0.16..+0.20):**
the joint posterior infers the PSD diameters are ~18% undersized, tightly
clustered across the campaign. Sign and magnitude match the known
PSL-calibration bias (PSL RI 1.59 vs ambient ~1.5 assigns ambient
particles too-small optical diameters; Moore et al. 2021). This is 1.8x
the 0.10 lnD nuisance prior, so the marginalized CRI posterior is
sizing-dominated: **the retrieved dry RRI collapses to ~1.52-1.53 for
nearly every window (little independent information)** — visible in the
PI-Neph comparison below. The raw-signal LAS refit (todo, HIGH PRIORITY)
is the structural fix; a PSL->geometric pre-correction was deliberately
NOT applied (no fudge factors).

**Independent comparisons** (`processing_logs/SEAC4RS_V1_plots/`):
- ISARA vs PI-Neph GRASP RI at 532 nm (n=777 co-windowed): essentially
  uncorrelated (r ~ 0.05 RRI, ~0.0 IRI). Honest reading: ISARA RRI has
  no dynamic range here (above); PI-Neph spans 1.42-1.65 and IRI to 0.1
  (smoke/dust through a ~2 um-radius inlet at its own RH state, GRASP
  spheroid kernels). The two retrievals answer different questions on
  different air in this campaign; do not interpret as a validation
  failure of either without the sizing fix.
- ISARA vs DASH-SP kappa (n=1828): genuinely correlated (r = 0.40) with a
  systematic offset — ISARA ~+0.22 higher (med 0.41 vs ~0.19). Candidate
  causes, in likely order: the assumed 30% nephelometer RH in the wet
  target synthesis (higher true dry-RH would lower ISARA kappa), DASH
  being size-resolved at select Dp vs ISARA's bulk kappa, and residual
  sizing-bias leakage. Worth a sensitivity rerun with rh_sc_assumed = 40
  before interpreting the offset.
- Ambient-state climatology is physically sensible: SSA rising with
  loading (0.96 -> 0.99), ambient RRI falling with extinction
  (water uptake, 1.46 -> 1.39), AE 2.1-2.5 (fine smoke), IRI < 0.01.
