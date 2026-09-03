# ASCENT-ACP pipeline: theory basis (as implemented, v5-layout era, 2026-09-03)

The algorithm as it stands, end to end, with pointers to the deeper
documents: `GRASP_KERNEL_PLAN.md` (forward-engine investigation),
`UNCERTAINTY_MODULE_PLAN.md` (uncertainty design + decisions),
`NETCDF_OUTPUT_SPEC.md` (file layout), `todo_reed.txt` (findings log,
open items), and ISARA_code/`THEORY.md` (the retrieval core).

## 1. Data flow

ICARTT files -> merged 1 Hz pickle -> clock alignment -> row QC ->
60 s windows -> ISARA retrieval (table engine, chi2-wmean, covariance
chi^2) -> uncertainty propagation -> grouped netCDF
(`ISARA_{campaign}_{year}_{variant}_{window}s_{version}.nc`).

## 2. Row QC (1 Hz; Kacenelenbogen et al. 2022 A1.1 heritage)

Bitmask per second: cloud (CDP/FCDP number or LWC over threshold,
+/-5 s pad; FCDP values scaled from native #/m^3, kg/m^3 — the V5 fix:
CDP-unit thresholds applied to FCDP had falsely flagged 41-52% of rows),
inlet flag nonzero, dry Sc450 <= 10 Mm^-1 (NaN fails), SSA <= 0.7.
Windows need >= 20 valid seconds; PSD bins need >= 10 valid samples.

## 3. PSD construction and the impactor model

Grid: SMPS (mobility, 0.003-0.094 um) + LAS (dry AmmSO4-optical
diameter, 0.094-3.55 um edges), truncated by a hard cap `psd_max_um`.

**Submicron variant (2021):** the nephelometer samples behind an
impactor with D50 = 1.0 um AERODYNAMIC (Schlosser et al. 2025). A D50
device transmits ~50% at its cut with a gradual rolloff, so any sharp
PSD cut misrepresents the sampled aerosol (both sharp 1.0 and 0.75 um
cuts under-predicted red scattering; V5/V6 finding). Since V6 the
retrieval input is instead WEIGHTED by a log-logistic penetration
P(D_a) = 1/(1+(D_a/D50)^s), s = ln(5.25)/ln(gsd), D_a = dpg*sqrt(rho),
with defaults gsd = 1.15 (assumed; LARGE's measured curve requested) and
rho = 1.77 g cm^-3 (AmmSO4, consistent with the LAS sizing). The
reported windowed PSD stays unweighted; the curve is exported as
`impactor_penetration`. Absorption is bulk (no impactor) in all years —
a documented inconsistency carried by the covariance below.

**Total variant (2020):** no impactor (D50 = 0 disables the weighting
bit-exactly); grid to the LAS top (3.16 um centers) vs a ~5 um inlet.

## 4. Retrieval configuration

Forward engine `table` (in-process Mie over the MOPSMAP single-sphere
efficiency extract; <= 0.21% vs exact Mie; ~100x faster than the
subprocess — both engines and the full ground-truth analysis in
`GRASP_KERNEL_PLAN.md`). Estimator `chi2-wmean` (posterior estimates
over the CRI and kappa grids; `scripts/estimator_study.py`). Dry RRI
grid 1.47-1.56 step 0.01; IRI 0-0.030 (decade points near zero,
2.5/5/7.5e-4, then 0.001 steps); kappa grid `kappa_min` (default -0.10)
to 1.4 step 0.001.

Posterior mechanics (2026-09-03): weights are exp(-n chi^2/2) times the
per-candidate GRID CELL WIDTH (quadrature measure), so grid density is
resolution, not prior — without this, the five quasi-zero IRI points
acted as a density spike that pulled the posterior-mean IRI low and left
absorption under-forecast in clean air (ACTIVATE 2020 low-IRI tercile
closed at 0.56). Point estimates: RRI and kappa = posterior mean; IRI =
posterior MEDIAN (boundary-robust on the one-sided IRI >= 0 axis).

The implicit priors this creates: RRI uniform on [rri_min, rri_max]
(mean 1.515, std 0.0287) — and since the per-window likelihood is flat
in RRI (nuisance-parallel; measured posterior std 0.0286), the reported
RRI IS the prior mean with a ~+/-0.002 data tilt. The grid choice is
therefore the RRI prior statement. IRI, by contrast, is well-posed: it
is constrained by the sca/abs RATIO, which is immune to PSD amplitude
error (LAS vs UHSAS IRI agree at r = 0.98 despite a 1.8x amplitude
disagreement between the sizers).

Kappa objective (`isara.kappa_objective`, default `ratio`, 2026-09-03):
the kappa stage fits the forward-modeled scattering ENHANCEMENT
(wet/dry at the retrieved CRI) to the synthesized-wet/measured-dry
target, not the absolute wet coefficient. Under the old absolute fit,
with RRI prior-bound, kappa was the only remaining amplitude degree of
freedom and absorbed the full dry-closure error (SEAC4RS: LAS closure
~0.61 inflated kappa to ~0.30 while UHSAS ~1.06 gave ~0.05 — the entire
sizer split; under `ratio` the two agree to |diff| ~0.001). Negative
kappa (floor `kappa_min = -0.10`): windows whose target enhancement is
< 1 (gamma noise around f ~ 1; thick smoke) retrieve an honest EFFECTIVE
negative kappa instead of hard-failing, and the posterior is not
truncated (biased high) at 0. Negative kappa is statistical, not water
loss: humidified-state products are computed only for kappa >= 0, and
candidates with kappa-Kohler gf^3 < 0.3 at the fit RH are excluded
(complex/divergent growth).

## 5. The chi^2: instrument sigmas + marginalized model uncertainty (V7-V9)

Per-window 1-sigma instrument models (port of
`ACMAP_Meloe/ISARA/aerosol_insitu_uncertainty_models.md`, "UM";
`ASCENT_ACP/uncertainty_models.py`): nephelometer three-term model
(f_rel by regime, white noise @30 s reference, zero-cycle floor), PSAP
model including the (0.016 * b_sp)^2 scattering-subtraction term with the
MEASURED scattering, LAS per-bin Poisson + relative terms. Rule: sigma
models always evaluate on measured window means, never retrieved values.

The CRI-stage misfit is the generalized chi^2 = r' S^-1 r / 6 with

    S = Sigma_meas + sum_k dy_k dy_k'

- Sigma_meas: white/floor diagonals plus the marginal relative terms
  split half-independent (diagonal) / half-common (rank-1) — the UM
  f_rel values are marginals with unquantified cross-channel
  correlation; the even split preserves each channel's marginal sigma.
- dy_k: secant coefficient signatures of the correlated MODEL nuisances,
  evaluated per window at a reference CRI: PSD diameter scale
  (lnD +/- `sizing_residual_lnd`, fully correlated — the dominant term),
  PSD concentration scale (+/- `n_scale_sigma`: sizer-specific since
  2026-09-03 — 0.20 in the LAS campaign configs after the SEAC4RS LAS
  dry-closure ~0.6 finding, 0.10 for UHSAS/default), and (submicron)
  impactor D50 +/-10%, gsd, rho +/- 0.2.

This makes the gate (min reduced chi^2 <= 1) and the posterior weights
MARGINAL over known model uncertainty: residuals along a known nuisance
direction are forgiven; spectrally inconsistent residuals of the same
size still fail. Consequences observed: 2021 successes 1442 -> 2173 vs
the conditional (diagonal) gate, and the 2021 median RRI relaxed from
1.552 (railed against the grid ceiling, compensating the sizing deficit)
to 1.516.

The kappa-stage sigma is ratio-based (the target is synthesized from the
same dry channel, so calibration cancels): 1% gamma-parameterization
term + the non-cancelling noise floor. Since the `ratio` kappa objective
(sec. 4) the RESIDUAL is ratio-based too, so the sigma budget and the
fit are finally the same framing — previously the sigma assumed
cancellation the absolute fit did not deliver.

Interpretation (2026-09-03): this marginalized chi^2 is the PROFILE
LIKELIHOOD of a joint retrieval over (CRI, N-scale, lnD shift, impactor
parameters) with Gaussian priors — the PSD amplitude is retrieved
implicitly; we simply report the measured-PSD forward state and carry
the nuisance MAP separately (sec. 6). A large calc-vs-measured
scattering gap is therefore not an unexplained residual: it is the
retrieved sizer amplitude error displayed in scattering space
(`scattering_dry_closure_ratio`). The SSA closure offsets in the v4
files are exactly the sca-vs-abs closure DIFFERENTIAL (common-mode
amplitude cancels in SSA); letting CRI or kappa absorb that amplitude
error is the leak class this design exists to prevent.

## 6. Uncertainty propagation (V9; `ASCENT_ACP/uncertainty_propagation.py`)

Per retrieved window, with NO retrieval reruns (all linear algebra on
quantities the grid search already produces, plus ~milliseconds of
forward evaluations):

1. **Posterior (noise) term:** the chi2-wmean posterior covariance of
   (RRI, IRI) — computed with the same S as the retrieval — mapped
   through finite-difference Jacobians of every product (sca/abs/ext/SSA
   per state and wavelength, AE, humidified CRI, gf) w.r.t. (RRI, IRI),
   plus the kappa-posterior term through d/d kappa.
2. **Nuisance term, joint-posterior accounting:** the nuisance
   amplitudes theta (1-sigma units, coefficient signatures D from S) are
   CONDITIONED on the fit residual at the reported CRI:

       theta_hat = D (S + Cov_yy)^-1 r,   Sigma_theta = I - D (S + Cov_yy)^-1 D'

   Product sigmas use the posterior second moment
   E[theta theta'] = Sigma_theta + theta_hat theta_hat' contracted with
   each nuisance's DIRECT product sensitivity (signed secant at fixed
   CRI). Products are reported at theta = 0, so the known-but-uncorrected
   shift theta_hat contributes to the uncertainty rather than being
   corrected (per project decision — no fudge factors). Directions the
   measurements observe collapse toward measurement precision (e.g.
   calculated scattering at measured wavelengths); unobserved directions
   stay at prior width (e.g. most of the AE tilt).
3. **Diagnostics:** `sizing_scale_shift` = the posterior-mean lnD shift
   per window (2021: median +0.045, IQR +0.03..+0.06 — a systematic
   LAS-calibration offset in the submicron configuration; 2020: median
   +0.006, centered on zero); `uncertainty_flag` bitmask marking windows
   where the linearization is stressed (RRI/IRI near a grid edge,
   min chi^2 near the gate, large ambient growth).
4. **MAP-fit PSD outputs (2026-09-03):** the same theta_hat is exported
   as point diagnostics under `/windowed/retrievals`:
   `psd_scale_factor_fit` (1 + theta*sigma_N) and
   `scattering/absorption_dry_fit` = y0 + D' theta_hat, the first-order
   forward state at the retrieved CRI and the MAP-adjusted PSD. This
   makes the implicit joint retrieval explicit: the residual of
   `*_dry_fit` vs measured should sit at instrument level when the
   nuisance model is adequate (SEAC4RS LAS 2-flight: raw closure 0.60 ->
   fit 0.98, decomposed as scale 1.14 x lnD +0.11). The archived
   dndlogdp is never modified.

Documented v1 simplifications: kappa sigma is the grid-posterior std
only (PSD nuisances cancel in the wet/dry ratio — an assumption under
the old absolute kappa fit, a construction under the `ratio` objective);
nuisances
independent; theta conditioning uses the posterior (not prior) CRI
spread; PSD gap-filling unmodeled. One-off OAT re-retrieval validation
of the linearization is on the todo list (not release-gating).

Output: `/windowed_uncertainty/{observations,retrievals}` netCDF groups
mirroring `/windowed` names and dims (1-sigma values; see
`NETCDF_OUTPUT_SPEC.md`).

## 7. Key empirical findings (full log in todo_reed.txt)

- FCDP units bug (V5): cloud filter applied CDP-unit thresholds to
  FCDP's #/m^3 / kg/m^3 data; fix multiplied valid windows ~6x.
- Forward-engine ground truth: MOPSMAP dataset point-values and
  (de-smoothed) GRASP kernels are both ~exact for broad PSDs; the table
  engine ships; GRASP kernels retained for shape studies via
  GEOSmie_TOA `kerneloptics.py` (`GRASP_KERNEL_PLAN.md`).
- Estimator study: chi2-weighted mean beats mean-of-accepted (fragile
  binary gate), plain mean of chi^2<1 (grid-shape dependent) and
  min-chi^2 (noisiest).
- 2021 spectral misfit: both sharp submicron cuts under-predict red
  scattering; no impactor steepness fixes it; a uniform +~4% diameter
  scale closes it — a correlated LAS sizing offset (RI different from
  the AmmSO4 calibration), degenerate with RRI. Now marginalized in the
  chi^2, quantified per window by `sizing_scale_shift`, and the target
  of the HIGH-PRIORITY raw-signal LAS refit.
- 2020 failed windows: dominated (91% of chi^2) by absorption spectra
  with substantial blue absorption but Abs660 <= 0 (54% of failures) —
  unphysical, likely a PSAP correction artifact; reconsideration queued.
- sigma(AE) is dominated by the sizing nuisance (~90%), rising with
  AE-vs-size sensitivity; after V9 conditioning it is flat-to-decreasing
  with ambient extinction.
- Kappa amplitude leak (2026-09-03, SEAC4RS): with RRI prior-bound,
  the ABSOLUTE kappa fit made kappa the only amplitude knob — the
  LAS/UHSAS kappa split (0.30 vs 0.05, DASH-SP ~0.18 between) was
  entirely the dry-closure ratio (0.61 vs 1.06) leaking into kappa
  (kappa vs ln-closure r ~ -0.4 within each run). The `ratio` objective
  collapses the split to |diff| ~0.001, r = 1.000. Remaining
  ISARA-vs-DASH offset = LARGE-gamma-vs-DASH-GF disagreement + size
  selection, not retrieval error.
- IRI zero-boundary bias (2026-09-03): absorption closure was
  IRI-tercile-dependent (ACTIVATE 2020: 0.56 low tercile vs 0.87 high)
  — the quasi-zero IRI grid cluster + one-sided posterior mean pulled
  IRI low in clean air. Fixed by quadrature weights + posterior-median
  IRI; SEAC4RS LAS low tercile 0.51 -> 0.64, stratification gone.
- RRI prior dominance measured: per-window posterior std 0.0286 vs
  uniform-grid prior std 0.0287 (<1% narrowing); LAS and UHSAS retrieve
  identical RRI despite opposite amplitude errors. The reported RRI is
  the grid-center prior with a ~+/-0.002 tilt; choose [rri_min,rri_max]
  accordingly.
- LAS-vs-UHSAS amplitude: in the 110-700 nm overlap band LAS carries
  ~69% of UHSAS's surface (median; number ~84%) in SEAC4RS — the basis
  for the sizer-specific `n_scale_sigma` (LAS 0.20 vs UHSAS 0.10).
