# ASCENT-ACP pipeline: theory basis (as implemented, V9, 2026-08-31)

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
`GRASP_KERNEL_PLAN.md`). Estimator `chi2-wmean` (posterior mean over the
CRI and kappa grids; `scripts/estimator_study.py`). Dry RRI grid
1.47-1.56 step 0.01; IRI 0-0.030.

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
  (lnD +/- 0.10, fully correlated — the dominant term), PSD
  concentration scale (+/-10%), and (submicron) impactor D50 +/-10%,
  gsd, rho +/- 0.2.

This makes the gate (min reduced chi^2 <= 1) and the posterior weights
MARGINAL over known model uncertainty: residuals along a known nuisance
direction are forgiven; spectrally inconsistent residuals of the same
size still fail. Consequences observed: 2021 successes 1442 -> 2173 vs
the conditional (diagonal) gate, and the 2021 median RRI relaxed from
1.552 (railed against the grid ceiling, compensating the sizing deficit)
to 1.516.

The kappa-stage sigma is ratio-based (the target is synthesized from the
same dry channel, so calibration cancels): 1% gamma-parameterization
term + the non-cancelling noise floor.

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

Documented v1 simplifications: kappa sigma is the grid-posterior std
only (PSD nuisances largely cancel in the wet/dry ratio); nuisances
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
