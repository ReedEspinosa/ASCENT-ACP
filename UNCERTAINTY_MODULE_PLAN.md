# Plan: measurement-uncertainty module (chi^2 sigmas + propagated product uncertainties)

Drafted 2026-08-31 (Reed + Claude). Status: **plan only — review before
implementation.** Target release: V7.

Source error models: `/Users/wrespino/Synced/ACMAP_Meloe/ISARA/aerosol_insitu_uncertainty_models.md`
(referred to below as UM; all 1-sigma; three-term form
`sigma^2 = (f_rel*y)^2 + a^2*(t_ref/t) + b^2`). Its Section 5 reference
implementation is ported nearly verbatim as the starting point.

## Goals

1. Replace the ad-hoc chi^2 sigmas (20% sca, 1 Mm^-1 abs, 1% wet sca) with
   the UM instrument models, per window and per channel.
2. Propagate measurement AND structural uncertainties (size cuts, sizing
   scale, densities) into every retrieved product.
3. Publish the result as a new top-level netCDF group `/windowed_uncertainty`
   mirroring `/windowed` where appropriate — same variable names and dims,
   values are 1-sigma uncertainties.

Ground rule (Reed): wherever a sigma model needs an optical value (SSA, b_sp
for the PSAP term, ...), use the MEASURED window mean, never the
retrieved/calculated one.

## Part 1 — realistic sigmas in chi^2

New module `ASCENT_ACP/uncertainty_models.py` (port of UM Sect. 5):

- `sigma_scattering(b_sp, t=60, wavelength, regime)` — three-term neph model;
  `a_lambda` = {450: 0.15, 550: 0.10, 700: 0.30} Mm^-1 @30 s, zero-cycle
  floor a/3.16. Regime from config: `"pm1"` (f_rel 0.08) for the submicron
  variant; `"pm10"` 0.09 default for total, with the UM caveat that
  coarse-dominated legs deserve 0.15-0.25 (open question 6).
- `sigma_absorption(b_ap, b_sp, t=60)` — PSAP model INCLUDING the
  `(0.016*b_sp)^2` scattering-subtraction term with the MEASURED b_sp, the
  frozen-below-60 s white term (window = 60 s, so t_eff = 60), and the
  0.015 Mm^-1 floor. Sanity gate for tests: reproduce the UM SSA table
  (12% at omega 0.5, 33% at 0.95, 160% at 0.99).
- `sigma_number(N, t, Q, density_ratio, edge_bins)` — LAS/SMPS per-bin
  diagonal (Poisson + f_rel 0.10, 0.30 edge bins). Window mean over
  n_valid rows divides the Poisson/white part by n_valid, NOT f_rel.
  density_ratio from window-mean pressure altitude when available.
- Correlated nuisances (NOT in any diagonal): `OPC_DLND = 0.10` diameter
  scale; APS priors ready for the APS integration.

ISARA change (small): `Retr_CRI`/`Retr_kappa` accept optional per-channel
`sigma` arrays; `estimator='chi2-wmean'` uses them; `None` keeps the legacy
tolerances (regression path). Bridge computes the arrays from measured
window means. Config: `isara.chi2_sigma = "legacy" | "instrument"`.

DECIDED (Reed, 2026-08-31): instrument sigmas feed the SAME chi2-wmean
machinery for gating AND weighting — no split treatment.

V8 REFINEMENT (Reed, 2026-08-31): the diagonal-sigma chi^2 (V7) gates
conditionally on a perfect model, rejecting windows for residuals our own
budget expects. V8 marginalizes instead: chi^2 = r' S^-1 r / n_ch with
S = Sigma_meas + sum_k dy_k dy_k' — the model nuisances (PSD lnD scale,
concentration scale, impactor D50/gsd/rho) enter as rank-1 outer products
of their secant coefficient shifts, so residual patterns along known
nuisance directions are forgiven while inconsistent spectral shapes still
fail. Measurement f_rel terms are split half-independent (diagonal) /
half-common (rank-1), preserving each channel's marginal sigma (the UM
values are marginals with unquantified cross-channel correlation).
Bookkeeping: in this mode the propagation drops the gain terms for
everything inside S (posterior width already carries them) and keeps only
the nuisances' DIRECT product effects at fixed CRI.
Config: isara.chi2_sigma = "instrument-cov" (default) | "instrument" (V7)
| "legacy".

V9 REFINEMENT (2026-08-31): joint-posterior nuisance accounting. The V8
direct terms used PRIOR nuisance widths, overstating sigma for products
the data directly pin (calculated coefficients at measured wavelengths).
V9 conditions the nuisance amplitudes theta on the residual at the
reported CRI: theta_hat = D (S + Cov_yy)^-1 r, Sigma_theta = I - D(...)D';
product sigmas use E[theta theta'] = Sigma_theta + theta_hat theta_hat'
so the known-but-uncorrected shift is counted as uncertainty (products
stay reported at theta = 0). Observable directions collapse toward
measurement precision; unobserved ones stay at prior width; AE barely
moves (its slope direction is weakly observed). New diagnostic exported:
sizing_scale_shift (posterior lnD shift; ~+0.04 in 2021 — the sizing
offset the raw-signal refit will address). Wet-channel sigma corrected to
the ratio model (1% gamma + noise floor; calibration cancels against the
dry channel the target is synthesized from). Expected
population effects (quantify in the U2 A/B anyway): tighter than 20% at
high signal, looser than the effective floor at low signal, which should
largely cure the 3.5x min-chi^2 inflation at Sc550 < 20 Mm^-1 seen in the
V5/V6 diagnostics.

Wet channel v1: apply the neph 550 model to the synthesized wet Sc550.
Honest caveat recorded in metadata: the target is dry Sc550 gamma-adjusted,
so its true sigma also carries the gamma-fit uncertainty; a gamma term is a
v2 enhancement (open question 1).

## Part 2 — propagation into retrieved products

Error budget = two mechanically different pieces:

**(a) Measurement noise -> grid posterior.** With realistic sigmas, the
chi2-wmean posterior widths ARE the noise-driven uncertainty: RRI/IRI
weighted stds, kappa_std (already computed), extended to every derived
product by taking the posterior-weighted spread of the per-candidate
forward-modeled quantities (SSA, sca/abs/ext per lambda per state) — cheap,
the candidate coefficients already exist inside the search.

**(b) Correlated/structural nuisances -> ensemble-gain propagation (default)
with OAT rerun validation.** The grid search already holds every candidate's
forward coefficients y_i and posterior weight w_i; that cloud yields the
retrieval's local linear response to any coefficient perturbation as a
Kalman-style gain computed per window from quantities already in memory:

    G = Cov_w(x, y) [Cov_w(y, y) + Sigma]^-1 ,   x = (RRI, IRI)

Each nuisance then costs ONE forward evaluation at the reported CRI
(~0.4 ms, table engine) to get its coefficient shift dy_k, and the product
shift is dx_k ~= -G dy_k; derived products (SSA, coefficients per
state/lambda) propagate through the same weighted candidate-cloud
regressions, and kappa through its own posterior. Total cost: milliseconds
per window — no measurable slowdown, per-source breakdown preserved.

DECIDED (Reed, 2026-08-31): the gain method is the ONLY propagation path
for v1 — including the PSD diameter-scale term. OAT re-retrievals are NOT
part of the release pipeline; they move to the todo list as a
validation-when-convenient item (run once on a stratified subsample to
calibrate the linearization, no per-release requirement). Two cheap
robustness measures replace them: (i) evaluate each nuisance's dy at BOTH
+1 and -1 sigma (two 0.4 ms forward calls) and average |G dy| — a secant
that catches first-order asymmetry of large perturbations like the 0.10
lnD scale without any retrieval rerun; (ii) an `uncertainty_flag` variable
marking windows where the linearization is suspect (posterior railing at a
grid edge, min chi^2 within ~0.2 of the gate, gf > ~1.5) so users know the
sigma there is a linear estimate under stress rather than silently trusting
it.

Nuisances (same list for both tiers):

| nuisance | 1-sigma perturbation | basis |
|---|---|---|
| PSD diameter scale | D -> D*exp(+/-0.10), all bins | UM 3b (RI-driven, correlated) |
| PSD concentration scale | N -> N*(1 +/- 0.10) common-mode | UM 3a f_rel (systematic part) |
| impactor D50 | +/-10% (config `impactor_d50_sigma`) | assumption — ask LARGE (Q2) |
| impactor steepness | gsd 1.15 -> {1.08, 1.25} | assumption — ask LARGE (Q2) |
| aero->geo density | rho 1.77 -> 1.77 +/- 0.20 (enters D50geo as 0.5*dlnrho) | UM 4b logic |
| neph calibration | all sca channels *(1 +/- f_rel) together | UM 1 (common-mode part) |
| PSAP sca-subtraction | all abs channels +/- 0.016*b_sp_measured together | UM 2 |

sigma_total^2 = sigma_posterior^2 + sum(sigma_nuisance^2), per product per
window. Optionally (config flag) store the per-source contributions as
extra variables (`*_unc_source` with a `source` dimension) for debugging;
default = total only.

Known double-count / correlation caveats, accepted for v1 and documented
(UM Sect. 0 warning): the common-mode calibration terms appear both inside
the chi^2 sigmas (marginal) and as perturbations (correlated); nuisances are
treated independent though composition couples them; the LAS diameter scale
is correlated with the retrieved CRI itself (Moore et al. 2021 circularity).

## netCDF layout

```
/windowed_uncertainty            (same (flight,time[,wavelength|dp_mid]) grids)
  /observations                  1-sigma of the measured window means:
      scattering_dry_measured    neph model  (per lambda)
      absorption_measured        PSAP model  (per lambda; uses measured b_sp)
      scattering_humidified_synthesized   wet-channel sigma (v1: neph model)
      dndlogdp                   per-bin diagonal sigma
      diameter_scale_sigma       scalar 0.10 (correlated lnD nuisance, NOT
                                 representable per-bin — documented as such)
  /retrievals                    1-sigma of the retrieved/derived products:
      refractive_index_real/imag, kappa,
      ssa / sca / abs / ext (state x wavelength),
      growth_factor_wet/ambient, wet/ambient CRI
      [optional] *_by_source with a "source" dimension
```

Mirror rule: identical names/dims to `/windowed` so users difference the
groups trivially; flags/counters/coordinates are NOT mirrored. Group attrs
cite UM and its reference list, state the 1-sigma convention, the
measured-values rule, and the caveats above verbatim-ish.

## Phases

- U1: `uncertainty_models.py` + unit tests (UM SSA table; Kupc et al.
  moment check for the PSD model). No behavior change.
- U2: chi^2 sigma switch (ISARA sigma args, bridge wiring, config with
  default "instrument" per the decision above), A/B of gate populations
  legacy-vs-instrument on both years to document the population change.
- U3: ensemble-gain propagation (`ASCENT_ACP/uncertainty_propagation.py`)
  + posterior-spread extension for derived products + uncertainty_flag;
  results into the bundle. (OAT validator script: todo, not release-gating.)
- U4: `/windowed_uncertainty` export + NETCDF_OUTPUT_SPEC addendum.
- U5: validation + release (V7): posterior widths vs perturbation sizes
  sanity, PSAP-SSA closed form vs propagated abs sigma, spot-check that
  kappa sigma grows toward RH ceiling.

## Open questions for Reed

1. Wet-channel sigma: include a gamma-parameterization term now or v2?
2. Impactor D50/gsd uncertainties: ask LARGE for the measured penetration
   curve, or accept the +/-10% / gsd-range assumptions?
3. ~~Gate on instrument sigmas or legacy tolerances?~~ DECIDED 2026-08-31:
   instrument sigmas drive gating AND weighting via the existing chi2-wmean
   machinery; legacy tolerances remain only as a regression option.
4. Per-source breakdown variables in the file, or total sigma only?
5. Report asymmetric intervals where perturbations are one-sided (e.g.
   volume-ish +12/-28% from UM 3b validation), or symmetrize? (v1 plan:
   symmetrize, note in attrs.)
6. 2020 neph regime: fixed "pm10" f_rel = 0.09, or leg-dependent inflation
   to 0.15-0.25 when coarse-dominated (AE-based switch)?
```
