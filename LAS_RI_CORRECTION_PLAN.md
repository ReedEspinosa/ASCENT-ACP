# Plan: LAS refractive-index sizing correction (scoping, not yet implemented)

Drafted 2026-08-31 (Reed + Claude). Status: **IMPLEMENTED 2026-09-01**
(sizing_correction.py + qsca_partial kernel in mopsmap_spheres_v2.nc +
per-candidate remapping in ISARA Retr_CRI/Retr_PSD + marginal/propagation
consistency; identity exact to 1e-9; geometry = Moore et al. 2021's
33-147/72.5-104.8 for both instruments). Results below the design
(Sect. 10). This is the concrete realization of the "raw-signal LAS
refit" (todo_reed.txt HIGH PRIORITY): because each LAS bin is a fixed
*signal threshold*, refitting the raw signal is exactly equivalent to
remapping the bin diameters per candidate refractive index — which is the
form implemented here.

## 1. Problem and evidence

The LAS assigns optical-equivalent diameters using a calibration RI
(AmmSO4 1.52 on ACTIVATE, PSL 1.59 on SEAC4RS). Ambient particles with a
different RI scatter differently, so the assigned diameters are biased in
a correlated, RI-dependent way. The V9 joint posterior measures this
per window as `sizing_lnD_shift`:

| campaign | LAS calibration | median lnD shift |
|---|---|---|
| ACTIVATE 2021 | AmmSO4 (1.52) | +0.045 |
| ACTIVATE 2020 | AmmSO4 (1.52) | +0.006 |
| SEAC4RS 2013 | PSL (1.59) | **+0.181** |

Sign and magnitude match Moore et al. (2021, AMT): PSL calibration
undersizes ambient aerosol by ~10-25% in diameter. Consequence on
SEAC4RS: the sizing nuisance dominates the CRI posterior and dry RRI
collapses to ~1.52-1.53 (no dynamic range; PI-Neph comparison
uncorrelated). Correcting sizing per candidate RI removes the Moore
circularity at its root.

## 2. Physics

Instrument response for a sphere of diameter D and complex RI m at the
LAS wavelength lambda_LAS:

    R(D, m)  ∝  C_sca(D, m) * [ (1/2) ∫_{theta1}^{theta2} P11(theta; D, m) sin(theta) dtheta ]

i.e. the partial scattering cross section into the collection solid
angle. The calibration fixes the signal thresholds: bin edge k sits at
signal S_k = R(D_k^cal, m_cal). For a candidate RI m the corrected edge
D_k(m) solves

    R(D_k(m), m) = R(D_k^cal, m_cal).

Counts in a bin are conserved; dN/dlogDp is rescaled by the bin-width
Jacobian dlogD_cal/dlogD(m).

**Non-monotonicity**: at 633 nm, R(D) undulates above ~0.5 um (the reason
UHSAS moved to 1054 nm). The instrument firmware itself sizes through an
effectively monotonic calibration curve, so the correction must use the
same convention: construct a monotonic effective response (bin-averaged /
monotone-envelope smoothing of R over the multivalued region) for BOTH
m_cal and the candidate m, and map threshold-to-threshold through those.
Which smoothing convention best matches TSI's is open question Q2 —
Moore et al. 2021's treatment is the anchor to reproduce first.

## 3. Geometry sensitivity — MEASURED, and it mostly doesn't matter (2026-08-31)

Reed's differential argument tested numerically (ALPH1-reconstructed P11,
monotone-envelope responses, mean lnD shift over D_cal 0.1-1.0 um):

| RI change | 22-158 | 33-147 blocked | 35-145 | total 0-180 |
|---|---|---|---|---|
| PSL 1.60 -> 1.52 | +0.045 | +0.061 | +0.068 | +0.031 |
| PSL 1.60 -> 1.44 | +0.113 | +0.144 | +0.156 | +0.091 |
| 1.52 -> 1.48 | +0.032 | +0.043 | +0.044 | +0.027 |

Across every plausible collection geometry the correction changes by only
~+/-0.01-0.02 in lnD — small against the 0.05-0.18 corrections being
targeted. Because the mapping depends on the RATIO of responses at two
RIs, geometry errors largely cancel; we can proceed with a nominal
geometry and carry a ~0.015 lnD geometry term in the residual sigma
instead of blocking on TSI drawings. The monotonicity convention is the
bigger lever, but only in the resonance region (~0.5-1 um); below
~0.4 um (where most submicron PSD signal lives) all conventions
coincide with the Rayleigh-regime 6th-root-of-cross-section behavior
Moore et al. use. (A boxcar-smoothing variant was also tested but its
implementation had an edge artifact — redo properly during
implementation, anchored on Moore et al. Fig. 7.)

## 3b. Measurement geometry (Q1, downgraded from blocker to parameter)

From Moore et al. (2021, AMT, open access — fetched): LAS 3340 =
intracavity HeNe 633 nm (TEM00, ~1-10 W intracavity), two pairs of
wide-angle Mangin mirrors (exact LAS polar angles not stated in the
retrieved text; the full PDF or the TSI manual would pin them, but per
Sect. 3 this is now a ~1-2% effect). **UHSAS geometry is stated exactly:
33-147 deg with 72.5-104.8 deg blocked, at 1054 nm** — so the UHSAS
kernel (in scope per Reed) can be built with no assumptions. Implementation keeps the
geometry as config: `las_lambda_nm`, `las_theta_min/max_deg`,
`las_azimuth` (full/partial), per campaign+instrument. With full-azimuth
collection the response weight is S11 alone regardless of laser
polarization; partial azimuth would add an S12 term (the a2/b1 expansion
coefficients are also stored in the dataset, so this is extensible).

## 4. LUT update (MOPSMAP-derived P11 response kernel)

VERIFIED feasible offline from the existing MOPSMAP single-particle
files: each sphere_<mr>_<mi>.nc stores the Legendre expansion
coefficients of the phase matrix (Mishchenko ALPH1... convention;
variable `a1`, cumulative (lmax+1) indexing over the 2085 size
parameters; reconstruction test normalizes (1/2)∫P11 sin = 1.0000).

Extend `mopsmap_sphere_table/build_mopsmap_sphere_table.py` to add, for
the configured geometry:

    qsca_partial(mreal, mimag, sizepara)
      = qsca * (1/2) ∫_{theta1}^{theta2} P11(theta) sin(theta) dtheta

so the response is R ∝ qsca_partial * D^2 evaluated at
x = pi D / lambda_LAS. Geometry and provenance go in the variable attrs;
bump the table to `mopsmap_spheres_v2.nc` (same GPL packaging). Build
cost: Legendre evaluation over 170 files x 2085 sizes — minutes.
The mr grid (1.28-1.64) covers both calibration RIs and the dry
candidate grid; mi coverage (to 0.1376) covers absorbing-particle
response depression, which matters (IRI reduces R, shifting sizing).

## 5. Mapping construction and retrieval integration

At engine load, per (mr, mi) table node: build the monotonic effective
R(D) at lambda_LAS on a fine logD grid; build the calibration curve once
at m_cal; store the node's mapping D_cal -> D(m) as a small spline.
At retrieval time, per candidate (RRI, IRI): bilinear-interp the mapping
in (mr, log mi), shift the optically-sized instruments' bin centers/edges (LAS via the
633 nm kernel; UHSAS via its own 1054 nm kernel with the exactly-known
geometry — included per Reed; SMPS is mobility-based and unchanged), rescale dN/dlogDp by the width Jacobian, re-evaluate
the impactor penetration on the corrected (true) diameters, then forward
model as usual. Growth factors apply to corrected diameters. Cost per
candidate: one vector interpolation (~microseconds) — runtime unchanged.

Config: `psd.las_calibration_ri` (1.52 ACTIVATE / 1.59 SEAC4RS),
`isara.las_sizing_correction: bool` (default off until validated), plus
the geometry fields above. The SMPS/LAS stitch stays defined on the
calibrated grid (the hand-off is a data-provenance boundary, not a
physical size).

## 6. Uncertainty bookkeeping

- Shrink the correlated lnD nuisance prior from 0.10 to a residual
  ~0.03-0.05 (calibration transfer, geometry uncertainty, non-spherical
  response error) — value to be set from validation.
- Keep `sizing_lnD_shift` as the acceptance dial: after correction it
  should collapse toward 0 on BOTH calibration conventions.
- New (v2, optional) nuisances: collection-angle limits, calibration RI.

## 7. Validation gates (in order)

1. **Kernel anchor**: reproduce Moore et al. (2021) response-vs-size
   curves for PSL and AmmSO4 and their reported LAS sizing offsets for
   ambient RIs. This validates geometry + smoothing convention together.
2. **Identity check**: candidate m = m_cal must give the identity mapping
   to machine precision.
3. **ACTIVATE rerun**: expect sizing_lnD_shift ~ 0 (from +0.045), RRI up
   slightly (consistent with the earlier +4% analysis), 2021 red-residual
   reduced.
4. **SEAC4RS rerun (decisive)**: sizing_lnD_shift from +0.181 to ~0; dry
   RRI dynamic range restored; PI-Neph RI comparison rerun — emergence of
   correlation is the external validation of the whole chain. Cross-check
   the corrected LAS PSD against the UHSAS-AmmSO4 PSD in the 0.1-0.9 um
   overlap (independent sizing convention, same air).

## 8. Open questions (Reed)

- Q1 (downgraded): LAS exact angles are a ~1-2% effect (Sect. 3); use a
  nominal 22-158 deg, carry a geometry term in the residual sigma, refine
  if the Moore et al. PDF / TSI manual is obtained.
- Q2: monotonicity convention — matters only in 0.5-1 um; anchor on
  reproducing Moore et al. Fig. 7 during implementation; default =
  monotone envelope.
- Q3: residual lnD prior after correction — set from validation; expect
  ~0.03-0.05 (geometry ~0.015 + calibration transfer + convention).
- Q4: RESOLVED (Reed): UHSAS kernel included (geometry known exactly).
- Q5: spectral-RI caveat (Reed): the correction kernels are evaluated at
  the instrument wavelength (633 / 1054 nm) using the retrieval's
  spectrally FLAT candidate RI. RRI dispersion visible->1054 nm is small
  for most aerosol; IRI can differ strongly (brown carbon) but perturbs
  sizing weakly at high SSA. Documented approximation; revisit alongside
  the per-wavelength CRI retrieval (todo item 8).

## 9. Effort estimate

Table extension + kernel validation vs Moore et al.: ~1/2 session.
Mapping + engine integration + tests: ~1 session. Campaign reruns +
validation report: ~1/2 session. All behind a default-off config switch
until gate 4 passes.


## 10. Implementation results (2026-09-01)

Runs: ACTIVATE V10 (2021+2020), SEAC4RS V2 (LAS run + UHSAS run).

| run | CRI ok (prev) | residual sizing shift (prev) |
|---|---|---|
| ACTIVATE 2021 V10 | 2442 (2173) | +0.028 (+0.045) |
| ACTIVATE 2020 V10 | 1579 (1832) | -0.006 (+0.006) |
| SEAC4RS LAS V2 | 2770 (2073) | +0.125 (+0.181) |
| SEAC4RS UHSAS V2 | 1673 (new) | -0.027 |

- Validation gates: identity exact; ACTIVATE residual halves; UHSAS run
  near-zero residual. SEAC4RS LAS retains a ~+0.12 lnD NON-RI sizing bias
  (the RI part, ~+0.06, is removed; the UHSAS cross-run proves the rest is
  LAS-2013-specific) — its residual prior is set to 0.10 accordingly
  (config), 0.05 elsewhere. 2020's success drop reflects the honest
  tightening: its wide sizing prior had been absorbing non-sizing misfit
  (PSAP artifacts).
- Cross-comparison matrix (processing_logs/sizing_v2_plots/): ISARA
  internal consistency LAS-vs-UHSAS runs: IRI r=0.98, RRI med diff 0.002;
  kappa disagrees strongly (LAS 0.30 vs UHSAS 0.05 medians, DASH-SP ~0.19
  between; r=0.25) — kappa is sensitive to the PSD tail near/above the
  550 nm resonance (UHSAS grid ends at 0.92 um). DASH-vs-LAS-run kappa
  improved to Delta +0.12 (was +0.22).
- HONEST LIMIT: ISARA dry-RRI dynamic range is still ~0.01 wide (both
  runs) — with a 0.05-0.10 sizing nuisance still marginalized in S, the
  CRI-sizing degeneracy keeps the posterior mean prior-dominated, so the
  PI-Neph RRI comparison remains uncorrelated. The structural next step is
  a JOINT LAS+UHSAS retrieval (both sizers in one fit constrains the
  sizing nuisance per window; their demonstrated internal consistency
  makes this promising), or an externally validated tighter residual.
