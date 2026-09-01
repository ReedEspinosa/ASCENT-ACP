# Dry-RRI information content: diagnosis and path forward

Status: ANALYSIS COMPLETE 2026-09-01 (post-V10/V2 sizing-corrected runs).
Scripts: `scripts/posterior_ic_study.py`, `scripts/backscatter_fingerprint.py`,
`scripts/smps_opc_overlap.py`.

## 1. The question

Every V8+ run shows a flat dry-RRI posterior: per-window posterior std
~0.0285 (a uniform posterior over the 1.47-1.56 grid would give 0.0260) with
all window means pinned at the grid center ~1.515. Earlier versions (V5/V7,
and prior ISARA-style studies) showed window-to-window RRI dynamic range of
~0.01. Which is right — were the earlier retrievals fitting noise, or is the
covariance machinery suppressing real information?

## 2. Diagnosis: the nuisances point along the RRI signal

The measurement vector is [Sc450, Sc550, Sc700, Abs...]. Each error source
moves it in a characteristic pattern. For a median retrieved ACTIVATE window
(fractional change per channel):

| perturbation                    | dSca [450, 550, 700]      | dAE    |
|---------------------------------|---------------------------|--------|
| RRI 1.515 -> 1.56 (half grid)   | +11.2%, +10.9%, +12.2%    | -0.021 |
| sizing +5% lnD (1-sigma prior)  | +20.3%, +20.4%, +21.3%    | -0.019 |
| sizing +2% (theta-hat scatter)  |  +7.7%,  +7.7%,  +8.1%    | -0.009 |
| concentration +10% (1-sigma)    | +10.0%, +10.0%, +10.0%    |  0     |
| neph calibration +8% (1-sigma)  |  +8.0%,  +8.0%,  +8.0%    |  0     |

Half the RRI grid is comparable to ONE sigma of EACH of three nuisances with
the same spectral shape. Quantitatively (posterior_ic_study, 170-180 windows
per campaign): the whitened-space alignment cos(nuisance, RRI-signal) is
0.997 (sizing) and 0.99 (concentration scale); the production covariance
retains 7-12% of the RRI Fisher information; implied per-window sigma_RRI:

| error model                              | per-window sigma_RRI |
|------------------------------------------|----------------------|
| white + floor noise only                 | 0.001-0.003          |
| diagonal instrument sigma (V5/V7)        | 0.009-0.020          |
| instrument covariance w/ common modes    | 0.017-0.026          |
| production S (all nuisances, per window) | ~0.067 (> grid)      |

Variant re-runs (drop each covariance term) confirm no single term is
responsible: sizing, concentration scale, and the common-mode calibration
term all flatten RRI individually because they share its direction. The
multi-wavelength (AE) lever separates RRI from concentration/calibration but
NOT from sizing (same tilt sign, table above), and per-window AE noise is
comparable to the signal.

CAVEAT for users of the netCDF: `dry_RRI_accepted_std` saturates at the
grid-uniform value (~0.0286); the unbounded absolute posterior is ~0.07-0.10.
The reported std is GRID-CLIPPED and understates absolute uncertainty.

## 3. Interpretation

- Per-window ABSOLUTE RRI is not retrievable from neph+PSAP+OPC magnitude
  closure. This is a physical information deficit, not a processing bug.
- The V5/V7 (and literature) dynamic range is window-to-window CONTRAST:
  valid relative information (sigma ~0.01) under the implicit assumption
  that calibration/sizing/concentration commons are stable across windows,
  riding on an absolute scale uncertain to ~0.1 at face value:
  sizing 5% lnD -> 0.085 RRI-equivalent, concentration 10% -> 0.042,
  calibration 8% -> 0.034 (quadrature ~0.10). External anchors (RI sizing
  correction cross-validated by UHSAS ~1.5% lnD; SMPS-OPC stitch consistency
  ~5%; gas-calibration records ~3%) can bring the campaign-mean scale to
  ~0.035 — every 0.01 of absolute RRI must be bought with an external
  anchor; ensemble averaging removes noise but never the one-direction/
  four-knob degeneracy. Unmodeled physics (spectral RRI slope, coatings/
  non-sphericity, inlet cut) may push this back toward 0.05-0.1.
- IRI, SSA, extinction closure, and kappa are NOT affected: absorption sits
  outside the scattering common modes (LAS-vs-UHSAS IRI r = 0.98), and
  kappa is ratio-based (its weakness is the PSD tail, a separate issue).
- Chemically-derived RI (AMS) is deliberately NOT used as a constraint:
  the project goal is comparing optical vs chemical RRI (circularity).

## 4. Path forward: observables that break the degeneracy

Design principle: the degenerate direction is spectrally-flat MAGNITUDE.
What is needed are RATIOS (concentration/calibration cancel) whose RRI and
size responses differ in sign or shape.

### 4.1 TSI hemispheric backscatter fraction (b = Bsp/Sp) — primary lever

Computed from MOPSMAP phase functions for a representative retrieved PSD
(`backscatter_fingerprint.py`; both ideal 90-180 and TSI 90-170/7-170
truncated geometry give the same answer):

| perturbation        | db/b [450, 550, 700]   |
|---------------------|------------------------|
| RRI 1.52 -> 1.56    | +4.2%, +5.4%, +4.4%    |
| RRI 1.52 -> 1.48    | -5.1%, -5.4%, -4.4%    |
| sizing +5% lnD      | -3.7%, -0.7%, -0.2%    |
| sizing +2% lnD      | -1.5%, -0.1%, -0.2%    |
| concentration +10%  |  0, 0, 0               |

At 550/700, b is a nearly pure RRI channel (strong response, opposite-sign
weak sizing crosstalk, exact concentration immunity, calibration mostly
cancelling in the ratio); the blue-heavy sizing response means the SPECTRAL
SHAPE of b additionally separates sizing from RRI. Rough IC: b measured to
3-5% per window -> per-window absolute sigma_RRI ~0.02-0.03, independent of
the magnitude commons. Caveats: b is maximally sensitive to non-sphericity
(needs a coarse/dust screen) and to IRI (well-constrained by PSAP; retrieve
jointly). BLOCKER: neither ACTIVATE nor SEAC4RS archived the Bs channels —
raw backscatter must be requested from LARGE.

### 4.2 Direct instrument-signal forward modeling (no truncation correction)

Both campaigns' archived scattering has the Anderson & Ogren (1998)
truncation correction applied (stated in ICARTT headers). Plan: forward-model
the TRUNCATED signal (7-170 total, 90-170 backscatter, with the A&O angular
sensitivity) directly from the MOPSMAP a1 Legendre coefficients — the same
machinery as the LAS `qsca_partial` kernel — eliminating the correction and
its RI/size-dependent error. If only corrected data are available, the A&O
correction is invertible: it is multiplicative, C = a + b*Angstrom, so
uncorrected values follow from a fixed-point iteration (converges in 2-3
passes). The exact coefficient set used (sub-um vs no-cut; which Angstrom
pair) is not stated in the headers — a 1-3% ambiguity — confirm with LARGE.

### 4.3 SMPS-OPC overlap as an independent sizing measurement

SMPS diameters are electrical-mobility (RI-independent). The lnD shift
aligning each optical sizer onto the SMPS in their overlap (0.11-0.235 um;
SMPS bins above 0.235 are unusable — end-of-scan rolloff of ~10x) is a
retrieval-free measurement of that sizer's total sizing error
(`smps_opc_overlap.py`; geometric-mean flight spectra, high-signal rows).
SEAC4RS results:

- LAS: median +0.145 over well-fit flights (rms < 0.2). Predicted:
  +0.03 (Rayleigh-regime PSL-1.58-vs-1.52 calibration part) + 0.125
  (retrieval's non-RI theta-hat) = +0.155. Independent quantitative
  confirmation of the LAS sizing bias.
- UHSAS: overlap says +0.27 while neph closure says -0.03. Both can hold
  only if the sizing error is size-DEPENDENT (overlap probes 0.10-0.23 um
  near threshold; neph closure weights 0.3-0.8 um) or the SMPS overlap
  shape is biased. A constant-shift model is insufficient there.
- Per-flight scatter 0.12-0.19: currently a DIAGNOSTIC (it caught the LAS),
  not yet a per-flight anchor. Needs threshold counting-efficiency curves
  and a size-resolved (not constant-shift) model to graduate.

### 4.4 Joint LAS+UHSAS retrieval — retained, gated

Two sizers with different calibration materials and wavelengths respond to
RI differently, giving a per-window fingerprint that is not magnitude-shaped.
Gate: vet (and if needed refit) the LAS per flight via 4.3 first — the
SEAC4RS LAS bias would otherwise poison the joint fit.

## 5. Data requests to LARGE

1. Raw (uncorrected) TSI-3563 total scattering AND backscatter channels
   (Bs450/550/700), dry and humidified, ACTIVATE + SEAC4RS.
2. Which A&O98 coefficient set was applied to the archived data.
3. Any LAS/UHSAS threshold-region (0.10-0.25 um) counting-efficiency or
   sizing characterization; SMPS top-of-scan behavior.
4. Measured impactor penetration curve (existing todo item).
