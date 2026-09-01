# Plan: GRASP-kernel forward model to replace MOPSMAP (scoping, not yet implemented)

Drafted 2026-08-26 (Reed + Claude). Status: **scoped only — do not implement
until discussed.**

> **DECISION 2026-08-31 (Reed): go with the MOPSMAP-dataset engine, not the
> GRASP kernels.** A 1.5 MB extract of the sphere qext/qsca table
> (10 mr × 17 mi × 2085 size parameters) now lives in the ISARA_code fork
> at `mopsmap_sphere_table/` (commit eb0cd02) with its extraction script,
> GPL text, and provenance README (GPL data as mere aggregation in the
> MIT repo). Table validated to 0.21% worst-case vs exact Mie including
> GSD-1.05 PSDs and mid-cell CRIs. The engine-swap architecture below
> (forward_engine switch, vectorized CRI/kappa searches, NaN-fill,
> validation tiers) carries over unchanged — only the table and the
> (now unneeded) de-smoothing step differ. The GRASP-kernel analysis is
> retained below as reference; `kerneloptics.py` in GEOSmie_TOA remains
> the validated evaluator for the GRASP-format files (shape studies).

## Goal

Replace the per-candidate MOPSMAP Fortran subprocess (and the per-bin-pattern
optics LUT machinery it forced) with direct integration over precomputed
GRASP kernel tables. Motivation, in order:

1. **Campaign portability** — no Fortran build, no per-campaign LUT builds,
   any bin structure works immediately (the whole point of this pipeline).
2. **Speed** — the kappa search collapses from ~10^2-10^3 subprocess launches
   per window to vectorized NumPy; the 4x retrieval load from the FCDP-fix
   rerun becomes irrelevant.
3. **Correct wet-CRI coverage** — the kernel mr axis (1.29–1.70) natively
   spans the water-mixed refractive indices (→1.33) that the dry MOPSMAP LUT
   grid (1.47–1.56) cannot reach.
4. **Future shapes** — same-format kernel files exist for single-aspect-ratio
   spheroids and Saito hexahedra (see "Shape flexibility").

## Source tables

`/Users/wrespino/Synced/STG_AerosolModelExchange/GRASP-LUT-Export/GRASP-Kernels_netCDF-Versions/`

| file | size | use |
|---|---|---|
| `kernel-grasp-v1.1.3-integrated_V4.nc` | 242 MB | primary (sphere = ratio index 0; one spheroid ratio 2.99 at index 1) |
| `kernel-DLS-spheroids-singleAspectRatios_V4.nc` | 2.9 GB | future: per-aspect-ratio spheroids |
| `kernel-Saito-Hexahedra_psi0.7_*.nc` | 0.6/1.5 GB | future: hexahedra |

Grid of the primary file: `x` 41 log pts (~8.5/decade, x = 2πr/λ ∈
0.012–626), `mr` 22 pts (1.291–1.696, step ~0.019), `mi` 16 log pts
(0–0.5), `ratio` 2. Variables used: `ext`, `sca` (volume-normalized cross
sections, 1/um). `abs = ext − sca`. Memory footprint of what we load:
2×22×16×41 floats ≈ 0.5 MB — read once, keep in RAM; never touch `scama`
(the 181-angle matrix) unless phase functions are wanted later.

**Convention VERIFIED exactly (2026-08-31):** the volume-normalized kernels
are stored at reference wavelength λ₀ = 0.340 µm; use at λ requires
`K_λ(x) = K_stored(x) · (λ₀/λ)`. This is now closed: the identity
`ext ≡ 3π·qext/(2x·0.340)` holds to machine precision over the entire grid
(`scripts/mie_ground_truth.py`), so the scaling is exact physics
(K = 3πQ/(2xλ)), not an empirical fit.

## Measured baseline accuracy (2026-08-26 A/B)

30 real retrieved 2021 PSDs, GRASP (bilinear mr/log-mi, log-x interp,
midpoint-volume integration) vs the MOPSMAP 51-bin LUT at identical CRI:

```
CRI 1.52+0.005i          450 nm         550 nm         700 nm
ext                      0.983          0.990          1.000
sca                      0.982          0.989          0.999
abs (ext−sca)            1.013          1.017          1.021
per-PSD spread (sca)     ±1% (p5–p95 around median)
```

Same picture at 1.55+0.002i. A coherent ~1.5% blue→red slope remains; part
of it is likely the quick integration scheme (midpoint volumes vs MOPSMAP's
piecewise-linear dN/dlogDp integration), to be resolved in validation
Tier 1.

## Ground-truth accuracy: who is actually right? (2026-08-31)

The A/B above measures *difference*, not *error*. Benchmark against direct
high-resolution exact Mie (`scripts/mie_ground_truth.py`: broad lognormal
Dg=0.1 µm, lnσ=0.7, piecewise-linear dN/dlogDp on 51 nodes, exact CRI so no
engine interpolates in m; 450/550/700 nm; 3 CRI cases on- and off-node):

| engine / exact | 450 nm | 550 nm | 700 nm |
|---|---|---|---|
| MOPSMAP (ext & sca) | 1.001–1.002 | 1.001–1.002 | 1.002 |
| GRASP kernels (best usage) | 0.983–0.984 | 0.980–0.982 | 0.982–0.986 |

**RESOLVED (2026-08-31, later same day): the GRASP kernels contain no
measurable Mie error.** The generation rule was identified exactly: each
stored value is the triangular-in-ln(x) basis average of the *volume*
kernel K_v = 3πQ/(2xλ₀) — reproduced to machine precision at on-node CRI
for both ext and sca (averaging Q instead of K_v, or a triangle in x, or
a dN-weighted ratio all fail at the 1–3% level; that mis-guess was the
source of the earlier "~1% intrinsic" claim).

Consequently the naive-usage deficit is pure representation error and is
**exactly invertible**: de-smooth the stored averages by solving the
tridiagonal mass-matrix system (g_{i−1}+4g_i+g_{i+1})/6 = K̄_i (grid is
uniform in ln x, h=0.2716) once per (mr,mi) slice at load time, then
linearly interpolate the de-smoothed nodes in log x and integrate against
the fine PSD. Measured accuracy of the de-smoothed engine vs exact Mie:
**worst case 0.18%** over {broad Dg=0.10/lnσ=0.70, narrow Dg=0.15/lnσ=0.35,
coarse Dg=0.30/lnσ=0.50} × CRI {1.52+.0061i, 1.54+.005i, 1.50+.01i,
1.40+.002i (wet-side, off-node)} × {450,550,700} nm — slightly *better*
than MOPSMAP, with **no calibration factor**. Usage notes: de-smooth after
mr/mi interpolation (linear ops, order immaterial); never use raw K̄ as
point values (−1.7%) nor node-quadrature of the PSD (−0.5 to −4%).

**Ripple-aliasing robustness (2026-08-31, addressing the "wiggles beating
with node spacing" concern):** swept gf 1.00→1.38 in 33 sub-node steps
(node spacing = ×1.312 in x) at 550 nm — i.e., sliding every Mie wiggle
continuously across the node grid. De-smoothed error stays bounded and
phase-stable: peak-to-peak oscillation 0.19% (broad lnσ=0.70), 0.03%
(lnσ=0.35), 0.61% (lnσ=0.20, GSD 1.22); worst single point 0.34%. The
naive smoothed-K̄ interpolant is the one that aliases: p2p 0.6%, 2.5%,
4.3% on the same sweeps. Why: solving the mass-matrix system makes the
piecewise-linear K̂ the *Galerkin (L2) projection* of the true kernel —
node values come only from integrals, never point samples, so sub-grid
wiggles are already correctly integrated into K̄ and the projection
preserves their integrated effect; the residual (K−K̂) is orthogonal to
all piecewise-linear functions, so its integral against a smooth PSD is
second-order small regardless of ripple phase. The mass-matrix inverse
gain is bounded by 3 (eigenvalues in [1/3, 1]) — mild sharpening, not an
ill-posed deconvolution. GRASP's own retrieval never interpolates the
kernel: it represents the PSD in the same triangular basis, where
Σ c_i K̄_i is exact — the smoothing is that Galerkin pairing, not a
ripple defense. De-smoothing simply restores the pairing when the PSD
lives on a finer grid than the kernel nodes. (Limit: a quasi-monodisperse
PSD narrower than the node spacing would defeat 41 nodes either way;
angular/phase-function quantities would be far less forgiving — not used.)

**Real measured PSDs (2026-08-31):** 60 unique all-finite windowed stitched
PSDs from the shipped 2021 V4 file, de-smoothed GRASP vs exact Mie at
on-node CRI 1.5229+0.0059i: median ratios 0.9988–1.0004 (ext & sca,
450/550/700 nm), p5–p95 within ±0.3%, worst single case −0.41%. Measured
jaggedness couples to the ripple residual with random sign (variance, not
bias) — real PSDs perform at least as well as the smooth lognormal tests.

## Critical assessment: the speedup, honestly (2026-08-31)

MOPSMAP is itself a LUT (precomputed single-particle tables, interpolated
in Fortran) — neither engine computes Mie on the fly. A measured MOPSMAP
subprocess call costs ~33 ms, nearly all process overhead (temp files,
fork, dataset reads, stdout parse). The CRI grid search already amortizes
this into our own hat-kernel LUT and is effectively free; **the entire
speed win is the kappa scan** (~300 subprocess calls ≈ 10 s per window —
which reproduces the measured 363 s 2021 runtime almost exactly). GRASP
takes the retrieval stage from ~1 h/year (post-FCDP-fix load) to seconds;
end-to-end becomes ~2–4 min, bounded by loading/windowing/export. So:
~10–30× on the retrieval stage, but "1 hour → 3 minutes", not
"days → minutes" — an iteration convenience, not a capability win.

**The same speedup is achievable while keeping MOPSMAP's data** — and
without its Fortran: `optical_dataset/spheres/` IS the single-particle
kernel table (170 nc files, exact Mie ~231 pts/decade, mreal 1.28–1.64
step 0.04 — covers the wet side — mimag 21 log nodes; 1.9 GB with angular
data, but the ext/sca extract is a few MB). A one-time Python repackaging
plus the same ~150 lines of interpolation/integration yields an engine
formally identical to the GRASP one. So Fortran-vs-not is no longer the
decision axis; both options are "a small nc table + NumPy", and the
subprocess kappa scan dies either way. Remaining trade (near-tie):
GRASP table = validated already (kerneloptics.py in GEOSmie_TOA, ≤0.4%
worst on real PSDs), finer mr (0.019 vs 0.04), self-owned distribution,
shape files in-format; repackaged-MOPSMAP table = point values (no
de-smoothing step), likely ~0.1–0.2%, but the full validation exercise
would have to be repeated and the derived data is redistributed from the
Gasteiger & Wiegner dataset. Do-nothing baseline (subprocess + NaN-fill)
remains workable at ~1 h/year reruns with a per-machine Fortran build.

**Measured facts on the repackaged/direct-read MOPSMAP-dataset option
(2026-08-31):** (a) no extraction needed for speed — lazy netCDF reads of
sizepara/qext/qsca are <1 ms/file and only the 4–6 files bracketing the
CRI are touched; the 1.9 GB costs disk/distribution only. (b) License:
GPL (v2 per Zenodo 10.5281/zenodo.1284217, v3 in the bundled COPYING);
subset extraction + redistribution permitted but the derived file carries
GPL + Gasteiger & Wiegner (2018) citation. (c) The 0.04 mr grid is benign:
worst 0.11% across a fine mr sweep through a full cell, smooth in mr, so a
0.01-step CRI search is not distorted. (d) 231 pts/decade vs GRASP's 41
basis-averaged nodes matters ONLY for narrow PSDs: gf-sweep worst error
0.03% (dataset) at every width tested vs GRASP 0.26%/1.9%/3.5% at GSD
1.22/1.11/1.05. Both engines ≤0.4% for broad ambient PSDs. Decision axes
that remain: robustness to unseen narrow PSDs and GPL-vs-own-license
(favors MOPSMAP-dataset / GRASP respectively), shape files in Reed's
format (GRASP), validated module already committed (GRASP; the dataset
engine passed the same harness ad hoc but is unpackaged).

## Architecture

New module (proposed: `ISARA_code/grasp_optics.py`, kept in ISARA_code so
ISARA stays self-contained):

- `GraspKernels.load(path, shape_index=0)` — reads x/mr/mi grids + ext/sca
  once; caches in module state (workers inherit via fork).
- `coefficients(dndlogdp, dpg_um, mr, mi, wvl_nm, gf=1.0)` → (ext, sca) in
  Mm⁻¹ per wavelength. Growth enters as `x = π·gf·dpg/λ` — kernels are
  continuous in x, so **no bin remapping, no spillover truncation, no
  second LUT**. Vectorize over CRI candidates and kappa candidates with one
  einsum each.
- Engine switch in `IsaraConfig`: `forward_engine: "mopsmap" | "grasp"`
  (default stays `"mopsmap"` until validation passes). MOPSMAP path is kept
  permanently as the validation reference.

Integration points inside `ISARA.py`:

1. `Retr_CRI` grid search: predicted coefficients for all 350 candidates =
   one tensor product (kernels interpolated to the window's bin diameters
   once per window, then contracted with the candidate CRI axis).
2. Kappa search: full κ = 0–1.4 scan at 0.001 step = 1400 vectorized
   forward evaluations (microseconds). Monotonicity concerns and bisection
   both become moot.
3. `humidified_optics` (wet/ambient states): same call with gf and mixed
   CRI.
4. The surface-conserving remap (`humidify.py`) remains **output-only**
   (fixed nc4 bin grid); it is no longer needed anywhere in the retrieval.

Interpolation scheme (initial): linear in mr, linear in log(mi) with an
mi=0 guard (first node is 1e-10), linear in log(x). Evaluate cubic-in-log-x
during validation; the kernels are basis-integrated and smooth, so linear
may already suffice.

### PSD integration scheme

Match MOPSMAP's representation to isolate kernel error from scheme error:
integrate the piecewise-linear dN/dlogDp between bin centers (trapezoid in
logD against kernel values at quadrature points), not midpoint volumes.
This is the first thing to try against the residual 1.5% slope.

### NaN-bin handling (independent "no-brainer", do regardless of engine)

Fill interior NaN bins by linear interpolation of dN/dlogDp (in logD) from
neighbors — numerically almost identical to what dropping bins already
makes MOPSMAP do — zero-fill edge gaps, refuse gaps wider than N bins
(configurable, default ~3), and record `n_bins_filled` as a diagnostic.
Under the MOPSMAP engine this lets every window reuse the master LUT
(kills the 2020 per-pattern fallback); under the GRASP engine it simply
defines the PSD before integration. Same code either way.

## Validation plan (gates, in order)

1. **Forward closure** (script: `scripts/validate_grasp_forward.py`):
   ~100 windows per year × dry state × all 6 λ × several CRI points, GRASP
   vs direct MOPSMAP. Refine integration scheme until the residual is flat
   in λ or understood. Target: |median bias| ≤ 0.5% per λ for sca/ext after
   any per-λ calibration; spread ≤ ~1%. Repeat for humidified states
   (gf 1.1–1.8) — this exercises the x-shift path MOPSMAP does natively.
2. **Calibration decision**: if a λ-dependent residual persists and is
   kernel-intrinsic, adopt a fixed per-λ correction factor (stored with
   provenance in the module + nc header) or accept it — Reed's call. The
   kappa 1% match criterion is the only tolerance tight enough to care.
3. **End-to-end A/B**: full retrieval on both years with both engines;
   compare CRI/kappa/forward-calc distributions. Expect shifts well inside
   the acceptance widths (20% sca, 1 Mm⁻¹ abs); quantify like the CRI-grid
   expansion was quantified (`scripts/compare_cri_grids.py` pattern).
4. Only then flip `forward_engine` default and rerun for release.

## Shape flexibility (later)

The loader takes any same-format file + shape index, so spheroid
aspect-ratio and hexahedra sensitivity studies become a config change.
Files are GB-scale, but we only ever read ext/sca slices (lazy per-shape
load stays ~MB). Suggested config: `kernel_file`, `kernel_shape_index`.

## Open questions for Reed

- ~~Confirm the λ₀ = 0.340 µm convention~~ CLOSED 2026-08-31: verified
  exactly via the `ext ≡ 3π·qext/(2x·0.340)` identity (mi stored negative).
- ~~Calibration policy~~ CLOSED 2026-08-31: no calibration needed — the
  basis convention was identified exactly (triangular-lnx average of the
  volume kernel) and mass-matrix de-smoothing recovers exact Mie to ≤0.2%.
- Portability vs exactness: with de-smoothing, GRASP matches MOPSMAP
  accuracy, so the extended-MOPSMAP-LUT alternative is likely moot unless
  Fortran-availability is a non-issue. Reed to confirm direction.
- Where the module should live (ISARA_code vs ASCENT_ACP) — proposal:
  ISARA_code.
- Does anything downstream ever need phase-function/lidar quantities? If
  yes, `scama`/`qb`/`lidar_ratio` are in the same tables.

## Rough effort

Module + tests ~200–300 lines; ISARA integration behind the engine switch
~100 lines; validation scripts reuse existing harness patterns. Estimate:
one session to working prototype + Tier 1, one more for Tier 3 A/B and
review. Independent of (and compatible with) the pending FCDP-fix rerun —
but sequencing the rerun AFTER the engine swap would fold the 4x retrieval
load into a fast engine and avoid one throwaway multi-hour MOPSMAP run.
