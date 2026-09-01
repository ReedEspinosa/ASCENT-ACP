#!/usr/bin/env python
"""Compare CRI solution-selection estimators for the ISARA grid search.

Estimators (all operate on the same per-candidate forward coefficients):
  linf-mean  : current ISARA — accept candidates with ALL sca within 20% and
               ALL abs within 1 Mm^-1 (an L-infinity gate), report the mean.
  chi2-mean  : accept candidates with reduced chi^2 < 1 (sigma = 20% sca,
               1 Mm^-1 abs), report the mean.
  chi2-min   : report the single best-fit candidate (gate: chi2_min < 1).
  chi2-wmean : chi^2-weighted mean over all candidates, weights
               exp(-n_ch*chi2/2) (Gaussian posterior mean on the grid);
               same chi2_min < 1 success gate.

Part 1 (synthetic truth): real window PSDs, truth CRI drawn uniformly in
RRI 1.47-1.56 / log-uniform IRI 1e-4-0.02 (plus 20% near-zero IRI),
forward-modeled with the table engine and perturbed with noise; reports
bias/RMSE of retrieved RRI, IRI and dry SSA(550) per estimator. Same-engine
inverse crime is intentional: this isolates estimator statistics.

Part 2 (real windows): the same estimators on measured coefficients;
reports success rates and how far each estimator moves RRI/IRI/SSA from
the current one.

Usage: python scripts/estimator_study.py [n_psd] (default 150)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/wrespino/Synced/Local_Code_MacBook/ISARA_code")
import sphere_optics  # noqa: E402

from ASCENT_ACP.config import PipelineConfig  # noqa: E402
from ASCENT_ACP import filtering, pipeline, sizebins, windows  # noqa: E402
from ASCENT_ACP.windows import psd_col_name  # noqa: E402

WVL_SCA = [450, 550, 700]
WVL_ABS = [470, 532, 660]
SIG_SCA = 0.20        # relative, per channel (also the L-inf tolerance)
SIG_ABS = 1e-6        # m^-1 (1 Mm^-1)


def cri_grid():
    import ISARA
    return ISARA.default_CRI_grid(1.47, 1.56, 0.01)


def candidate_coeffs(dpg_um, dnd_cm3, grid_cri):
    """(n_cand, 3) sca and abs in m^-1 for every grid candidate + SSA550 fn."""
    args = dict(size_equ={'m': 'cs'}, nonabs_fraction={'m': 0},
                shape={'m': 'sphere'}, density={'m': 1.0}, RH=0, kappa=0,
                num_theta=2)
    sd = {'m': dnd_cm3 * 1e6}
    dp = {'m': dpg_um}
    wvl = np.array(sorted(WVL_SCA + WVL_ABS))
    sca = np.empty((len(grid_cri), 3))
    ab = np.empty((len(grid_cri), 3))
    ssa550 = np.empty(len(grid_cri))
    for k, (rri, iri) in enumerate(grid_cri):
        r = sphere_optics.Model(wvl, dndlogdp=sd, dpg=dp,
                                RRI={'m': rri}, IRI={'m': iri}, **args)
        sca[k] = [r[f'ssa_{w}'] * r[f'ext_coeff_{w}_m-1'] for w in WVL_SCA]
        ab[k] = [(1 - r[f'ssa_{w}']) * r[f'ext_coeff_{w}_m-1'] for w in WVL_ABS]
        ssa550[k] = r['ssa_550']
    return sca, ab, ssa550


def chi2(sca_c, abs_c, sca_m, abs_m):
    r = ((sca_c - sca_m) / (SIG_SCA * sca_m)) ** 2
    a = ((abs_c - abs_m) / SIG_ABS) ** 2
    return (r.sum(axis=1) + a.sum(axis=1)) / (r.shape[1] + a.shape[1])


def estimators(grid_cri, sca_c, abs_c, ssa_c, sca_m, abs_m):
    """Returns {name: (rri, iri, ssa550, n_used) or None}."""
    out = {}
    # linf-mean (current ISARA acceptance)
    rel = np.abs(sca_c - sca_m) / sca_m
    ada = np.abs(abs_c - abs_m)
    acc = (rel < SIG_SCA).all(axis=1) & (ada < SIG_ABS).all(axis=1)
    out['linf-mean'] = _mean_of(grid_cri, ssa_c, acc)
    x2 = chi2(sca_c, abs_c, sca_m, abs_m)
    out['chi2-mean'] = _mean_of(grid_cri, ssa_c, x2 < 1.0)
    k = int(np.argmin(x2))
    ok = x2[k] < 1.0
    out['chi2-min'] = (grid_cri[k, 0], grid_cri[k, 1], ssa_c[k], 1) if ok else None
    if ok:
        w = np.exp(-0.5 * 6 * (x2 - x2[k]))
        w /= w.sum()
        out['chi2-wmean'] = (float(w @ grid_cri[:, 0]), float(w @ grid_cri[:, 1]),
                             float(w @ ssa_c), int((w > 1e-3).sum()))
    else:
        out['chi2-wmean'] = None
    return out


def _mean_of(grid_cri, ssa_c, mask):
    if not mask.any():
        return None
    return (float(grid_cri[mask, 0].mean()), float(grid_cri[mask, 1].mean()),
            float(ssa_c[mask].mean()), int(mask.sum()))


def main(n_psd=150, seed=0):
    cfg = PipelineConfig.from_json("configs/activate_2021_full.json")
    df, _ = pipeline.load_inputs(cfg)
    grid = sizebins.build_grid(df, cfg.psd)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)
    wdf = windows.aggregate(df, optical, masks, grid, cfg)
    good = wdf[wdf.window_qc_flag == 0]
    sel = good.iloc[:: max(1, len(good) // n_psd)][:n_psd]
    gcri = cri_grid()
    rng = np.random.default_rng(seed)

    # precompute candidate coefficients per window (shared by both parts)
    cases = []
    for _, row in sel.iterrows():
        dnd = np.array([row[psd_col_name(d)] for d in grid.dpg_um], float)
        fin = np.isfinite(dnd)
        if fin.sum() < 10:
            continue
        sca_c, abs_c, ssa_c = candidate_coeffs(grid.dpg_um[fin], dnd[fin], gcri)
        sca_m = np.array([row[f"Sc{w}_dry_mean"] for w in WVL_SCA], float) * 1e-6
        abs_m = np.array([row[f"Abs{w}_mean"] for w in WVL_ABS], float) * 1e-6
        cases.append((sca_c, abs_c, ssa_c, sca_m, abs_m))
    print(f"{len(cases)} windows with candidate coefficients")

    # ---------------- Part 1: synthetic truth --------------------------------
    names = ['linf-mean', 'chi2-mean', 'chi2-min', 'chi2-wmean']
    for sig_s, sig_a, lab in [(0.10, 0.5e-6, "moderate noise (10%, 0.5 Mm-1)"),
                              (0.20, 1.0e-6, "tolerance-level noise (20%, 1 Mm-1)")]:
        errs = {n: [] for n in names}
        nsucc = {n: 0 for n in names}
        ntot = 0
        for sca_c, abs_c, ssa_c, _, _ in cases:
            for _rep in range(3):
                kt = rng.integers(len(gcri))     # truth ON grid: no discretization floor
                rri_t, iri_t = gcri[kt]
                sca_t, abs_t, ssa_t = sca_c[kt], abs_c[kt], ssa_c[kt]
                sca_m = sca_t * np.exp(rng.normal(0, sig_s, 3))
                abs_m = np.maximum(abs_t + rng.normal(0, sig_a, 3), 0)
                ntot += 1
                for n, r in estimators(gcri, sca_c, abs_c, ssa_c, sca_m, abs_m).items():
                    if r is None:
                        continue
                    nsucc[n] += 1
                    errs[n].append((r[0] - rri_t, r[1] - iri_t, r[2] - ssa_t))
        print(f"\n=== synthetic truth, {lab}, {ntot} cases ===")
        print(f"{'estimator':11s} {'succ%':>6} {'RRI bias':>9} {'RRI rmse':>9} "
              f"{'IRI bias':>9} {'IRI rmse':>9} {'SSA550 bias':>11} {'SSA550 rmse':>11}")
        for n in names:
            e = np.array(errs[n])
            print(f"{n:11s} {100*nsucc[n]/ntot:6.1f} "
                  f"{e[:,0].mean():+9.4f} {np.sqrt((e[:,0]**2).mean()):9.4f} "
                  f"{e[:,1].mean():+9.5f} {np.sqrt((e[:,1]**2).mean()):9.5f} "
                  f"{e[:,2].mean():+11.4f} {np.sqrt((e[:,2]**2).mean()):11.4f}")

    # ---------------- Part 2: real measured windows --------------------------
    res = {n: [] for n in names}
    for sca_c, abs_c, ssa_c, sca_m, abs_m in cases:
        if not (np.isfinite(sca_m).all() and np.isfinite(abs_m).all()
                and (sca_m > 0).all()):
            for n in names:
                res[n].append(None)
            continue
        r = estimators(gcri, sca_c, abs_c, ssa_c, sca_m, abs_m)
        for n in names:
            res[n].append(r[n])
    print(f"\n=== real windows ({len(cases)}) ===")
    base = res['linf-mean']
    for n in names:
        vals = res[n]
        succ = [v for v in vals if v is not None]
        drri = [v[0] - b[0] for v, b in zip(vals, base) if v and b]
        dssa = [v[2] - b[2] for v, b in zip(vals, base) if v and b]
        extra = ""
        if n != 'linf-mean' and drri:
            extra = (f"  vs current: dRRI med {np.median(drri):+.4f} "
                     f"p95|d| {np.percentile(np.abs(drri),95):.4f}, "
                     f"dSSA550 med {np.median(dssa):+.4f}")
        nu = [v[3] for v in succ]
        print(f"{n:11s} success {len(succ):3d}/{len(vals)}  "
              f"n_used med {int(np.median(nu)) if nu else 0:3d}{extra}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
