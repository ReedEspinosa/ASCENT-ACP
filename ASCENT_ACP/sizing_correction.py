"""Optical-sizer refractive-index sizing correction (LAS/UHSAS).

Optical particle sizers assign diameters through a calibration at a fixed
refractive index (AmmSO4 1.52 or PSL ~1.58); ambient particles of a
different RI are mis-sized in a correlated way (Moore et al. 2021, AMT).
Because each instrument bin is a fixed signal threshold, the correction is
a per-candidate-RI remapping of the bin diameters:

    R(D_true, m_candidate) = R(D_cal, m_cal)

with R(D, m) = qsca_partial(x = pi D / lambda; m) * D^2, the partial
scattering cross section into the collection solid angle (33-147 deg with
72.5-104.8 deg blocked -- the Moore et al. 2021 geometry for both LAS and
UHSAS, stored in mopsmap_spheres_v2.nc). Both response curves are made
monotonic with the log-space monotone envelope before threshold mapping
(conventions coincide below the ~0.4 um resonance region). Bin counts are
conserved: dN/dlogDp is rescaled by the bin-width Jacobian.

The mapping is precomputed once per run for every candidate on the CRI
grid; per-candidate application is a vector multiply. See
LAS_RI_CORRECTION_PLAN.md for the full design and validation gates.
"""

import os

import numpy as np
import netCDF4 as nc

_TAB = None
_TAB_PATH = None


def _load_table(isara_dir):
    global _TAB, _TAB_PATH
    path = os.path.join(isara_dir, "mopsmap_sphere_table",
                        "mopsmap_spheres_v2.nc")
    if _TAB is None or _TAB_PATH != path:
        with nc.Dataset(path) as d:
            _TAB = {
                "mr": d["mreal"][:].filled(np.nan),
                "logmi": np.log(np.maximum(d["mimag"][:].filled(np.nan), 1e-9)),
                "logsp": np.log(d["sizepara"][:].filled(np.nan)),
                "qp": d["qsca_partial"][:].filled(np.nan).astype(float),
            }
        _TAB_PATH = path
    return _TAB


def _qp_curve(t, mr, mi, logx):
    """qsca_partial at log size parameters (bilinear mr / log-mi)."""
    i = int(np.clip(np.searchsorted(t["mr"], mr) - 1, 0, len(t["mr"]) - 2))
    fr = (mr - t["mr"][i]) / (t["mr"][i + 1] - t["mr"][i])
    lmi = np.log(max(mi, 1e-9))
    j = int(np.clip(np.searchsorted(t["logmi"], lmi) - 1, 0,
                    len(t["logmi"]) - 2))
    fi = (lmi - t["logmi"][j]) / (t["logmi"][j + 1] - t["logmi"][j])
    q = ((1 - fr) * (1 - fi) * t["qp"][i, j] + fr * (1 - fi) * t["qp"][i + 1, j]
         + (1 - fr) * fi * t["qp"][i, j + 1] + fr * fi * t["qp"][i + 1, j + 1])
    good = np.isfinite(q)
    return np.interp(logx, t["logsp"][good], q[good])


def _monotone_response(t, mr, mi, lnD, lam_um):
    logx = np.log(np.pi) + lnD - np.log(lam_um)
    q = _qp_curve(t, mr, mi, logx)
    r = np.log(np.maximum(q, 1e-300)) + 2.0 * lnD   # ln(qp * D^2)
    r = np.maximum.accumulate(r)
    # strictly increasing (tiny ramp): keeps the forward/inverse threshold
    # mapping bijective across resonance plateaus, so candidate == cal RI
    # gives the exact identity and plateau regions map one-to-one
    return r + np.linspace(0.0, 1e-6, r.size)


def build_state(isara_dir, dpl_um, dpg_um, dpu_um, optical_mask, lambda_nm,
                cal_ri, cri_grid):
    """Precompute the per-candidate bin mapping.

    Returns a plain-dict state (picklable): 'mask' over the grid bins,
    'shift' (n_candidates x n_optical_bins) lnD center shifts, and 'wr'
    (same shape) bin-width ratios dlnD_true/dlnD_cal. Identity for
    candidates equal to the calibration RI by construction.
    """
    t = _load_table(isara_dir)
    lam = lambda_nm / 1000.0
    mask = np.asarray(optical_mask, bool)
    lnc = np.log(np.asarray(dpg_um, float)[mask])
    lnl = np.log(np.asarray(dpl_um, float)[mask])
    lnu = np.log(np.asarray(dpu_um, float)[mask])
    lnD = np.linspace(min(lnl.min(), np.log(0.04)),
                      max(lnu.max() + 0.7, np.log(3.0)), 1200)
    r_cal = _monotone_response(t, float(cal_ri), 0.0, lnD, lam)
    s_c = np.interp(lnc, lnD, r_cal)
    s_l = np.interp(lnl, lnD, r_cal)
    s_u = np.interp(lnu, lnD, r_cal)
    n_cand = len(cri_grid)
    n_bins = len(mask)
    # full-width arrays (identity outside the optical instrument's bins) so
    # downstream code can slice them exactly like the bin arrays themselves
    shift = np.zeros((n_cand, n_bins))
    wr = np.ones((n_cand, n_bins))
    for k, (rri, iri) in enumerate(np.asarray(cri_grid, float)):
        r_amb = _monotone_response(t, rri, iri, lnD, lam)
        shift[k, mask] = np.interp(s_c, r_amb, lnD) - lnc
        l_t = np.interp(s_l, r_amb, lnD)
        u_t = np.interp(s_u, r_amb, lnD)
        wr[k, mask] = np.clip((u_t - l_t) / (lnu - lnl), 0.2, 5.0)
    return {"shift": shift, "wr": wr,
            "cri_grid": np.asarray(cri_grid, float),
            "lambda_nm": float(lambda_nm), "cal_ri": float(cal_ri)}


def nearest_candidate(state, rri, iri):
    g = state["cri_grid"]
    return int(np.argmin((g[:, 0] - rri) ** 2
                         + ((g[:, 1] - iri) / 0.002) ** 2 * 1e-6))


def apply(state, k, dpg_um, dndlogdp, cols=None):
    """Corrected (dpg, dndlogdp) for candidate index k (counts conserved).

    ``cols`` optionally selects/reorders bins (e.g. the surviving-bin index
    after NaN dropping) so the state rows align with filtered arrays.
    """
    sh = state["shift"][k] if cols is None else state["shift"][k][cols]
    w = state["wr"][k] if cols is None else state["wr"][k][cols]
    return (np.asarray(dpg_um, float) * np.exp(sh),
            np.asarray(dndlogdp, float) / w)
