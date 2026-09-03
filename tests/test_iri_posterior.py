"""Numerical test of the IRI posterior fix (quadrature grid weights + median).

Injects a forward model with scattering independent of CRI and absorption
LINEAR in IRI, so the IRI likelihood is an analytic Gaussian and the correct
posterior is known. Verifies:

* boundary case (true IRI between the fine near-zero grid points, sigma
  spanning the IRI >= 0 boundary): the retrieved IRI stays near the truth
  instead of being dragged toward the quasi-zero density spike, and the
  forward absorption closes;
* interior case: median == mean == truth;
* the RRI posterior stays at the grid (prior) mean when the likelihood is
  flat in RRI.

Skipped when the sibling ISARA_code checkout is not present.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ISARA_DIR = Path(__file__).resolve().parents[2] / "ISARA_code"
pytestmark = pytest.mark.skipif(not (_ISARA_DIR / "ISARA.py").exists(),
                                reason="ISARA_code checkout not found")


@pytest.fixture(scope="module")
def isara():
    sys.path.insert(0, str(_ISARA_DIR))
    import ISARA  # noqa: PLC0415
    yield ISARA
    sys.path.remove(str(_ISARA_DIR))


def absorbing_model(wvl, size_equ, sd, dpg, RRI, IRI, nonabs, shape, rho,
                    _a, _b, num_theta, path1, path2):
    """sca = 1e-9 * volume (CRI-independent); abs = sca * 10 * IRI."""
    mode = next(iter(sd))
    vol = float(np.sum(sd[mode] * np.asarray(dpg[mode]) ** 3))
    sca = 1e-9 * vol
    ab = sca * 10.0 * float(IRI[mode])
    out = {}
    for w in np.atleast_1d(wvl):
        out[f"ext_coeff_{int(w)}_m-1"] = sca + ab
        out[f"ssa_{int(w)}"] = sca / (sca + ab)
    return out


def run_cri(isara, iri_true, sigma_iri):
    dpg = np.array([0.1, 0.2, 0.4])
    sd = np.array([2.0e8, 1.0e8, 1.0e7])  # m^-3
    sca = 1e-9 * float(np.sum(sd * dpg ** 3))
    meas = {"dry_meas_sca_coef_550_m-1": sca,
            "dry_meas_abs_coef_532_m-1": sca * 10.0 * iri_true}
    grid = isara.default_CRI_grid(1.47, 1.56, 0.01)
    return isara.Retr_CRI(
        {"sca": np.array([550]), "abs": np.array([532])}, None, meas,
        {"PSD": sd}, {"PSD": dpg}, grid,
        {"PSD": "cs"}, {"PSD": 0}, {"PSD": "sphere"}, {"PSD": 1.0}, 2,
        ".", ".", model=absorbing_model, estimator="chi2-wmean",
        sca_sigma=np.array([0.01 * sca]),
        abs_sigma=np.array([sca * 10.0 * sigma_iri]))


def test_boundary_iri_not_dragged_to_zero_spike(isara):
    # truth 4e-4 with sigma 5e-4: posterior presses on IRI >= 0; the weighted
    # median of the truncated Gaussian is ~5.4e-4 (grid-snapped)
    res = run_cri(isara, iri_true=4e-4, sigma_iri=5e-4)
    iri = res["dry_IRI_unitless"]
    assert 2.5e-4 <= iri <= 8e-4
    # forward absorption at the reported IRI closes to better than 2x
    # (the old density-spike-pulled mean landed several-fold low here)
    assert 0.5 <= iri / 4e-4 <= 2.0


def test_interior_iri_recovered(isara):
    res = run_cri(isara, iri_true=5e-3, sigma_iri=5e-4)
    assert res["dry_IRI_unitless"] == pytest.approx(5e-3, abs=1e-3)


def test_iri_median_is_continuous_not_grid_snapped(isara):
    # the continuous (interpolated) median must track sub-grid-step changes
    # in the truth and must not return grid nodes for off-node truths
    grid = np.hstack((0, 1e-7, 1e-6, 1e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4,
                      np.arange(0.001, 0.031, 0.001)))
    outs = [run_cri(isara, iri_true=t, sigma_iri=5e-4)["dry_IRI_unitless"]
            for t in (4.3e-3, 4.6e-3)]
    assert outs[0] != outs[1]                      # sub-step sensitivity
    for o in outs:
        assert not np.isclose(grid, o, rtol=0, atol=1e-9).any()  # off-node


def test_flat_rri_posterior_stays_at_prior_mean(isara):
    # scattering is CRI-independent in this model -> RRI likelihood flat ->
    # posterior mean = uniform-grid prior mean (1.515), std ~ prior std
    res = run_cri(isara, iri_true=5e-3, sigma_iri=5e-4)
    assert res["dry_RRI_unitless"] == pytest.approx(1.515, abs=0.005)
    assert res["dry_RRI_accepted_std_unitless"] == pytest.approx(0.028, abs=0.004)
