"""Numerical test of the 'ratio' kappa objective against the real ISARA code.

Injects a fake forward model (scattering proportional to total particle
volume, so enhancement = gf^3 = 1 + kappa*RH/(100-RH) exactly) into
ISARA.Retr_kappa. With a PSD whose amplitude is biased low by 2x:

* 'absolute' objective: kappa must double the scattering AND supply the
  measured enhancement -> inflated kappa (the SEAC4RS LAS failure mode).
* 'ratio' objective (dry_closure passed): the amplitude bias cancels and
  the true kappa is recovered.

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


def volume_model(wvl, size_equ, sd, dpg, RRI, IRI, nonabs, shape, rho,
                 _a, _b, num_theta, path1, path2):
    """sca = 1e-9 * total volume; SSA = 1 (RI-independent by design)."""
    mode = next(iter(sd))
    vol = float(np.sum(sd[mode] * np.asarray(dpg[mode]) ** 3))
    out = {}
    for w in np.atleast_1d(wvl):
        out[f"ext_coeff_{int(w)}_m-1"] = 1e-9 * vol
        out[f"ssa_{int(w)}"] = 1.0
    return out


def run_kappa(isara, dry_closure, kappa_true=0.30, amp_bias=0.5, rh=80.0):
    """Retr_kappa on a 2x-undercounted PSD; the truth has kappa_true."""
    dpg_true = np.array([0.1, 0.2, 0.4])
    sd_true = np.array([2.0e8, 1.0e8, 1.0e7])  # m^-3
    gf3 = 1.0 + kappa_true * rh / (100.0 - rh)
    dry_true = 1e-9 * float(np.sum(sd_true * dpg_true ** 3))
    wet_true = dry_true * gf3          # volume model: enhancement = gf^3
    sd_biased = {"PSD": sd_true * amp_bias}
    dpg_d = {"PSD": dpg_true}
    meas = {"wet_meas_sca_coef_550_m-1": wet_true}
    return isara.Retr_kappa(
        {"sca": np.array([550])}, None, meas, sd_biased, dpg_d, rh,
        np.arange(0.0, 1.4, 0.001), np.array([1.5, 0.001]),
        {"PSD": "cs"}, {"PSD": 0}, {"PSD": "sphere"}, {"PSD": 1.0}, 2,
        ".", ".", model=volume_model, estimator="chi2-wmean",
        wet_sigma=np.array([0.001 * wet_true]), dry_closure=dry_closure)


def test_absolute_objective_absorbs_amplitude_bias(isara):
    res = run_kappa(isara, dry_closure=None)
    # amplitude bias 0.5 -> needed gf^3 = 2*(1+4*0.3) = 4.4 -> kappa = 0.85
    assert res["kappa_unitless"] == pytest.approx(0.85, abs=0.02)


def test_ratio_objective_recovers_true_kappa(isara):
    res = run_kappa(isara, dry_closure=np.array([0.5]))
    assert res["kappa_unitless"] == pytest.approx(0.30, abs=0.01)


def test_nonfinite_closure_falls_back_to_absolute(isara):
    res = run_kappa(isara, dry_closure=np.array([np.nan]))
    assert res["kappa_unitless"] == pytest.approx(0.85, abs=0.02)


def test_retr_psd_rejects_bad_kappa_fit(isara):
    with pytest.raises(ValueError, match="kappa_fit"):
        isara.Retr_PSD(np.array([0.05, 0.1]), np.array([100.0, 50.0]),
                       np.array([1e-5]), np.array([1e-6]),
                       {"sca": [550], "abs": [532]}, kappa_fit="bogus")
