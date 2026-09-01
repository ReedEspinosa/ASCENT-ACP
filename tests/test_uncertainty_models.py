"""Instrument sigma models vs the published sanity anchors (UM doc)."""

import numpy as np
import pytest

from ASCENT_ACP import uncertainty_models as um


def test_psap_ssa_table():
    # UM Sect. 2: ~12% at omega=0.5, ~33% at 0.95, ~160% at 0.99
    assert um.sigma_absorption_from_ssa(0.50) == pytest.approx(0.12, abs=0.01)
    assert um.sigma_absorption_from_ssa(0.95) == pytest.approx(0.33, abs=0.02)
    assert um.sigma_absorption_from_ssa(0.99) == pytest.approx(1.60, abs=0.10)


def test_neph_floor_binds_at_long_t():
    # at t -> inf the zero-cycle floor a/3.16 remains
    s = um.sigma_scattering(0.0, t=1e9, wavelength=550)
    assert s == pytest.approx(0.10 / np.sqrt(10.0), rel=1e-3)


def test_neph_relative_term_dominates_at_high_signal():
    s = um.sigma_scattering(100.0, t=60, wavelength=550, regime="pm1")
    assert s == pytest.approx(0.08 * 100.0, rel=0.01)


def test_psap_sca_term_uses_measured_scattering():
    lo = um.sigma_absorption(1.0, b_sp=0.0, t=60)
    hi = um.sigma_absorption(1.0, b_sp=100.0, t=60)
    assert hi ** 2 - lo ** 2 == pytest.approx((0.016 * 100.0) ** 2, rel=1e-6)


def test_psap_noise_frozen_below_internal_window():
    assert um.sigma_absorption(0, 0, t=1) == um.sigma_absorption(0, 0, t=60)
    assert um.sigma_absorption(0, 0, t=600) < um.sigma_absorption(0, 0, t=60)


def test_opc_poisson_floor_and_edge_inflation():
    N = np.array([100.0, 100.0, 100.0])
    s = um.sigma_number(N, t=60, Q=1.0, edge_bins=1)
    assert s[0] > s[1]                       # edge bin inflated
    assert s[1] == pytest.approx(np.sqrt(100 / 60 + (0.10 * 100) ** 2))
    # lower air density aloft degrades counting statistics
    s_alt = um.sigma_number(N, t=60, Q=1.0, density_ratio=0.5)
    assert (s_alt[1:] > s[1:]).all()
