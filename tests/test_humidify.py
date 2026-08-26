"""Surface-conserving humidified-PSD remap."""

import numpy as np
import pytest

from ASCENT_ACP import humidify


def make_grid(nbins=30, lo=0.05, hi=1.0):
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    dpl, dpu = edges[:-1], edges[1:]
    dpg = np.sqrt(dpl * dpu)
    dnd = 3000 * np.exp(-0.5 * (np.log(dpg / 0.15) / np.log(1.6)) ** 2)
    return dnd, dpg, dpl, dpu


def total_surface(dnd, dpg, dpl, dpu):
    return np.nansum(np.pi * dpg**2 * dnd * np.log10(dpu / dpl))


def test_growth_factor():
    assert humidify.growth_factor(0.0, 80.0) == pytest.approx(1.0)
    assert humidify.growth_factor(0.3, 80.0) == pytest.approx((1 + 0.3 * 4) ** (1 / 3))
    assert np.isnan(humidify.growth_factor(0.3, 100.0))
    assert np.isnan(humidify.growth_factor(np.nan, 80.0))


def test_identity_at_gf1():
    dnd, dpg, dpl, dpu = make_grid()
    out, spill = humidify.humidified_psd(dnd, dpg, dpl, dpu, 1.0)
    assert np.array_equal(out, dnd)
    assert spill == 0.0


def test_surface_conserved_including_spillover():
    dnd, dpg, dpl, dpu = make_grid()
    for gf in (1.1, 1.4, 2.0):
        out, spill = humidify.humidified_psd(dnd, dpg, dpl, dpu, gf)
        s_in = total_surface(dnd, dpg, dpl, dpu) * gf**2  # grown surface
        s_out = total_surface(out, dpg, dpl, dpu) + spill
        assert s_out == pytest.approx(s_in, rel=1e-12)


def test_distribution_shifts_upward():
    dnd, dpg, dpl, dpu = make_grid()
    out, _ = humidify.humidified_psd(dnd, dpg, dpl, dpu, 1.5)
    # mode diameter of the remapped distribution exceeds the dry mode
    assert dpg[np.nanargmax(out)] > dpg[np.argmax(dnd)]


def test_spillover_grows_with_gf():
    dnd, dpg, dpl, dpu = make_grid()
    spills = [humidify.humidified_psd(dnd, dpg, dpl, dpu, gf)[1]
              for gf in (1.0, 1.3, 1.8, 2.5)]
    assert all(b >= a for a, b in zip(spills, spills[1:]))
    assert spills[-1] > 0


def test_nan_source_bins_propagate():
    dnd, dpg, dpl, dpu = make_grid()
    dnd[10] = np.nan
    out, spill = humidify.humidified_psd(dnd, dpg, dpl, dpu, 1.3)
    # bins overlapped by the SHIFTED NaN bin are NaN; far-away bins are not
    shift_bins = np.log10(1.3) / np.log10(dpu[0] / dpl[0])  # ~2.6 bins
    hit = 10 + int(np.floor(shift_bins))
    assert np.isnan(out[hit]) or np.isnan(out[hit + 1])
    assert np.isfinite(out[0])
    # NaN in the top bins poisons the spillover estimate
    dnd2, *_ = make_grid()
    dnd2[-1] = np.nan
    _, spill2 = humidify.humidified_psd(dnd2, dpg, dpl, dpu, 1.8)
    assert np.isnan(spill2)


def test_bad_gf_returns_nan():
    dnd, dpg, dpl, dpu = make_grid()
    out, spill = humidify.humidified_psd(dnd, dpg, dpl, dpu, np.nan)
    assert np.isnan(out).all() and np.isnan(spill)
