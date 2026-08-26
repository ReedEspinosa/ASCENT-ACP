"""Humidified particle size distributions on the dry bin grid.

A bulk kappa grows every particle by the same factor
``gf = (1 + kappa*RH/(100-RH))**(1/3)``, so the humidified PSD is the dry
PSD uniformly shifted by ``log10(gf)`` in log-diameter. To keep dry/wet/
ambient PSDs on one shared bin axis, the shifted distribution is remapped
back onto the dry bin edges.

The remap conserves SURFACE AREA exactly (surface is the quantity closest
to the optics at these size parameters): per-bin surface concentrations of
the grown distribution are redistributed to the dry bins in proportion to
log-diameter overlap, assuming uniform surface density in logD within each
source bin. Number and volume are consequently conserved only approximately
(no rebinning can conserve all three). Surface grown past the last bin edge
is returned separately as ``surface_beyond_grid`` instead of being clipped.
"""

import numpy as np


def growth_factor(kappa, rh):
    """kappa-Kohler diameter growth factor at ``rh`` percent (rh < 100)."""
    kappa = np.asarray(kappa, float)
    rh = np.asarray(rh, float)
    with np.errstate(invalid="ignore"):
        gf = (1.0 + kappa * rh / (100.0 - rh)) ** (1.0 / 3.0)
    return np.where(np.isfinite(gf) & (rh > 0) & (rh < 100), gf, np.nan)


def humidified_psd(dndlogdp, dpg_um, dpl_um, dpu_um, gf):
    """Remap a grown PSD back onto the dry bin grid, conserving surface.

    Parameters: per-bin dry ``dndlogdp`` (cm-3), bin center/lower/upper
    diameters (um), and a scalar growth factor. Returns
    ``(dndlogdp_hum, surface_beyond_grid)`` where ``dndlogdp_hum`` is the
    humidified dN/dlogDp expressed on the DRY bin centers and
    ``surface_beyond_grid`` (um2 cm-3) is the surface concentration grown
    past the last bin's upper edge.

    NaN handling: a target bin overlapped by any NaN source bin is NaN, and
    NaN source bins in the spillover range make ``surface_beyond_grid`` NaN.
    """
    n = np.asarray(dndlogdp, float)
    dpg = np.asarray(dpg_um, float)
    lo = np.log10(np.asarray(dpl_um, float))
    hi = np.log10(np.asarray(dpu_um, float))
    if not np.isfinite(gf) or gf <= 0:
        return np.full_like(n, np.nan), np.nan
    if gf == 1.0:
        return n.copy(), 0.0

    dlog = hi - lo
    # per-bin surface concentration (um2 cm-3) of the GROWN distribution:
    # number is preserved per particle, surface scales by gf^2
    s_src = np.pi * dpg**2 * n * dlog * gf**2
    src_lo = lo + np.log10(gf)
    src_hi = hi + np.log10(gf)

    s_out = np.zeros_like(n)
    nan_out = np.zeros(n.size, dtype=bool)
    spill = 0.0
    spill_nan = False
    grid_top = hi[-1]
    for i in range(n.size):
        seg = np.clip(np.minimum(src_hi[i], hi) - np.maximum(src_lo[i], lo), 0.0, None)
        frac = seg / (src_hi[i] - src_lo[i])
        beyond = max(src_hi[i] - max(src_lo[i], grid_top), 0.0) / (src_hi[i] - src_lo[i])
        if np.isnan(s_src[i]):
            nan_out |= frac > 0
            if beyond > 0:
                spill_nan = True
            continue
        s_out += frac * s_src[i]
        spill += beyond * s_src[i]

    dnd_out = s_out / (np.pi * dpg**2 * dlog)
    dnd_out[nan_out] = np.nan
    return dnd_out, (np.nan if spill_nan else spill)
