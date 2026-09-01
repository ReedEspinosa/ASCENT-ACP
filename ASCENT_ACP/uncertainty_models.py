"""Instrument 1-sigma error models for the in-situ channels.

Port of the reference implementation in
``ACMAP_Meloe/ISARA/aerosol_insitu_uncertainty_models.md`` (Sect. 5; "UM"),
three-term form  sigma^2 = (f_rel*y)^2 + a^2*(t_ref/t) + b^2.
All coefficients 1-sigma; optical inputs/outputs in Mm^-1.

Rule (Reed): wherever a model needs an optical value (b_sp for the PSAP
scattering-subtraction term, SSA sanity forms), feed the MEASURED window
mean, never a retrieved/calculated value.
"""

import numpy as np

# ------------------------------------------------------------- nephelometer
NEPH_A = {450: 0.15, 550: 0.10, 700: 0.30}    # Mm^-1 at t_ref = 30 s
NEPH_T_REF = 30.0                              # s
NEPH_ZERO_DUR = 300.0                          # s, zero-cycle averaging period

NEPH_FREL = {
    "pm1":           0.08,
    "pm10":          0.09,
    "pm1_absorbing": 0.10,
    "coarse_dust":   0.20,
}


def sigma_scattering(b_sp, t, wavelength=550, regime="pm1",
                     zero_duration=NEPH_ZERO_DUR):
    """1-sigma on TSI 3563 scattering coefficient [Mm^-1] (UM Sect. 1)."""
    a = NEPH_A[int(wavelength)]
    f = NEPH_FREL[regime]
    white = a ** 2 * (NEPH_T_REF / t)
    floor = (a * np.sqrt(NEPH_T_REF / zero_duration)) ** 2  # zero-subtraction
    return np.sqrt((f * np.asarray(b_sp, float)) ** 2 + white + floor)


# --------------------------------------------------------------------- PSAP
PSAP_FREL = 0.12
PSAP_FSCA_ERR = 0.016      # Bond et al. (1999): K1/K2 uncertainty = 100%
PSAP_A = 0.05              # Mm^-1 at t_ref = 3600 s (PSAP; CLAP is 0.02)
PSAP_T_REF = 3600.0        # s
PSAP_FLOOR = 0.015         # Mm^-1, non-averaging offset
PSAP_T_INTERNAL = 60.0     # s, firmware smoothing window


def sigma_absorption(b_ap, b_sp, t, t_internal=PSAP_T_INTERNAL):
    """1-sigma on PSAP absorption coefficient [Mm^-1] (UM Sect. 2).

    ``b_sp`` is the co-located MEASURED scattering coefficient: the dominant
    high-SSA term scales with it, not with ``b_ap``.
    """
    t_eff = max(t, t_internal)   # 1 Hz samples are not independent below this
    white = PSAP_A ** 2 * (PSAP_T_REF / t_eff)
    return np.sqrt(
        (PSAP_FREL * np.asarray(b_ap, float)) ** 2
        + (PSAP_FSCA_ERR * np.asarray(b_sp, float)) ** 2
        + white
        + PSAP_FLOOR ** 2
    )


def sigma_absorption_from_ssa(omega):
    """Relative 1-sigma vs SSA, noise-free limit (UM sanity table)."""
    omega = np.asarray(omega, float)
    return np.sqrt(PSAP_FREL ** 2
                   + (PSAP_FSCA_ERR * omega / (1.0 - omega)) ** 2)


# ---------------------------------------------------------------- LAS/UHSAS
OPC_FREL = 0.10        # flow, coincidence, inlet transmission
OPC_FREL_EDGE = 0.30   # lowest 2-3 bins (counting-efficiency roll-off)
OPC_DLND = 0.10        # 1-sigma on ln(D), FULLY CORRELATED across bins


def sigma_number(N, t, Q=1.0, density_ratio=1.0, edge_bins=0):
    """1-sigma per-bin number concentration, diagonal part (UM Sect. 3a).

    The correlated diameter-scale error (OPC_DLND) is deliberately NOT
    here — it is a single nuisance parameter, not per-bin noise.
    """
    N = np.asarray(N, dtype=float)
    f = np.full(N.shape, OPC_FREL)
    if edge_bins:
        f[:edge_bins] = OPC_FREL_EDGE
    poisson = N / (Q * t * density_ratio)   # variance
    return np.sqrt(poisson + (f * N) ** 2)
