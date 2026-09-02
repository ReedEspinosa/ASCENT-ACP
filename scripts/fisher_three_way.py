#!/usr/bin/env python
"""Three-way sigma_RRI Fisher comparison: original ISARA / current V10 /
current + 3-lambda TSI backscatter fraction.

Linear-Gaussian marginal: sigma_RRI = 1/sqrt(g' S^-1 g), with
S = diag(meas noise^2) + sum_i d_i d_i' (nuisance outer products, d_i = 1-sigma
fractional response). Jacobians from the committed fingerprint tables
(RRI_INFORMATION_CONTENT.md; median retrieved ACTIVATE window):
  sca:  RRI half-grid (0.045) -> [11.2, 10.9, 12.2]%   sizing +5% lnD -> ~[20.3,20.4,21.3]%
  b:    RRI +0.04 -> [4.2, 5.4, 4.4]%                  sizing +5% lnD -> [-3.7,-0.7,-0.2]%
Concentration and neph calibration are exactly flat on sca and cancel in b.
"""
import numpy as np

# Jacobians per 1.0 of RRI, fractional units
g_sca = np.array([11.2, 10.9, 12.2]) / 100 / 0.045
g_b   = np.array([4.2, 5.4, 4.4]) / 100 / 0.04

# 1-sigma nuisance responses (fractional) on [sca(3), b(3)]
def d(sca, b=(0, 0, 0)):
    return np.concatenate([np.asarray(sca, float), np.asarray(b, float)])

siz_sca = np.array([20.3, 20.4, 21.3]) / 100      # per 5% lnD
siz_b   = np.array([-3.7, -0.7, -0.2]) / 100

def sigma_rri(channels, noise, nuisances, g_full):
    """channels: boolean mask over the 6 slots."""
    m = np.asarray(channels, bool)
    g = g_full[m]
    S = np.diag(np.asarray(noise, float)[m] ** 2)
    for dv in nuisances:
        dv = dv[m]
        S = S + np.outer(dv, dv)
    return 1.0 / np.sqrt(g @ np.linalg.solve(S, g))

g_full = np.concatenate([g_sca, g_b])
sca_only = [1, 1, 1, 0, 0, 0]
sca_b    = [1, 1, 1, 1, 1, 1]

meas = lambda b_noise: [0.02, 0.02, 0.02, b_noise, b_noise, b_noise]

nu_current = lambda f_lnd: [
    d(siz_sca * f_lnd, siz_b * f_lnd),        # sizing residual (f_lnd scales 5%)
    d([0.10, 0.10, 0.10]),                    # concentration scale
    d([0.08, 0.08, 0.08]),                    # neph calibration common
    d([0, 0, 0], [0.02, 0.02, 0.02]),         # residual b common (shutter/cal)
]

print("=== per-window sigma_RRI (linear Fisher, median ACTIVATE window) ===\n")

# --- 1. Original ISARA: diagonal errors, no common modes, no sizing term ---
nu_none = []
noise_orig = [0.054, 0.054, 0.054, 1, 1, 1]   # 2% white (+) 5% independent per channel
s1 = sigma_rri(sca_only, noise_orig, nu_none, g_full)
print(f"1. ORIGINAL ISARA (diagonal 5%/channel, nuisances ignored)")
print(f"   apparent (reported) precision:          {s1:.4f}")
# unmodeled terms become bias: RRI-equivalents via response ratios
r = np.mean(siz_sca / 0.05) / np.mean(g_sca)  # RRI per unit lnD
b_act = np.sqrt((r * 0.03) ** 2 + 0.042 ** 2 + 0.034 ** 2)
b_sea = np.sqrt((r * 0.125) ** 2 + 0.042 ** 2 + 0.034 ** 2)
print(f"   hidden systematic (cal-RI ~3% lnD, ACTIVATE-like): ~{b_act:.3f}")
print(f"   hidden systematic (LAS 12.5% lnD, SEAC4RS):        ~{b_sea:.3f}\n")

# --- 2. Current V10: full covariance, sizing-corrected ---
for tag, f in [("ACTIVATE (resid lnD 5%)", 1.0), ("SEAC4RS (resid lnD 10%)", 2.0)]:
    s2 = sigma_rri(sca_only, meas(1), nu_current(f), g_full)
    print(f"2. CURRENT V10, {tag}:  sigma_RRI = {s2:.4f}")
print()

# --- 3. Current + 3-lambda backscatter fraction ---
for bn in (0.03, 0.04, 0.05):
    for tag, f in [("ACTIVATE", 1.0), ("SEAC4RS", 2.0)]:
        s3 = sigma_rri(sca_b, meas(bn), nu_current(f), g_full)
        print(f"3. +Bs (b noise {bn*100:.0f}%), {tag}:  sigma_RRI = {s3:.4f}")
print()

# decomposition: what does b alone give (sca channels removed)?
b_only = [0, 0, 0, 1, 1, 1]
for bn in (0.03, 0.04, 0.05):
    s = sigma_rri(b_only, meas(bn), nu_current(1.0), g_full)
    print(f"   b channels alone (noise {bn*100:.0f}%): sigma_RRI = {s:.4f}")
