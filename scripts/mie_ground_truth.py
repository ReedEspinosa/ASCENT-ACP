"""Ground-truth benchmark: exact Mie vs MOPSMAP vs GRASP-kernel integration.

PSD: piecewise-linear dN/dlogDp between 51 log-spaced nodes (0.01-1.0 um),
lognormal shape, zero outside — the representation MOPSMAP itself assumes.
Reference: exact Mie at 20k quadrature points in logD, exact CRI (no interp).

Results 2026-08-31 (broad lognormal Dg=0.1um, ln-sigma=0.7; 450/550/700 nm;
CRI 1.52+0.006081i on-node, 1.54+0.005i and 1.50+0.010i off-node):
  MOPSMAP/exact:        1.001-1.002 everywhere (ext & sca; on- and off-node).
    NOTE: raw ext_coeff_* outputs here are m^-1 for number input read as m^-3;
    with dndlogdp supplied in cm^-3, multiply by 1e12 to get Mm^-1.
  GRASP-kernel/exact:   0.980-0.986 under NAIVE usage (stored values treated
    as point samples of the kernel, linear-log-x interp).
  RESOLVED same day: the stored values are EXACTLY (machine precision, ext &
    sca) the triangular-in-lnx basis average of the volume kernel
    Kv = 3*pi*Q/(2*x*lambda0) — zero intrinsic Mie error. The naive-usage
    deficit is pure representation error and is invertible: solve the
    tridiagonal mass matrix (g[i-1]+4g[i]+g[i+1])/6 = Kbar[i] (grid uniform
    in lnx, h=0.2716) to de-smooth, then interp/integrate as before.
    De-smoothed GRASP vs exact: worst case 0.18% over 3 PSD shapes x
    4 CRIs (incl. wet-side 1.40+0.002i, off-node) x 3 lambdas — no
    calibration factor needed; slightly better than MOPSMAP.
  lambda0 = 0.340 um verified EXACTLY: ext == 3*pi*qext/(2*x*0.340) over the
    whole grid. mr/mi interpolation error negligible (on-node == off-node).
  Node-quadrature usage (PSD sampled at x-nodes) is WORSE (-0.5 to -4%,
    lambda-dependent): integrate the (de-smoothed) kernel on the fine grid.
  Ripple-aliasing sweep (gf 1.00-1.38 in 33 sub-node steps, 550 nm): the
    de-smoothed error is phase-stable (p2p 0.19%/0.03%/0.61% for
    lnsig 0.70/0.35/0.20; worst point 0.34%); the naive smoothed-Kbar
    interpolant is what aliases (p2p up to 4.3% for narrow PSDs).
    De-smoothing = Galerkin/L2 projection: node values derive only from
    integrals (never point samples), residual orthogonal to piecewise-
    linears, mass-matrix inverse gain bounded by 3.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/wrespino/Synced/Local_Code_MacBook/ISARA_code")
import mopsmap_wrapper

GRASP_NC = ("/Users/wrespino/Synced/STG_AerosolModelExchange/GRASP-LUT-Export/"
            "GRASP-Kernels_netCDF-Versions/kernel-grasp-v1.1.3-integrated_V4.nc")
MOPS_EXE = "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/mopsmap"
MOPS_DAT = "/Users/wrespino/Synced/Resources/GeneralSoftware/MOPSMAP/mopsmap/optical_dataset/"


# ---------------- exact Mie (vectorized over x, fixed complex m) -------------
def mie_q(m, x):
    """Qext, Qsca for complex m, array x. Standard Bohren-Huffman recurrences."""
    x = np.atleast_1d(np.asarray(x, float))
    xmax = x.max()
    nmax = int(xmax + 4.05 * xmax ** (1 / 3) + 2) + 1
    nstart = nmax + 15
    mx = m * x
    D = np.zeros((nmax + 1, x.size), complex)
    d = np.zeros(x.size, complex)
    for n in range(nstart, 0, -1):
        d = n / mx - 1.0 / (d + n / mx)   # this is D_{n-1}
        if n - 1 <= nmax:
            D[n - 1] = d
    psi0, psi1 = np.cos(x), np.sin(x)          # psi_{-1}, psi_0
    chi0, chi1 = -np.sin(x), np.cos(x)         # chi_{-1}, chi_0
    qext = np.zeros(x.size)
    qsca = np.zeros(x.size)
    for n in range(1, nmax + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi, xi1 = psi - 1j * chi, psi1 - 1j * chi1
        t_a = D[n] / m + n / x
        t_b = D[n] * m + n / x
        a = (t_a * psi - psi1) / (t_a * xi - xi1)
        b = (t_b * psi - psi1) / (t_b * xi - xi1)
        qext += (2 * n + 1) * (a.real + b.real)
        qsca += (2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)
        psi0, psi1, chi0, chi1 = psi1, psi, chi1, chi
    return 2 / x ** 2 * qext, 2 / x ** 2 * qsca


# sanity: Rayleigh limit
m0 = 1.5 + 0.0j
_, qs = mie_q(m0, np.array([0.01]))
ray = 8 / 3 * 0.01 ** 4 * abs((m0 ** 2 - 1) / (m0 ** 2 + 2)) ** 2
assert abs(qs[0] / ray - 1) < 1e-3, (qs[0], ray)

# ---------------- PSD ---------------------------------------------------------
dpg = np.logspace(np.log10(0.01), np.log10(1.0), 51)          # um, nodes
sd = 1000 * np.exp(-0.5 * ((np.log(dpg) - np.log(0.1)) / 0.7) ** 2)  # cm-3

logd_f = np.linspace(np.log10(dpg[0]), np.log10(dpg[-1]), 20001)
d_f = 10 ** logd_f
n_f = np.interp(logd_f, np.log10(dpg), sd)                    # piecewise-linear

WVL = np.array([450., 550., 700.])                            # nm


def exact_coeffs(m):
    out = {}
    for w in WVL:
        x = np.pi * d_f / (w / 1000)
        qe, qs = mie_q(m, x)
        area = np.pi / 4 * d_f ** 2                            # um2
        # integral over logD; um2 * cm-3 = Mm-1
        out[w] = (np.trapz(qe * area * n_f, logd_f),
                  np.trapz(qs * area * n_f, logd_f))
    return out


# ---------------- MOPSMAP -----------------------------------------------------
def mops_coeffs(mr, mi):
    kw = dict(size_equ={'m': 'cs'}, dndlogdp={'m': sd}, dpg={'m': dpg},
              RRI={'m': mr}, IRI={'m': mi}, nonabs_fraction={'m': 0},
              shape={'m': 'sphere'}, density={'m': 1.0}, RH=0, kappa=0,
              num_theta=2, path_optical_dataset=MOPS_DAT,
              path_mopsmap_executable=MOPS_EXE)
    r = mopsmap_wrapper.Model(WVL, **kw)
    return r


# ---------------- GRASP kernels ----------------------------------------------
import xarray as xr
ds = xr.open_dataset(GRASP_NC)
XK = ds.x.values
MR = ds.mr.values
MI = -ds.mi.values                                            # stored negative
EXT = ds.ext.values[0]                                        # sphere
SCA = ds.sca.values[0]
LAM0 = 0.340                                                  # um


def interp_kernel(K, mr, mi, xq):
    """bilinear in (mr, log mi), linear in log x."""
    i = np.clip(np.searchsorted(MR, mr) - 1, 0, len(MR) - 2)
    fmr = (mr - MR[i]) / (MR[i + 1] - MR[i])
    lmi = np.log(max(mi, 1e-9))
    LMI = np.log(np.maximum(MI, 1e-10))
    j = np.clip(np.searchsorted(LMI, lmi) - 1, 0, len(MI) - 2)
    fmi = (lmi - LMI[j]) / (LMI[j + 1] - LMI[j])
    Kmm = ((1 - fmr) * (1 - fmi) * K[i, j] + fmr * (1 - fmi) * K[i + 1, j]
           + (1 - fmr) * fmi * K[i, j + 1] + fmr * fmi * K[i + 1, j + 1])
    lx = np.log(XK)
    return np.interp(np.log(xq), lx, Kmm)


def grasp_coeffs(mr, mi):
    out = {}
    vol_f = np.pi / 6 * d_f ** 3 * n_f                         # um3 cm-3 per logD
    for w in WVL:
        lam = w / 1000
        xq = np.pi * d_f / lam
        ke = interp_kernel(EXT, mr, mi, xq) * (LAM0 / lam)
        ks = interp_kernel(SCA, mr, mi, xq) * (LAM0 / lam)
        out[w] = (np.trapz(ke * vol_f, logd_f), np.trapz(ks * vol_f, logd_f))
    return out


# ---------------- run cases ---------------------------------------------------
CASES = [("A: MOPSMAP-node 1.5200+0.006081i", 1.52, 0.006081),
         ("B: off-node      1.5400+0.005000i", 1.54, 0.005),
         ("C: off-node      1.5000+0.010000i", 1.50, 0.010)]

for label, mr, mi in CASES:
    m = complex(mr, mi)
    ex = exact_coeffs(m)
    mo = mops_coeffs(mr, mi)
    gr = grasp_coeffs(mr, mi)
    print(f"\n=== {label} ===")
    print(f"{'wvl':>5} {'exact ext':>10} {'MOPS/exact':>11} {'GRASP/exact':>12}"
          f" {'exact sca':>10} {'MOPS/exact':>11} {'GRASP/exact':>12}")
    for k, w in enumerate(WVL):
        e_ext, e_sca = ex[w]
        mo_ext = float(mo[f'ext_coeff_{w}_m-1']) * 1e6
        mo_sca = mo_ext * float(mo[f'ssa_{w}'])
        g_ext, g_sca = gr[w]
        print(f"{w:5.0f} {e_ext:10.3f} {mo_ext/e_ext:11.4f} {g_ext/e_ext:12.4f}"
              f" {e_sca:10.3f} {mo_sca/e_sca:11.4f} {g_sca/e_sca:12.4f}")
