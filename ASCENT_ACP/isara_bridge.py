"""Bridge between windowed ASCENT-ACP data and the ISARA retrieval library.

ISARA (sibling repo, not pip-installed) is imported by path from
``cfg.paths.isara_code_dir``. Each surviving window becomes one call to
``ISARA.Retr_PSD`` (a MOPSMAP grid search over refractive index and kappa);
windows are distributed over a process pool.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import sizing_correction as szc
from . import uncertainty_models as um
from .windows import psd_col_name

# Set per-process by _worker_init (workers) or import_isara (serial path)
_ISARA = None
_LUTS = {}  # bin-pattern key -> optics_lut.OpticsLUT
_SIZING = None  # sizing_correction state dict (or None)


def build_sizing_state(grid, cfg):
    """Per-run optical-sizer RI correction state (None when disabled)."""
    if not cfg.isara.sizing_correction:
        return None
    mask = grid.instrument == cfg.psd.optical_instrument_tag
    return szc.build_state(cfg.paths.isara_code_dir, grid.dpl_um, grid.dpg_um,
                           grid.dpu_um, mask, cfg.psd.optical_lambda_nm,
                           cfg.psd.optical_cal_ri, _cri_grid(cfg))


def import_isara(isara_code_dir):
    """Import the ISARA module from a sibling checkout; returns the module."""
    global _ISARA
    if isara_code_dir not in sys.path:
        sys.path.insert(0, isara_code_dir)
    import ISARA  # noqa: PLC0415

    _ISARA = ISARA
    return ISARA


def _cri_grid(cfg):
    """Candidate (RRI, IRI) grid from the config's RRI bounds."""
    if _ISARA is None:
        import_isara(cfg.paths.isara_code_dir)
    return _ISARA.default_CRI_grid(cfg.isara.rri_min, cfg.isara.rri_max,
                                   cfg.isara.rri_step)


def _union_wavelengths(ch):
    """Replicate Retr_CRI's wavelength layout: interleave sca/abs then sort."""
    pairs = []
    for s, a in zip(ch.dry_wvl_sca, ch.dry_wvl_abs):
        pairs += [s, a]
    return np.sort(np.array(pairs, float))


def _pattern_key(dndlogdp):
    """Hashable key for which PSD bins are finite in a window."""
    return tuple(np.nonzero(np.isfinite(np.asarray(dndlogdp, float)))[0])


def prepare_luts(good_windows, grid, cfg, verbose=True):
    """Build (or load cached) optics LUTs for the common PSD bin patterns.

    Retr_PSD drops NaN bins, so each distinct missing-bin pattern is a
    distinct bin grid needing its own LUT. Patterns rarer than
    ``lut_min_pattern_count`` are skipped; those windows transparently use
    the per-candidate MOPSMAP path inside ISARA.
    Returns {pattern_key: OpticsLUT-state-dict} (plain arrays, cheap to ship
    to worker processes).
    """
    import_isara(cfg.paths.isara_code_dir)
    import ISARA  # noqa: PLC0415
    import optics_lut  # noqa: PLC0415

    counts = {}
    for _, row in good_windows.iterrows():
        dnd = [row[psd_col_name(d)] for d in grid.dpg_um]
        counts[_pattern_key(dnd)] = counts.get(_pattern_key(dnd), 0) + 1

    cri_grid = _cri_grid(cfg)

    cache_dir = cfg.paths.lut_cache_dir or os.path.join(
        cfg.paths.output_dir, "lut_cache"
    )
    wvl = _union_wavelengths(cfg.channels)
    luts = {}
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n < cfg.isara.lut_min_pattern_count or len(key) < 2:
            continue
        if verbose:
            print(f"      LUT for bin pattern with {len(key)} bins ({n} windows)")
        lut = optics_lut.build(
            wvl,
            cri_grid,
            grid.dpg_um[list(key)],
            cfg.paths.optical_dataset_dir,
            cfg.paths.mopsmap_executable,
            size_equ=cfg.isara.size_equ,
            nonabs_fraction=cfg.isara.nonabs_fraction,
            shape=cfg.isara.shape,
            rho=cfg.isara.rho_dry,
            num_theta=cfg.isara.num_theta,
            n_workers=cfg.isara.n_workers,
            scratch_dir=cfg.paths.scratch_dir,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        luts[key] = {
            "wvl_nm": lut.wvl_nm,
            "cri_grid": lut.cri_grid,
            "dpg_um": lut.dpg_um,
            "K_ext": lut.K_ext,
            "K_sca": lut.K_sca,
            "size_equ": lut.size_equ,
            "nonabs_fraction": lut.nonabs_fraction,
            "shape": lut.shape,
            "rho": lut.rho,
            "num_theta": lut.num_theta,
        }
    return luts


def _install_luts(lut_states):
    """Reconstruct OpticsLUT objects in this process (after ISARA import)."""
    global _LUTS
    import optics_lut  # noqa: PLC0415

    _LUTS = {k: optics_lut.OpticsLUT(**v) for k, v in lut_states.items()}


def build_retr_kwargs(row, grid, cfg):
    """Translate one windows-DataFrame row into Retr_PSD keyword arguments.

    Scattering/absorption means are converted Mm^-1 -> m^-1; bin centers
    become radii. NaN PSD bins are passed through (Retr_PSD drops them).
    """
    ch = cfg.channels
    dndlogdp = np.array([row[psd_col_name(d)] for d in grid.dpg_um], dtype=float)
    if grid.penetration is not None:
        # weight the RETRIEVAL input by the impactor penetration so the
        # forward model sees what the (impactored) nephelometer saw; the
        # reported windowed PSD stays unweighted
        dndlogdp = dndlogdp * grid.penetration
    kwargs = {
        "radii_um": grid.dpg_um / 2.0,
        "dndlogdp_cm3": dndlogdp,
        "dry_sca_coef": np.array(
            [row[f"Sc{w}_dry_mean"] for w in ch.dry_wvl_sca], dtype=float
        )
        * 1e-6,
        "dry_abs_coef": np.array(
            [row[f"Abs{w}_mean"] for w in ch.dry_wvl_abs], dtype=float
        )
        * 1e-6,
        "dry_wvl": {"sca": list(ch.dry_wvl_sca), "abs": list(ch.dry_wvl_abs)},
        "wet_sca_coef": np.array(
            [row[f"Sc{w}_wet_mean"] for w in ch.wet_wvl_sca], dtype=float
        )
        * 1e-6,
        "wet_wvl": {"sca": list(ch.wet_wvl_sca)},
        "CRI_p": _cri_grid(cfg),
        "RH_wet": cfg.filters.wet_rh,
        # window-mean ambient RH (NaN -> no ambient forward state)
        "RH_ambient": float(row["RH_amb_mean"]) if "RH_amb_mean" in row else None,
        "val_wvl": np.array(ch.val_wvl) if ch.val_wvl else None,
        # report every state at the full wavelength union
        "out_wvl": np.array(sorted({*ch.dry_wvl_sca, *ch.dry_wvl_abs,
                                    *ch.wet_wvl_sca, *(ch.val_wvl or [])})),
        "size_equ": cfg.isara.size_equ,
        "nonabs_fraction": cfg.isara.nonabs_fraction,
        "shape": cfg.isara.shape,
        "rho_dry": cfg.isara.rho_dry,
        "rho_wet": cfg.isara.rho_wet,
        "num_theta": cfg.isara.num_theta,
        "path_optical_dataset": cfg.paths.optical_dataset_dir,
        "path_mopsmap_executable": cfg.paths.mopsmap_executable,
        "forward_engine": cfg.isara.forward_engine,
        "estimator": cfg.isara.estimator,
        "kappa_fit": cfg.isara.kappa_objective,
        # slightly-negative floor (see IsaraConfig.kappa_min); same upper
        # edge/step as ISARA.default_kappa_grid
        "kappa_p": np.arange(cfg.isara.kappa_min, 1.40, 0.001),
    }
    if cfg.isara.chi2_sigma in ("instrument", "instrument-cov"):
        kwargs.update(instrument_sigmas(row, cfg))
    if cfg.isara.chi2_sigma == "instrument-cov":
        cov = observation_covariance(row, dndlogdp, grid, cfg)
        if cov is not None:
            kwargs["obs_cov"] = cov
    return kwargs


def observation_covariance(row, dndlogdp_weighted, grid, cfg):
    # Full observation+model covariance ((m^-1)^2) marginalizing over the
    # correlated structural nuisances; see uncertainty_propagation
    # .build_obs_cov. None when the PSD is too sparse to forward-model.
    from . import uncertainty_propagation as up  # noqa: PLC0415

    ch = cfg.channels
    fin = np.isfinite(dndlogdp_weighted) & (grid.dpg_um > 0)
    if fin.sum() < 10:
        return None
    sca_meas = {w: float(row[f"Sc{w}_dry_mean"]) for w in ch.dry_wvl_sca}
    abs_meas = {w: float(row[f"Abs{w}_mean"]) for w in ch.dry_wvl_abs}
    if not (np.isfinite(list(sca_meas.values())).all()
            and np.isfinite(list(abs_meas.values())).all()):
        return None
    up._import_sphere_optics(cfg.paths.isara_code_dir)
    raw = np.array([row[psd_col_name(d)] for d in grid.dpg_um], float)
    regime = cfg.isara.neph_regime or (
        "pm1" if cfg.psd.impactor_d50_aero_um > 0 else "pm10")
    wvls = sorted({*ch.dry_wvl_sca, *ch.dry_wvl_abs})
    dpg_f = grid.dpg_um[fin]
    dnd_f = dndlogdp_weighted[fin]
    raw_f = raw[fin]
    lnd_sigma = um.OPC_DLND
    if _SIZING is not None:
        # evaluate S with the correction applied at the reference CRI and
        # with the smaller post-correction sizing residual
        k = szc.nearest_candidate(_SIZING, 1.52, 0.005)
        cols = np.where(fin)[0]
        dpg_f, dnd_f = szc.apply(_SIZING, k, dpg_f, dnd_f, cols=cols)
        _, raw_f = szc.apply(_SIZING, k, grid.dpg_um[fin], raw_f, cols=cols)
        raw_f = raw_f  # counts-conserving rescale only
        lnd_sigma = cfg.isara.sizing_residual_lnd
    S = up.build_obs_cov(
        dpg_f, dnd_f, raw_f,
        (cfg.psd.impactor_d50_aero_um, cfg.psd.impactor_gsd,
         cfg.psd.impactor_rho_gcm3),
        sca_meas, abs_meas, list(ch.dry_wvl_sca), list(ch.dry_wvl_abs),
        wvls, float(cfg.window.window_s), regime, lnd_sigma=lnd_sigma,
        n_scale_sigma=cfg.isara.n_scale_sigma)
    return S * 1e-12  # (Mm^-1)^2 -> (m^-1)^2


def instrument_sigmas(row, cfg):
    # Per-window 1-sigma arrays (m^-1) from the UM instrument models,
    # evaluated on the MEASURED window means (never retrieved values); the
    # PSAP model takes the measured scattering at the nearest nephelometer
    # wavelength for its scattering-subtraction term.
    ch = cfg.channels
    t = float(cfg.window.window_s)
    regime = cfg.isara.neph_regime or (
        "pm1" if cfg.psd.impactor_d50_aero_um > 0 else "pm10")
    sca_meas = {w: float(row[f"Sc{w}_dry_mean"]) for w in ch.dry_wvl_sca}
    sca_sigma = np.array([um.sigma_scattering(sca_meas[w], t, w, regime)
                          for w in ch.dry_wvl_sca])
    abs_sigma = []
    for w in ch.dry_wvl_abs:
        near = min(ch.dry_wvl_sca, key=lambda ws: abs(ws - w))
        abs_sigma.append(um.sigma_absorption(float(row[f"Abs{w}_mean"]),
                                             sca_meas[near], t))
    # The kappa target is SYNTHESIZED from the same dry nephelometer channel
    # (dry Sc x gamma adjustment), so the instrument calibration cancels in
    # the ratio; what remains is the gamma-parameterization uncertainty
    # (the historic 1% criterion) plus the non-cancelling noise floor.
    wet_sigma = []
    for w in ch.wet_wvl_sca:
        v = float(row[f"Sc{w}_wet_mean"])
        a = um.NEPH_A[int(w)]
        wet_sigma.append(np.sqrt((0.01 * v) ** 2 + a ** 2 * (um.NEPH_T_REF / t)
                                 + (a * np.sqrt(um.NEPH_T_REF
                                                / um.NEPH_ZERO_DUR)) ** 2))
    wet_sigma = np.array(wet_sigma)
    return {"sca_sigma": sca_sigma * 1e-6,
            "abs_sigma": np.array(abs_sigma) * 1e-6,
            "wet_sigma": wet_sigma * 1e-6}


def _retrieve_one(item):
    """Run one retrieval; never raises (failures become attempt flags of 0)."""
    timestamp, kwargs, lut_key = item
    lut = _LUTS.get(lut_key)
    try:
        result = _ISARA.Retr_PSD(**kwargs, lut=lut, sizing_corr=_SIZING)
    except ValueError as err:  # e.g. <2 valid PSD bins
        result = {
            "attempt_flag_CRI_unitless": 0,
            "attempt_flag_kappa_unitless": 0,
            "retrieval_error": str(err),
        }
    return timestamp, result


def _worker_init(isara_code_dir, scratch_dir, lut_states, sizing_state=None):
    """Per-worker setup: cwd for MOPSMAP temp files and a unique RNG state.

    mopsmap_wrapper names its temp files from time.time() and np.random.randn();
    forked workers inherit identical RNG state, so reseed per PID to avoid
    temp-file collisions.
    """
    global _SIZING
    os.makedirs(scratch_dir, exist_ok=True)
    os.chdir(scratch_dir)
    np.random.seed(os.getpid() & 0xFFFFFFFF)
    import_isara(isara_code_dir)
    _install_luts(lut_states)
    _SIZING = sizing_state


def run_all_windows(windows_df, grid, cfg, progress=True):
    """Retrieve every window with window_qc_flag == 0.

    Returns a DataFrame of Retr_PSD outputs indexed by window center time
    (only for attempted windows; join back onto windows_df afterwards).
    """
    good = windows_df[windows_df["window_qc_flag"] == 0]
    if good.empty:
        return pd.DataFrame()

    global _SIZING
    lut_states = {}
    if cfg.isara.use_lut and cfg.isara.forward_engine == "mopsmap":
        lut_states = prepare_luts(good, grid, cfg, verbose=progress)
    sizing_state = build_sizing_state(grid, cfg)
    _SIZING = sizing_state   # main process too (observation_covariance)
    items = []
    for ts, row in good.iterrows():
        kwargs = build_retr_kwargs(row, grid, cfg)
        items.append((ts, kwargs, _pattern_key(kwargs["dndlogdp_cm3"])))

    results = {}
    n_workers = cfg.isara.n_workers
    if n_workers <= 1:
        import_isara(cfg.paths.isara_code_dir)
        _install_luts(lut_states)
        _SIZING = sizing_state
        os.makedirs(cfg.paths.scratch_dir, exist_ok=True)
        prev_cwd = os.getcwd()
        os.chdir(cfg.paths.scratch_dir)
        try:
            for i, item in enumerate(items):
                ts, res = _retrieve_one(item)
                results[ts] = res
                if progress and (i + 1) % 10 == 0:
                    print(f"  retrieved {i + 1}/{len(items)} windows")
        finally:
            os.chdir(prev_cwd)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(cfg.paths.isara_code_dir, cfg.paths.scratch_dir, lut_states,
                      sizing_state),
        ) as pool:
            for i, (ts, res) in enumerate(pool.map(_retrieve_one, items, chunksize=1)):
                results[ts] = res
                if progress and (i + 1) % 25 == 0:
                    print(f"  retrieved {i + 1}/{len(items)} windows")

    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "time"
    return out
