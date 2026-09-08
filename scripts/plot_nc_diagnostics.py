#!/usr/bin/env python
"""Full diagnostic plots for an ISARA campaign NetCDF file.

Two figures, written to <nc_dir>/diagnostic_plots/<nc_stem>/:

  1. timeseries_F<NN>_<date>.png -- multipanel time series for a single
     flight (by default the flight with the most accepted ISARA CRI
     retrievals; override with --flight N).
  2. campaign_boxplots.png -- the same panels campaign-wide, one box per
     flight (median/IQR, whiskers 5-95%, outliers hidden).

Panels (top to bottom; panels may carry up to four stacked y-axes):
  1. all scattering: dry meas vs fit, humidified synth, wet fit, ambient model
  2. absorption meas vs fit                | dry co-albedo (1-SSA) meas vs fit
  3. retrieved real RI (y = full search grid) | retrieved imag RI (log)
  4. AMS composition (discovered per campaign, log)
  5. PSD moments: r_eff (+ ISARA r_eff fit) | v_eff | surface area
  6. flight state: altitude | ambient RH | kappa | ambient growth factor

Core panels are driven by the stable windowed/* schema shared by all
campaigns; AMS species are discovered by name pattern in the
campaign-specific observations/ tree. Missing variables skip their series
(annotated on the panel) rather than crashing.

Usage:
  plot_nc_diagnostics.py FILE.nc [--flight N] [--out DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4

# ---------------------------------------------------------------- palette
C_MEAS = "#2a72c8"   # measured / observed (blue)
C_CALC = "#d1495b"   # ISARA fit / retrieved (red)
C_LEFT = "#2a72c8"
C_RIGHT = "#e07b39"
C_TEAL = "#3aa7a3"
AMS_COLORS = {       # conventional AMS species colors
    "Organic": "#2ca02c",
    "Sulfate": "#d62728",
    "Nitrate": "#1f77b4",
    "Ammonium": "#ff7f0e",
    "Chloride": "#9467bd",
}
INK, MUTED = "#1a1a2e", "#6b6b7b"

AMS_PATTERNS = [
    ("Organic", r"^(OA|Org)(_|$)"),
    ("Sulfate", r"^(SO4|Sulfate)(_|$)"),
    ("Nitrate", r"^(NO3|Nitrate)(_|$)"),
    ("Ammonium", r"^(NH4|Ammonium)(_|$)"),
    ("Chloride", r"^(Chl|Chloride)(_|$)"),
]
AMS_EXCLUDE = re.compile(
    r"prec|_DL_|DL_60s|frac|CVI|ug_per_m3|Flag|Duration|StdtoVol|Corr|Balance",
    re.IGNORECASE,
)

# horizontal outward offset (points) of the extra stacked right spines
AXIS_OFFSET_PT = 42


# ---------------------------------------------------------------- loading
class NCBundle:
    """Thin reader around the ISARA nc file; returns float arrays with NaN."""

    def __init__(self, path):
        self.path = Path(path)
        self.ds = netCDF4.Dataset(path)
        self.nflight = self.ds.dimensions["flight"].size
        self.time_s = np.asarray(self.ds["time"][:], float)
        self.wvl = np.asarray(self.ds["wavelength"][:], float)
        self.window_s = int(float(self.ds.getncattr("window_seconds"))) \
            if "window_seconds" in self.ds.ncattrs() else 60
        self.flight_ids = [str(x) for x in self.ds["flight_id"][:]]
        self.flight_dates = [str(x) for x in self.ds["flight_date"][:]]

    def var(self, path):
        """Variable at group path like 'windowed/retrievals/kappa', or None."""
        node = self.ds
        parts = path.split("/")
        for g in parts[:-1]:
            if g not in node.groups:
                return None
            node = node.groups[g]
        return node.variables.get(parts[-1])

    def get(self, path, flight=None):
        v = self.var(path)
        if v is None:
            return None
        arr = v[flight] if flight is not None else v[:]
        return np.ma.filled(np.asarray(arr, float) if not np.ma.isMaskedArray(arr)
                            else arr.astype(float), np.nan)

    def units(self, path):
        v = self.var(path)
        return getattr(v, "units", "") if v is not None else ""

    # -------- wavelength-channel helpers
    def best_channel(self, path, target=550.0):
        """Index of the wavelength channel nearest `target` that has data."""
        v = self.var(path)
        if v is None or v.ndim != 3:
            return None
        counts = np.zeros(v.shape[2], int)
        for f in range(self.nflight):
            a = self.get(path, f)
            counts += np.isfinite(a).sum(axis=0)
        ok = np.where(counts > 0)[0]
        if not len(ok):
            return None
        return int(ok[np.argmin(np.abs(self.wvl[ok] - target))])

    # -------- window dedup
    def window_firsts(self, flight, values):
        """Mask keeping the first finite sample of each averaging window.

        Windowed variables are broadcast across every second of their window;
        uses windowed/observations/window_index when present, else fixed
        window_seconds bins of the time coordinate.
        """
        wi = self.get("windowed/observations/window_index", flight)
        if wi is None or not np.isfinite(wi).any():
            wi = np.floor(self.time_s / self.window_s)
        wi = np.where(np.isfinite(values), wi, np.nan)
        finite = np.isfinite(values)
        prev = np.full(values.shape, np.nan)
        prev[1:] = wi[:-1]
        return finite & (np.isnan(prev) | (wi != prev))


# ---------------------------------------------------------------- discovery
def walk_group(ds, path):
    node = ds
    for g in path.split("/"):
        if g not in node.groups:
            return None
        node = node.groups[g]
    return node


def find_ams(nb):
    """Discover AMS-style species columns -> list of (label, path, units)."""
    grp = walk_group(nb.ds, "observations/aerosol_composition")
    if grp is None:
        return []
    names = list(grp.variables)
    out = []
    for label, pat in AMS_PATTERNS:
        rx = re.compile(pat)
        for name in names:
            if AMS_EXCLUDE.search(name):
                continue
            v = grp.variables[name]
            if v.ndim != 2:
                continue
            if rx.search(name):
                out.append((label, f"observations/aerosol_composition/{name}",
                            getattr(v, "units", "")))
                break
    return out


# ---------------------------------------------------------------- derived
def psd_moments(nb, flight):
    """(reff um, veff, Ntot cm-3, area um2 cm-3) from the windowed dry PSD."""
    dn = nb.get("windowed/observations/dndlogdp", flight)  # (time, nbin)
    dp = nb.get("windowed/observations/dp_mid")
    if dn is None or dp is None:
        return None, None, None, None
    lo = nb.get("windowed/observations/dp_lower")
    hi = nb.get("windowed/observations/dp_upper")
    if lo is not None and hi is not None and np.isfinite(lo).all():
        dlog = np.log10(hi / lo)
    else:
        dlog = np.gradient(np.log10(dp))
    r = dp / 2.0
    w = dn * dlog                       # dN per bin, cm-3
    with np.errstate(invalid="ignore", divide="ignore"):
        ntot = np.nansum(w, axis=1)
        s2 = np.nansum(w * r**2, axis=1)
        s3 = np.nansum(w * r**3, axis=1)
        reff = s3 / s2
        veff = np.nansum(w * (r[None, :] - reff[:, None])**2 * r**2, axis=1) \
            / (reff**2 * s2)
        area = 4.0 * np.pi * s2
    empty = ~np.isfinite(dn).any(axis=1)
    for a in (ntot, reff, veff, area):
        a[empty] = np.nan
    return reff, veff, ntot, area


def rri_search_range(nb):
    """ISARA real-RI search grid bounds from config_json (fallback 1.47-1.56)."""
    try:
        cfg = json.loads(nb.ds.getncattr("config_json"))
        isara = cfg.get("isara", {})
        return float(isara["rri_min"]), float(isara["rri_max"])
    except Exception:
        return 1.47, 1.56


# ---------------------------------------------------------------- panel spec
def build_panels(nb):
    """Panel definitions. Each panel: dict(title, axes=[axis...]) where an
    axis is dict(series=[S...], yl, log, ylim). Series: dict(label, get,
    color, windowed, err)."""
    W = nb.wvl

    def wl_get(path, ch, transform=None):
        if path is None or ch is None or nb.var(path) is None:
            return None
        if transform is None:
            return lambda f, p=path, c=ch: nb.get(p, f)[:, c]
        return lambda f, p=path, c=ch: transform(nb.get(p, f)[:, c])

    ch_sca = nb.best_channel("windowed/observations/scattering_dry_measured")
    ch_abs = nb.best_channel("windowed/observations/absorption_measured")
    ch_ssa = nb.best_channel("windowed/observations/ssa_measured")
    ams = find_ams(nb)
    # the *_dry_fit variables (MAP-adjusted PSD scale, SEAC4RS V3+) are the
    # actual best fit to the measurements; older files only have *_calculated
    def first_path(*paths):
        for p in paths:
            if nb.var(p) is not None:
                return p
        return None

    sca_fit = first_path("windowed/retrievals/scattering_dry_fit",
                         "windowed/retrievals/scattering_dry_calculated")
    abs_fit = first_path("windowed/retrievals/absorption_dry_fit",
                         "windowed/retrievals/absorption_dry_calculated")
    rri_lo, rri_hi = rri_search_range(nb)
    coalb = lambda v: 1.0 - v

    def lbl(base, ch):
        return f"{base} ({W[ch]:.0f} nm)" if ch is not None else base

    S = lambda label, get, color, windowed=True, err=None: dict(
        label=label, get=get, color=color, windowed=windowed, err=err)
    A = lambda series, yl, log=False, ylim=None: dict(
        series=series, yl=yl, log=log, ylim=ylim)

    panels = []
    panels.append(dict(
        title="scattering: dry closure, humidified fit, ambient model",
        axes=[A([S(lbl("dry meas", ch_sca),
                   wl_get("windowed/observations/scattering_dry_measured", ch_sca),
                   C_MEAS,
                   err=wl_get("windowed/observations/scattering_dry_measured_std",
                              ch_sca)),
                 S(lbl("dry fit", ch_sca), wl_get(sca_fit, ch_sca), C_CALC),
                 S("humid synth (80%)",
                   lambda f: nb.get(
                       "windowed/observations/scattering_humidified_synthesized", f),
                   C_TEAL),
                 # kappa fits the wet/dry ENHANCEMENT ratio (kappa_objective=
                 # 'ratio'), so the comparable fit quantity is the modeled
                 # enhancement applied to the measured dry scattering; the raw
                 # absolute wet_calculated inherits the dry amplitude-closure
                 # error and sits below synth by exactly that factor.
                 S(lbl("wet fit (enh x dry meas)", ch_sca),
                   (lambda f, c=ch_sca:
                    nb.get("windowed/retrievals/scattering_wet_calculated", f)[:, c]
                    / nb.get("windowed/retrievals/scattering_dry_calculated", f)[:, c]
                    * nb.get("windowed/observations/scattering_dry_measured", f)[:, c])
                   if ch_sca is not None else None,
                   "#e39aa5"),
                 S(lbl("ambient model", ch_sca),
                   wl_get("windowed/retrievals/scattering_ambient_calculated",
                          ch_sca),
                   C_RIGHT)],
                "scattering (Mm$^{-1}$)")]))
    panels.append(dict(
        title="absorption closure / dry co-albedo (1$-$SSA)",
        axes=[A([S(lbl("abs meas", ch_abs),
                   wl_get("windowed/observations/absorption_measured", ch_abs),
                   C_MEAS,
                   err=wl_get("windowed/observations/absorption_measured_std",
                              ch_abs)),
                 S(lbl("abs fit", ch_abs), wl_get(abs_fit, ch_abs), C_CALC)],
                "absorption (Mm$^{-1}$)"),
              A([S(lbl("co-albedo meas", ch_ssa),
                   wl_get("windowed/observations/ssa_measured", ch_ssa, coalb),
                   "#6ea9ec"),
                 S(lbl("co-albedo fit", ch_ssa),
                   wl_get("windowed/retrievals/ssa_dry_calculated", ch_ssa,
                          coalb),
                   "#e39aa5")],
                "co-albedo (1$-$SSA)")]))
    panels.append(dict(
        title="retrieved complex refractive index (dry); "
              f"real-RI axis = full search grid {rri_lo:.2f}-{rri_hi:.2f}",
        axes=[A([S("real RI",
                   lambda f: nb.get("windowed/retrievals/refractive_index_real", f),
                   C_LEFT,
                   err=lambda f: nb.get(
                       "windowed/retrievals/refractive_index_real_accepted_std",
                       f))],
                "real RI", ylim=(rri_lo - 0.005, rri_hi + 0.005)),
              A([S("imag RI",
                   lambda f: nb.get("windowed/retrievals/refractive_index_imag", f),
                   C_RIGHT,
                   err=lambda f: nb.get(
                       "windowed/retrievals/refractive_index_imag_accepted_std",
                       f))],
                "imag RI", log=True)]))
    ams_units = ams[0][2] if ams else ""
    if re.search(r"microgram|ug ?s?m-3|ugsm-3", ams_units, re.IGNORECASE) \
            or len(ams_units) > 14:
        ams_units = "ug sm$^{-3}$"
    panels.append(dict(
        title="AMS composition",
        axes=[A([S(label, (lambda f, p=path: nb.get(p, f)),
                   AMS_COLORS.get(label, MUTED), windowed=False)
                 for label, path, _u in ams],
                f"mass ({ams_units})" if ams_units else "mass conc.",
                log=True)]))
    reff_fit_series = []
    if nb.var("windowed/retrievals/effective_radius_fit") is not None:
        reff_fit_series = [S(
            "r_eff fit (ISARA remap)",
            lambda f: nb.get("windowed/retrievals/effective_radius_fit", f),
            C_CALC)]
    panels.append(dict(
        title="windowed PSD moments: size / shape / surface area",
        axes=[A([S("r_eff", lambda f: psd_moments(nb, f)[0], C_LEFT)]
                + reff_fit_series, "r$_{eff}$ (um)"),
              A([S("v_eff", lambda f: psd_moments(nb, f)[1], C_RIGHT)],
                "v$_{eff}$"),
              A([S("surface area", lambda f: psd_moments(nb, f)[3], C_TEAL)],
                "area (um$^2$ cm$^{-3}$)", log=True)]))
    panels.append(dict(
        title="flight state / retrieved hygroscopicity",
        axes=[A([S("altitude", lambda f: nb.get("altitude", f), MUTED,
                   windowed=False)], "altitude (m)"),
              A([S("RH ambient",
                   lambda f: nb.get("windowed/observations/rh_ambient", f),
                   C_TEAL)], "RH (%)"),
              A([S("kappa",
                   lambda f: nb.get("windowed/retrievals/kappa", f), C_LEFT,
                   err=lambda f: nb.get("windowed/retrievals/kappa_std", f))],
                "kappa"),
              A([S("GF ambient",
                   lambda f: nb.get("windowed/retrievals/growth_factor_ambient",
                                    f),
                   C_RIGHT)], "growth factor")]))
    return panels


# ---------------------------------------------------------------- helpers
def pick_flight(nb):
    counts = []
    for f in range(nb.nflight):
        rri = nb.get("windowed/retrievals/refractive_index_real", f)
        if rri is None:
            counts.append(0)
            continue
        keep = nb.window_firsts(f, rri)
        counts.append(int(keep.sum()))
    return int(np.argmax(counts)), counts


def series_values(nb, s, flight, dedup):
    """1-D values for one flight; deduped to one sample/window if windowed."""
    if s["get"] is None:
        return None, None
    v = s["get"](flight)
    if v is None:
        return None, None
    e = None
    if s.get("err"):
        e = s["err"](flight)
    if dedup and s["windowed"]:
        keep = nb.window_firsts(flight, v)
        idx = np.where(keep)[0]
        return idx, (v[idx], None if e is None else e[idx])
    idx = np.where(np.isfinite(v))[0]
    return idx, (v[idx], None if e is None else e[idx])


def stack_axes(ax, panel):
    """Base + twin axes for a panel; extra right spines pushed outward."""
    axes = [ax]
    for k in range(1, len(panel["axes"])):
        tw = ax.twinx()
        if k >= 2:
            tw.spines["right"].set_position(("outward",
                                             AXIS_OFFSET_PT * (k - 1)))
        axes.append(tw)
    return axes


def style_axis(axis, aspec, k, npanel_axes):
    axis.set_ylabel(aspec["yl"], fontsize=9)
    if aspec["log"]:
        axis.set_yscale("log")
    if npanel_axes > 1 and len(aspec["series"]) == 1:
        c = aspec["series"][0]["color"]
        axis.yaxis.label.set_color(c)
        axis.tick_params(axis="y", which="both", labelcolor=c, labelsize=8)
    else:
        axis.tick_params(axis="y", which="both", labelsize=8)


def panel_legend(base_ax, axes_list):
    hl, ll = [], []
    for a in axes_list:
        h, l = a.get_legend_handles_labels()
        hl += h
        ll += l
    if hl:
        base_ax.legend(hl, ll, fontsize=7, ncol=min(len(hl), 6),
                       loc="upper right", framealpha=0.6)


def robust_ylim(axis, values, log, pad=0.12):
    """Percentile-driven limits so a few extreme values don't wash out
    the panel (whiskers/errorbars beyond are visually clipped)."""
    v = values[np.isfinite(values)]
    if log:
        v = v[v > 0]
    if len(v) < 5:
        return
    lo, hi = np.nanpercentile(v, [1, 99])
    if log and lo > 0:
        axis.set_ylim(lo * 0.5, hi * 2.0)
    elif hi > lo:
        p = pad * (hi - lo)
        axis.set_ylim(lo - p, hi + p)


# ---------------------------------------------------------------- figure 1
def plot_timeseries(nb, flight, panels, out_dir):
    hours = nb.time_s / 3600.0
    fig, base_axes = plt.subplots(len(panels), 1,
                                  figsize=(13.5, 2.3 * len(panels)),
                                  sharex=True, dpi=150)
    fig.patch.set_facecolor("white")
    t0 = nb.get("takeoff_time", flight)
    t1 = nb.get("landing_time", flight)

    for ax, p in zip(base_axes, panels):
        axes_list = stack_axes(ax, p)
        drew = False
        for k, (aspec, axis) in enumerate(zip(p["axes"], axes_list)):
            pooled, had_err = [], False
            for s in aspec["series"]:
                idx, vals = series_values(nb, s, flight, dedup=True)
                if idx is None or not len(idx):
                    continue
                v, e = vals
                if aspec["log"]:
                    v = np.where(v > 0, v, np.nan)
                marker = "." if s["windowed"] else None
                ls = "" if s["windowed"] else "-"
                axis.plot(hours[idx], v, marker=marker, ls=ls, ms=3.5, lw=0.9,
                          color=s["color"], label=s["label"], alpha=0.85)
                if e is not None and np.isfinite(e).any():
                    axis.errorbar(hours[idx], v, yerr=e, fmt="none",
                                  ecolor=s["color"], elinewidth=0.5, alpha=0.3)
                    had_err = True
                pooled.append(v[np.isfinite(v)])
                drew = True
            style_axis(axis, aspec, k, len(p["axes"]))
            # keep axis limits driven by the values, not error bars or
            # near-zero log spikes
            if aspec["ylim"]:
                axis.set_ylim(*aspec["ylim"])
            elif pooled:
                allv = np.concatenate(pooled)
                if aspec["log"] or had_err:
                    robust_ylim(axis, allv, aspec["log"], pad=0.15)
        if not drew:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", color=MUTED)
        ax.set_title(p["title"], fontsize=9.5, loc="left", color=INK)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        panel_legend(ax, axes_list)

    # valid-retrieval tick marks along the bottom panel
    rri = nb.get("windowed/retrievals/refractive_index_real", flight)
    keep = nb.window_firsts(flight, rri) if rri is not None \
        else np.zeros(len(nb.time_s), bool)
    if keep.any():
        base_axes[-1].eventplot(
            hours[keep], lineoffsets=base_axes[-1].get_ylim()[0],
            linelengths=(np.diff(base_axes[-1].get_ylim())[0] * 0.06),
            colors=C_CALC, lw=0.6)

    if np.isfinite(t0) and np.isfinite(t1):
        base_axes[-1].set_xlim(t0 / 3600 - 0.2, t1 / 3600 + 0.2)
    base_axes[-1].set_xlabel("time (UTC hours)", fontsize=9)
    fid, fdate = nb.flight_ids[flight], nb.flight_dates[flight]
    title = getattr(nb.ds, "title", nb.path.stem)
    fig.suptitle(f"{title}\nflight {flight} ({fid}, {fdate}) -- "
                 f"red ticks in bottom panel mark valid ISARA windows",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.965, 0.965])
    out = out_dir / f"timeseries_F{flight:02d}_{fdate.replace('-', '')}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 2
def _boxes(ax, groups, positions, width, color):
    ok = [i for i, g in enumerate(groups) if len(g) >= 3]
    if not ok:
        return False
    bp = ax.boxplot([groups[i] for i in ok], positions=[positions[i] for i in ok],
                    widths=width, whis=(5, 95), showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.55, edgecolor=color, lw=0.8)
    for el in ("whiskers", "caps", "medians"):
        for b in bp[el]:
            b.set(color=color, lw=0.9)
    for b in bp["medians"]:
        b.set(color=INK, lw=1.1)
    return True


def plot_campaign_boxes(nb, panels, counts, out_dir):
    nf = nb.nflight
    x = np.arange(nf)
    fig, base_axes = plt.subplots(len(panels), 1,
                                  figsize=(max(11, 0.55 * nf + 4),
                                           2.5 * len(panels)),
                                  sharex=True, dpi=150)
    fig.patch.set_facecolor("white")

    for ax, p in zip(base_axes, panels):
        axes_list = stack_axes(ax, p)
        nser_total = sum(len(a["series"]) for a in p["axes"])
        span = 0.62 if nser_total == 1 else 0.78
        j = 0  # global series slot across all of the panel's axes
        drew = False
        for k, (aspec, axis) in enumerate(zip(p["axes"], axes_list)):
            pooled = []
            for s in aspec["series"]:
                groups = []
                for f in range(nf):
                    idx, vals = series_values(nb, s, f, dedup=True)
                    v = vals[0] if idx is not None else np.array([])
                    if aspec["log"] and len(v):
                        v = v[v > 0]
                    v = v[np.isfinite(v)] if len(v) else v
                    groups.append(v)
                    if len(v):
                        pooled.append(v)
                off = -span / 2 + span * (j + 0.5) / nser_total
                width = span / nser_total * 0.85
                if _boxes(axis, groups, x + off, width, s["color"]):
                    drew = True
                axis.plot([], [], color=s["color"], lw=4, alpha=0.55,
                          label=s["label"])
                j += 1
            style_axis(axis, aspec, k, len(p["axes"]))
            if aspec["ylim"]:
                axis.set_ylim(*aspec["ylim"])
            elif pooled:
                robust_ylim(axis, np.concatenate(pooled), aspec["log"])
        if not drew:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", color=MUTED)
        ax.set_title(p["title"], fontsize=9.5, loc="left", color=INK)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
        panel_legend(ax, axes_list)

    # n valid ISARA windows across the top
    for f in range(nf):
        base_axes[0].annotate(str(counts[f]), (x[f], 1.015),
                              xycoords=("data", "axes fraction"),
                              ha="center", fontsize=6.5, color=C_CALC)
    base_axes[0].annotate("n ISARA:", (-0.9, 1.015),
                          xycoords=("data", "axes fraction"),
                          ha="right", fontsize=6.5, color=C_CALC)

    base_axes[-1].set_xticks(x)
    base_axes[-1].set_xticklabels(
        [f"{f}\n{nb.flight_dates[f][5:] if len(nb.flight_dates[f]) >= 10 else nb.flight_dates[f]}"
         for f in range(nf)], fontsize=6.5)
    base_axes[-1].set_xlabel("flight (number / date)", fontsize=9)
    base_axes[-1].set_xlim(-0.7, nf - 0.3)
    title = getattr(nb.ds, "title", nb.path.stem)
    fig.suptitle(f"{title}\nper-flight distributions "
                 "(boxes: median/IQR, whiskers 5-95%; axis ranges are "
                 "1-99% of the pooled data)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.975, 0.96])
    out = out_dir / "campaign_boxplots.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nc_path", help="ISARA campaign NetCDF file")
    ap.add_argument("--flight", type=int, default=None,
                    help="flight index for the time-series figure "
                         "(default: most valid ISARA retrievals)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    nb = NCBundle(args.nc_path)
    out_dir = Path(args.out) if args.out else \
        nb.path.parent / "diagnostic_plots" / nb.path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    best, counts = pick_flight(nb)
    flight = args.flight if args.flight is not None else best
    print(f"flights: {nb.nflight}; valid ISARA windows per flight: {counts}")
    print(f"time-series flight: {flight} "
          f"({nb.flight_ids[flight]}, {nb.flight_dates[flight]})")

    panels = build_panels(nb)
    p1 = plot_timeseries(nb, flight, panels, out_dir)
    print(f"wrote {p1}")
    p2 = plot_campaign_boxes(nb, panels, counts, out_dir)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
