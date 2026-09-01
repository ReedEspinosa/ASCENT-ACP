"""Size-bin definitions and the merged SMPS+LAS PSD grid fed to ISARA.

Bin tables live in ``ASCENT_ACP/data/*.csv`` (columns ``dpl,dpg,dpu`` in
micrometers, transcribed from the ACTIVATE ICARTT headers) in the same format
read by ISARA_code's ``load_sizebins.Load``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import varmap


def load_bins(csv_path):
    """Read a bin CSV -> dict of numpy arrays {dpl, dpg, dpu} (diameters, um)."""
    tbl = pd.read_csv(csv_path)
    missing = {"dpl", "dpg", "dpu"} - set(tbl.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns {sorted(missing)}")
    return {k: tbl[k].to_numpy(float) for k in ("dpl", "dpg", "dpu")}


@dataclass
class PSDGrid:
    """Fixed nominal grid of the merged size distribution."""

    dpl_um: np.ndarray
    dpg_um: np.ndarray
    dpu_um: np.ndarray
    columns: list  # merged-DataFrame column per slot, same order
    instrument: np.ndarray  # 'SMPS' or 'LAS' per slot
    # impactor penetration per bin (1.0 everywhere when no impactor is
    # configured); applied to the PSD fed to the RETRIEVAL only
    penetration: np.ndarray = None

    def __len__(self):
        return len(self.dpg_um)


def impactor_penetration(dpg_um, psd_cfg):
    """Log-logistic impactor penetration evaluated at geometric diameters.

    P(D_a) = 1/(1 + (D_a/D50)^s) with s = ln(5.25)/ln(gsd), so P = 0.5 at
    D50 and 0.84/0.16 at D50/gsd and D50*gsd (a slope expressed as the
    16-84% geometric standard deviation). Aerodynamic diameter from the
    geometric/optical grid via D_a = dpg*sqrt(rho) (spheres, slip neglected).
    Returns ones when impactor_d50_aero_um is 0 (no impactor).
    """
    dpg_um = np.asarray(dpg_um, float)
    if psd_cfg.impactor_d50_aero_um <= 0:
        return np.ones_like(dpg_um)
    if psd_cfg.impactor_gsd <= 1.0:
        raise ValueError("impactor_gsd must be > 1 (16-84% steepness)")
    d_aero = dpg_um * np.sqrt(psd_cfg.impactor_rho_gcm3)
    s = np.log(5.25) / np.log(psd_cfg.impactor_gsd)
    return 1.0 / (1.0 + (d_aero / psd_cfg.impactor_d50_aero_um) ** s)


def build_grid(df, psd_cfg):
    """Concatenate SMPS and LAS bins (ascending) and truncate to the variant cut.

    ``df`` is the merged DataFrame (used only to resolve bin column names).
    Truncation keeps bins whose *center* falls at or below ``psd_max_um``
    (and above ``smps_min_dp_um``), never beyond ``inlet_cutoff_um``.
    """
    smps = load_bins(psd_cfg.smps_bins_csv)
    las = load_bins(psd_cfg.las_bins_csv)
    smps_cols = varmap.resolve_bins(df, "SMPS")
    las_cols = varmap.resolve_bins(df, "LAS")
    if len(smps_cols) != len(smps["dpg"]) or len(las_cols) != len(las["dpg"]):
        raise ValueError(
            f"Bin-table/DataFrame mismatch: SMPS {len(smps['dpg'])} vs "
            f"{len(smps_cols)} cols, LAS {len(las['dpg'])} vs {len(las_cols)} cols"
        )
    if smps["dpu"][-1] > las["dpl"][0] + 1e-9:
        raise ValueError("SMPS and LAS bins overlap; merge logic assumes none")

    dpl = np.concatenate([smps["dpl"], las["dpl"]])
    dpg = np.concatenate([smps["dpg"], las["dpg"]])
    dpu = np.concatenate([smps["dpu"], las["dpu"]])
    cols = smps_cols + las_cols
    instr = np.array(["SMPS"] * len(smps_cols) + ["LAS"] * len(las_cols))

    if not np.all(np.diff(dpg) > 0):
        raise ValueError("Merged bin centers are not strictly increasing")

    cut = min(psd_cfg.psd_max_um, psd_cfg.inlet_cutoff_um)
    pen = impactor_penetration(dpg, psd_cfg)
    keep = ((dpg <= cut) & (dpg >= psd_cfg.smps_min_dp_um)
            & (pen >= psd_cfg.impactor_min_penetration))
    if keep.sum() < 2:
        raise ValueError(f"Fewer than 2 bins survive the {cut} um cut")
    return PSDGrid(
        dpl_um=dpl[keep],
        dpg_um=dpg[keep],
        dpu_um=dpu[keep],
        columns=[c for c, k in zip(cols, keep) if k],
        instrument=instr[keep],
        penetration=pen[keep],
    )
