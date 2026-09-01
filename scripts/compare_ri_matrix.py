#!/usr/bin/env python
"""SEAC4RS cross-comparison matrix: ISARA(LAS run) x ISARA(UHSAS run) x
PI-Neph (RRI/IRI at 532 nm) and x DASH-SP (kappa).

Usage:
  python scripts/compare_ri_matrix.py LAS_CONFIG UHSAS_CONFIG OUT_DIR
(bundles are found from each config's output_dir/version/variant)
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from compare_independent_ri import window_mean, panel, INK, MUTED


def load_bundle(cfg):
    p = (f"{cfg['paths']['output_dir']}/ISARA_{cfg['campaign']}_{cfg['year']}_"
         f"{cfg['psd']['variant_name']}_60s_{cfg['output']['version']}.pkl")
    return pickle.load(open(p, "rb"))["results"]


def main(las_cfg_p, uh_cfg_p, out_dir):
    cfg = json.load(open(las_cfg_p))
    cfg_u = json.load(open(uh_cfg_p))
    rl = load_bundle(cfg)
    ru = load_bundle(cfg_u)
    df = pd.read_pickle(cfg["paths"]["input_pkl"])
    win_s = cfg.get("window", {}).get("window_s", 60)

    okl = rl[rl["attempt_flag_CRI_unitless"] == 2]
    oku = ru[ru["attempt_flag_CRI_unitless"] == 2]
    both = okl.index.intersection(oku.index)

    def col_like(*subs):
        hits = [c for c in df.columns
                if all(s.lower() in c.lower() for s in subs)]
        return hits[0] if hits else None

    pn_rri = window_mean(df, col_like("PI-Neph", "Real_532"), rl.index, win_s)
    pn_iri = window_mean(df, col_like("PI-Neph", "Imag_532"), rl.index, win_s)
    gf = window_mean(df, col_like("DASH-SP", "GF"), rl.index, win_s)
    rh = window_mean(df, col_like("DASH-SP", "RH"), rl.index, win_s)
    dash_kappa = (gf ** 3 - 1) * (100 - rh) / rh

    def v(res, col, idx):
        return res[col].astype(float).reindex(idx).to_numpy()

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5), dpi=150)
    fig.patch.set_facecolor("white")

    # --- RRI row ---
    panel(axes[0, 0], v(rl, "dry_RRI_unitless", both), v(ru, "dry_RRI_unitless", both),
          "ISARA RRI (LAS run)", "ISARA RRI (UHSAS run)", "RRI: LAS vs UHSAS runs")
    panel(axes[0, 1], pn_rri.reindex(okl.index).to_numpy(float),
          v(rl, "dry_RRI_unitless", okl.index),
          "PI-Neph Real(m) 532", "ISARA RRI (LAS run)", "RRI: PI-Neph vs LAS run")
    panel(axes[0, 2], pn_rri.reindex(oku.index).to_numpy(float),
          v(ru, "dry_RRI_unitless", oku.index),
          "PI-Neph Real(m) 532", "ISARA RRI (UHSAS run)", "RRI: PI-Neph vs UHSAS run")
    # --- IRI row ---
    panel(axes[1, 0], v(rl, "dry_IRI_unitless", both), v(ru, "dry_IRI_unitless", both),
          "ISARA IRI (LAS run)", "ISARA IRI (UHSAS run)", "IRI: LAS vs UHSAS runs")
    panel(axes[1, 1], pn_iri.reindex(okl.index).to_numpy(float),
          v(rl, "dry_IRI_unitless", okl.index),
          "PI-Neph Imag(m) 532", "ISARA IRI (LAS run)", "IRI: PI-Neph vs LAS run")
    panel(axes[1, 2], pn_iri.reindex(oku.index).to_numpy(float),
          v(ru, "dry_IRI_unitless", oku.index),
          "PI-Neph Imag(m) 532", "ISARA IRI (UHSAS run)", "IRI: PI-Neph vs UHSAS run")
    # --- kappa row ---
    kl = rl[rl["attempt_flag_kappa_unitless"] == 2]
    ku = ru[ru["attempt_flag_kappa_unitless"] == 2]
    kboth = kl.index.intersection(ku.index)
    panel(axes[2, 0], v(rl, "kappa_unitless", kboth), v(ru, "kappa_unitless", kboth),
          "ISARA kappa (LAS run)", "ISARA kappa (UHSAS run)", "kappa: LAS vs UHSAS runs")
    panel(axes[2, 1], dash_kappa.reindex(kl.index).to_numpy(float),
          v(rl, "kappa_unitless", kl.index),
          "DASH-SP kappa", "ISARA kappa (LAS run)", "kappa: DASH-SP vs LAS run")
    panel(axes[2, 2], dash_kappa.reindex(ku.index).to_numpy(float),
          v(ru, "kappa_unitless", ku.index),
          "DASH-SP kappa", "ISARA kappa (UHSAS run)", "kappa: DASH-SP vs UHSAS run")

    fig.suptitle(f"SEAC4RS {cfg['year']}: cross-comparison of RI and kappa "
                 f"(V{cfg['output']['version'].lstrip('V')}, RI sizing "
                 "correction ON)", fontsize=13, color=INK)
    fig.text(0.01, 0.005,
             "All co-windowed 60 s means. PI-Neph: GRASP retrieval, own inlet "
             "(~2 um radius), spheroid kernels; DASH-SP kappa from GF at "
             "instrument RH; ISARA runs differ only in the optical sizer "
             "(LAS-PSL 633 nm vs UHSAS-AmmSO4 1054 nm) feeding the PSD.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = f"{out_dir}/SEAC4RS_2013_RI_kappa_matrix.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
