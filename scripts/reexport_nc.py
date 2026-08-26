#!/usr/bin/env python
"""Re-run only the netCDF export from an existing results bundle.

Development helper for iterating on netcdf_export without re-running ISARA::

    python scripts/reexport_nc.py --config configs/activate_2021_full.json \
        [--bundle path/to/bundle.pkl] [--out path/to/out.nc]

Defaults: bundle and output paths derived from the config (same rule as
ASCENT_ACP.run). The config's pipeline settings are used as-is; the bundle
supplies the window/retrieval results and PSD grid.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ASCENT_ACP import filtering, netcdf_export, pipeline  # noqa: E402
from ASCENT_ACP.config import PipelineConfig  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = PipelineConfig.from_json(args.config)
    stem = netcdf_export.output_filename(cfg).removesuffix(".nc")
    bundle_path = args.bundle or str(Path(cfg.paths.output_dir) / f"{stem}.pkl")
    out_path = args.out or str(Path(cfg.paths.output_dir) / f"{stem}.nc")

    with open(bundle_path, "rb") as f:
        b = pickle.load(f)
    res, grid = b["results"], b["grid"]

    df, meta = pipeline.load_inputs(cfg)
    optical = filtering.derive_optical_columns(df, cfg)
    masks = filtering.row_qc(df, optical, cfg)

    out = netcdf_export.export(df, masks, res, grid, cfg, meta=meta, path=out_path)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
