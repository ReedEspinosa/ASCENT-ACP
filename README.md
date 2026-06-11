# ASCENT-ACP
Atmospheric Suborbital Classification &amp; Evaluation Network Tool - Aerosol, Cloud, and Precipitation

End-to-end processing of airborne in-situ aerosol data (NASA ACTIVATE and similar
ICARTT-based campaigns):

1. **Read & merge** all ICARTT files for a campaign into a 1 Hz pandas DataFrame
   (`ASCENT_ACP.run_ascent_acp_merge`, wrapping the sibling `icartt_read_and_merge` package).
2. **Clock alignment** of instruments to the LAS reference via cross-correlation
   (`apply_clock_alignment.py`, configured by `clock_alignment_results/variable_shift_table.csv`).
3. **Quality filtering** following Kacenelenbogen et al. (2022, ACP 22, 3713, Appendix A1):
   cloud screening with CDP/FCDP probes, inlet flag, minimum dry Sc450 signal and SSA
   sanity filters (`ASCENT_ACP/filtering.py`).
4. **Window averaging** (default 60 s) with stability screening on the scattering
   Angstrom exponent (`ASCENT_ACP/windows.py`).
5. **ISARA retrieval** of the dry complex refractive index and hygroscopicity kappa
   per window, calling `ISARA.Retr_PSD` from the sibling
   [ISARA_code](../ISARA_code) checkout, which drives the MOPSMAP Fortran package
   (`ASCENT_ACP/isara_bridge.py`).
6. **netCDF export** (CF-1.8 style) with full provenance: config JSON, git SHAs and
   ICARTT instrument metadata in the global attributes (`ASCENT_ACP/netcdf_export.py`).

## Running the pipeline

```bash
python -m ASCENT_ACP.pipeline --config configs/activate_2021.json \
    [--dates 2021-05-13 ...] [--max-windows 20] [--no-netcdf] [--plots]
```

Year-specific settings (column-name suffixes, size-cut variant, paths to the merged
pickle, ISARA checkout and MOPSMAP executable/optical dataset) live in
`configs/activate_*.json`. Note that ACTIVATE 2020 reports *total* scattering while
2021 reports *submicron* (PM1) scattering, so each year runs a single matching PSD
truncation variant (`psd.psd_max_um`).

LAS and SMPS size-bin definitions (transcribed from the ICARTT headers) ship in
`ASCENT_ACP/data/*.csv`.

## Tests

```bash
python -m pytest tests/
```

Unit tests stub the ISARA/MOPSMAP layer and run without the Fortran executable.
