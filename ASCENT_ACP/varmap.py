"""Resolve logical variable names to merged-DataFrame columns.

Columns in the merged pickle carry long instrument-title prefixes
(e.g. ``In-situ_optical_aerosol_measurements_from_the_NASA_HU-25_Sc550_submicron``)
that differ between deployments. Modules therefore address variables by their
ICARTT suffix (``Sc550_submicron``) and resolve them here.
"""

import re


def resolve(df, suffix, required=True):
    """Return the unique column of ``df`` ending with ``_<suffix>``.

    Raises KeyError if no match (unless ``required=False``, then returns None)
    and ValueError if the suffix is ambiguous.
    """
    matches = [c for c in df.columns if c.endswith("_" + suffix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if required:
            raise KeyError(f"No column ends with '_{suffix}'")
        return None
    raise ValueError(f"Suffix '_{suffix}' is ambiguous: {matches}")


def resolve_bins(df, instrument):
    """Ordered per-bin columns of one sizing instrument.

    Two naming schemes are recognized (both campaign conventions seen so
    far): ordinal ``<instrument>_BinNN`` (ACTIVATE; ordered by bin number,
    required contiguous) and center-diameter ``<instrument>_<NNN>nm``
    (SEAC4RS LARGE; ordered by the nm value).
    """
    # ordinal names may carry a trailing qualifier and lowercase "bin"
    # (KORUS-AQ: SMPS_Bin01_stdPT, LAS_bin01_stdPT)
    pat = re.compile(rf"_{instrument}_[Bb]in(\d+)(?:_[A-Za-z0-9]+)?$")
    found = []
    for c in df.columns:
        m = pat.search(c)
        if m:
            found.append((int(m.group(1)), c))
    if found:
        found.sort()
        nums = [n for n, _ in found]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            raise ValueError(f"{instrument} bins are not contiguous: {nums}")
        return [c for _, c in found]
    pat = re.compile(rf"_{instrument}_(\d+)nm(?:_[A-Za-z0-9]+)?$")
    found = sorted((int(m.group(1)), c) for c in df.columns
                   for m in [pat.search(c)] if m)
    if found:
        return [c for _, c in found]
    # diameter-before-tag schemes (DISCOVER-AQ: dNdlogDp_PSL_100nm_LAS and
    # dNdlogDp_11nm_PSL_SMPS)
    pat = re.compile(
        rf"dNdlogDp_(?:[A-Za-z0-9]+_(\d+)nm|(\d+)nm_[A-Za-z0-9]+)_{instrument}$",
        re.IGNORECASE)
    found = sorted((int(m.group(1) or m.group(2)), c) for c in df.columns
                   for m in [pat.search(c)] if m)
    if found:
        return [c for _, c in found]
    # bare diameter-named columns, instrument only in the title prefix
    # (DISCOVER-AQ 2011: '..._DMT_UHSAS_..._61nm', '..._TSI_LAS_..._0.1006um')
    pat = re.compile(r"_(\d+(?:\.\d+)?)(nm|um)$")
    found = sorted((float(m.group(1)) * (1e3 if m.group(2) == "um" else 1.0), c)
                   for c in df.columns
                   for m in [pat.search(c)]
                   if m and instrument in c)
    if not found:
        raise KeyError(f"No '{instrument}_BinNN', '{instrument}_<NNN>nm', "
                       f"'dNdlogDp_*_<NNN>nm_{instrument}' or "
                       f"diameter-named '{instrument}' columns found")
    return [c for _, c in found]


def bin_centers_from_names(cols):
    """Center diameters (um) parsed from ``<TAG>_<NNN>nm`` column names, or
    None when the columns are not diameter-named."""
    vals = []
    for c in cols:
        m = re.search(r"_(\d+)nm(?:_[A-Za-z0-9]+)?$", c)
        if not m:
            return None
        vals.append(int(m.group(1)) / 1000.0)
    return vals
