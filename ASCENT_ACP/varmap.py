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
    """Return the ordered list of ``<instrument>_BinNN`` columns (e.g. 'LAS')."""
    pat = re.compile(rf"_{instrument}_Bin(\d+)$")
    found = []
    for c in df.columns:
        m = pat.search(c)
        if m:
            found.append((int(m.group(1)), c))
    found.sort()
    nums = [n for n, _ in found]
    if not nums:
        raise KeyError(f"No '{instrument}_BinNN' columns found")
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise ValueError(f"{instrument} bins are not contiguous: {nums}")
    return [c for _, c in found]
