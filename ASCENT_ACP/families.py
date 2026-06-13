"""Group merged-pickle columns into logical instrument families for netCDF.

Each column in the merged pickle carries a long instrument-title prefix (the
ICARTT data-source title). ``assign_families`` maps every column to a family by
**longest-prefix match** against the campaign's ``title_to_family`` map, so the
family map file alone is sufficient -- the meta pickle is only consulted later
for per-instrument provenance attributes.

Campaign maps live in ``ASCENT_ACP/data/<campaign>_instrument_families.json``.
If a campaign has no map, every distinct title becomes its own family.
"""

import json
import re
from pathlib import Path

_DATA = Path(__file__).parent / "data"


def map_path(campaign):
    return _DATA / f"{campaign}_instrument_families.json"


def load_family_map(campaign, path=None):
    """Return the family-map dict for ``campaign`` (or None if none exists)."""
    p = Path(path) if path else map_path(campaign)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _sanitize(title):
    """Fallback family name from a raw instrument title (valid netCDF group)."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_").lower()
    return s[:48] or "unnamed"


def assign_families(columns, family_map, titles=None):
    """Map each column to (family, title) by longest-prefix instrument title.

    Title candidates come from ``family_map``'s ``title_to_family`` keys when a
    map is given, otherwise from ``titles`` (e.g. the meta ``Data_Info`` keys).
    ``family`` is the mapped family (map present), the sanitized title (no map
    but title matched), or ``"other"`` with the column's leading token when no
    title prefixes the column.
    """
    if family_map:
        candidates = list(family_map.get("title_to_family", {}))
    else:
        candidates = list(titles or [])
    candidates.sort(key=len, reverse=True)

    out = {}
    for col in columns:
        title = next((t for t in candidates if col.startswith(t + "_")), None)
        if title is None:
            out[col] = ("other", col.split("_")[0])
        elif family_map:
            out[col] = (family_map["title_to_family"][title], title)
        else:
            out[col] = (_sanitize(title), title)
    return out


def family_order(family_map, present):
    """Ordered family list: configured order first, then any extras, 'other' last."""
    present = set(present)
    order = list(family_map.get("family_order", [])) if family_map else []
    ordered = [f for f in order if f in present]
    ordered += sorted(f for f in present if f not in order and f != "other")
    if "other" in present:
        ordered.append("other")
    return ordered


def family_long_name(family_map, family):
    if family_map:
        return family_map.get("family_long_name", {}).get(family)
    return None
