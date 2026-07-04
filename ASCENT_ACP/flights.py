"""Flight segmentation and the (flight x seconds-of-day) output grid.

The netCDF export replaces the single epoch time axis with two dimensions:
flight number (sequential by takeoff within the campaign year) and seconds
since UTC midnight of that flight's takeoff day. This module decides where
one flight ends and the next begins, and maps every merged-frame row onto
that grid.

Flight envelopes come from the marker instrument's ICARTT files when
available (``merge.flight_marker_instrument``; _L1/_L2 files are separate
flights). This matters because the merge engine interpolates some 1 Hz
variables across the hours *between* two same-day flights, so data presence
alone cannot find the boundary. Rows outside every flight envelope (only
such synthetic inter-flight interpolation) are dropped from the export.

Without ICARTT files (other campaigns, tests), envelopes fall back to
splitting on >20 min gaps in data presence.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import icartt_headers

_PAD_S = 120.0       # envelope padding, so clock-shifted rows near the edges stay
_PRESENCE_GAP_S = 1200.0


@dataclass
class FlightGrid:
    """Row -> (flight, second-of-day) mapping plus per-flight coordinates."""

    flight_number: np.ndarray   # 1..N, takeoff order
    flight_id: list             # e.g. "20200222_L1"
    date: list                  # takeoff day "YYYY-MM-DD"
    midnight_epoch_s: np.ndarray  # UTC midnight of takeoff day (s since 1970)
    takeoff_sod: np.ndarray     # takeoff, s since that midnight
    landing_sod: np.ndarray     # landing; exceeds 86400 if past UTC midnight
    n_seconds: int              # length of the seconds-of-day axis
    step_s: float               # native cadence (axis spacing)
    row_flight: np.ndarray      # per df row; -1 = outside every envelope
    row_sec: np.ndarray         # per df row; axis slot = round(sod / step_s)
    n_dropped: int
    source: str

    @property
    def n_flights(self):
        return len(self.flight_number)

    def time_axis_s(self):
        """Coordinate values: seconds since takeoff-day midnight per slot."""
        return np.arange(self.n_seconds) * self.step_s


def _native_step(index):
    if len(index) < 2:
        return 1.0
    return float(np.median(np.diff(index.view("int64"))) / 1e9)


def _midnight_epoch(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp()


def _presence_spans(df):
    """Fallback envelopes: contiguous data-presence runs split on long gaps."""
    present = df.notna().any(axis=1).to_numpy()
    epoch = np.asarray(df.index.view("int64"), dtype="int64") / 1e9
    t = epoch[present]
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > _PRESENCE_GAP_S)[0]
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(t) - 1]])
    spans = []
    for s, e in zip(starts, ends):
        date = datetime.fromtimestamp(t[s], tz=timezone.utc).strftime("%Y-%m-%d")
        spans.append(icartt_headers.FlightSpan(
            date=date, leg=0, start_epoch_s=t[s], end_epoch_s=t[e]))
    # number same-day flights as legs 1..n
    by_date = {}
    for sp in spans:
        by_date.setdefault(sp.date, []).append(sp)
    for group in by_date.values():
        if len(group) > 1:
            for i, sp in enumerate(group, start=1):
                sp.leg = i
    return spans


def _assign(spans, epoch):
    """Flight index per row from padded envelopes (clipped at midpoints)."""
    starts = np.array([s.start_epoch_s for s in spans]) - _PAD_S
    ends = np.array([s.end_epoch_s for s in spans]) + _PAD_S
    for i in range(len(spans) - 1):
        if ends[i] > starts[i + 1]:
            mid = 0.5 * (spans[i].end_epoch_s + spans[i + 1].start_epoch_s)
            ends[i] = starts[i + 1] = mid
    idx = np.searchsorted(starts, epoch, side="right") - 1
    ok = (idx >= 0) & (epoch <= ends[np.clip(idx, 0, None)])
    idx[~ok] = -1
    return idx


def build(df, cfg):
    """Segment the merged frame into flights and build the output grid."""
    m = cfg.merge
    spans, source = [], "data-presence gaps"
    if m.flight_marker_instrument and m.icartt_dir and Path(m.icartt_dir).is_dir():
        spans = icartt_headers.flight_spans(
            m.icartt_dir, m.filename_regex, m.flight_marker_instrument, m.date_format)
        source = f"ICARTT envelopes of {m.flight_marker_instrument}"

    epoch = np.asarray(df.index.view("int64"), dtype="int64") / 1e9
    lo, hi = epoch.min(), epoch.max()
    spans = [s for s in spans if s.end_epoch_s >= lo and s.start_epoch_s <= hi]
    if not spans:
        spans = _presence_spans(df)
        source = "data-presence gaps"
    spans.sort(key=lambda s: s.start_epoch_s)

    idx = _assign(spans, epoch)
    # days whose rows no envelope claims get a whole-day synthetic flight
    # (e.g. a frame date the marker instrument has no file for)
    left = idx < 0
    if left.any():
        day_key = (epoch // 86400).astype(np.int64)
        for day in np.unique(day_key[left]):
            rows = left & (day_key == day)
            date = datetime.fromtimestamp(day * 86400, tz=timezone.utc).strftime("%Y-%m-%d")
            if any(s.date == date for s in spans):
                continue  # envelope exists; leftover rows here are real drops
            spans.append(icartt_headers.FlightSpan(
                date=date, leg=0, start_epoch_s=epoch[rows].min(),
                end_epoch_s=epoch[rows].max()))
        spans.sort(key=lambda s: s.start_epoch_s)
        idx = _assign(spans, epoch)

    # keep only flights that own at least one row
    counts = np.bincount(idx[idx >= 0], minlength=len(spans))
    keep = np.where(counts > 0)[0]
    renum = np.full(len(spans), -1, np.int32)
    renum[keep] = np.arange(len(keep), dtype=np.int32)
    row_flight = np.where(idx >= 0, renum[np.clip(idx, 0, None)], -1).astype(np.int32)
    spans = [spans[i] for i in keep]

    midnight = np.array([_midnight_epoch(s.date) for s in spans])
    step = _native_step(df.index)
    sod = epoch - np.where(row_flight >= 0, midnight[np.clip(row_flight, 0, None)], 0)
    row_sec = np.rint(sod / step).astype(np.int32)
    row_sec[row_flight < 0] = 0

    valid = row_flight >= 0
    return FlightGrid(
        flight_number=np.arange(1, len(spans) + 1, dtype=np.int32),
        flight_id=[s.flight_id for s in spans],
        date=[s.date for s in spans],
        midnight_epoch_s=midnight.astype(np.int64),
        takeoff_sod=np.array([s.start_epoch_s for s in spans]) - midnight,
        landing_sod=np.array([s.end_epoch_s for s in spans]) - midnight,
        n_seconds=int(row_sec[valid].max()) + 1 if valid.any() else 0,
        step_s=step,
        row_flight=row_flight,
        row_sec=row_sec,
        n_dropped=int((~valid).sum()),
        source=source,
    )


def scatter(grid, values, fill=np.nan, dtype=np.float32):
    """Scatter per-row values onto the (flight, seconds-of-day) grid."""
    out = np.full((grid.n_flights, grid.n_seconds), fill, dtype)
    m = grid.row_flight >= 0
    out[grid.row_flight[m], grid.row_sec[m]] = np.asarray(values)[m]
    return out
