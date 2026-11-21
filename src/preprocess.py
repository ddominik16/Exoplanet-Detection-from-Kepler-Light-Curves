"""Turn each TCE into a global view (2001 bins) and a local view (201 bins).  [Weeks 3-4]

Usage:  python -m src.preprocess
Output: data/processed/global.npy  (N x 2001)
        data/processed/local.npy   (N x 201)
        data/processed/labels.csv  (kepid, tce_plnt_num, label, ...)

Steps follow the "Preprocessing pipeline" section of the project plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LC_DIR = ROOT / "data" / "raw" / "lightcurves"
OUT = ROOT / "data" / "processed"

GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_HALF_WIDTH = 2.0  # in transit durations


def load_light_curve(kepid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (time [BKJD], flux) for one star from data/raw/lightcurves/<kepid>.npz."""
    d = np.load(LC_DIR / f"{kepid}.npz")
    return d["time"], d["flux"]


def transit_mask(time: np.ndarray, period: float, t0: float, duration_days: float, factor: float = 1.5) -> np.ndarray:
    """True for points inside a transit (width = factor x duration)."""
    # TODO: phase = ((time - t0 + period / 2) % period) - period / 2 ; return |phase| < factor * duration / 2
    raise NotImplementedError


def detrend(time: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divide out slow variability while ignoring in-transit points.

    Option A: lightkurve  -> lk.LightCurve(time=time, flux=flux).flatten(window_length=..., mask=mask)
    Option B: robust spline with iterative sigma clipping (closer to AstroNet).
    Keep the window at least ~3x the transit duration so the dip survives.
    """
    # TODO
    raise NotImplementedError


def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in days, transit at 0, range [-period/2, period/2)."""
    # TODO
    raise NotImplementedError


def bin_median(phase: np.ndarray, flux: np.ndarray, n_bins: int, lo: float, hi: float) -> np.ndarray:
    """Median flux in n_bins equal bins between lo and hi; empty bins -> NaN (filled later)."""
    # TODO (scipy.stats.binned_statistic does this in one line)
    raise NotImplementedError


def fill_and_normalise(view: np.ndarray) -> np.ndarray:
    """Interpolate NaN bins, then shift/scale so median = 0 and the minimum (transit) = -1."""
    # TODO
    raise NotImplementedError


def make_views(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Global and local view for one TCE row of the TCE table."""
    period, t0 = row.tce_period, row.tce_time0bk
    duration = row.tce_duration / 24.0  # hours -> days
    time, flux = load_light_curve(int(row.kepid))
    # TODO: mask -> detrend -> fold -> global view (whole orbit) + local view (+-2 durations)
    raise NotImplementedError


def main() -> None:
    # TODO: loop over data/raw/subset.csv (joblib.Parallel for speed), skip stars without a .npz,
    #       stack the views, save global.npy / local.npy / labels.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
