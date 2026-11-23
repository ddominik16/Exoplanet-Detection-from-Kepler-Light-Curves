"""Turn each TCE into a global view (whole orbit, 2001 bins) and a local view (transit, 201 bins).

    python -m src.preprocess --subset data/raw/subset_full.csv --out-dir data/processed_full

Writes global.npy, local.npy, labels.csv, failed.csv and params.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LC_DIR = RAW / "lightcurves"
OUT = ROOT / "data" / "processed"

CADENCE_DAYS = 29.4 / 60 / 24

PARAMS = {
    "global_bins": 2001,
    "local_bins": 201,
    "local_half_width_durations": 2.0,
    "local_bin_width_durations": 0.16,
    "mask_width_durations": 1.5,
    "flatten_min_window_days": 0.75,
    "flatten_window_durations": 3.0,
}


def load_light_curve(kepid: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(LC_DIR / f"{kepid}.npz")
    t, f = d["time"].astype(np.float64), d["flux"].astype(np.float64)
    ok = np.isfinite(t) & np.isfinite(f)
    order = np.argsort(t[ok])
    return t[ok][order], f[ok][order]


def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in days, transit at 0, range [-period/2, period/2)."""
    return (time - t0 + period / 2) % period - period / 2


def transit_mask(time: np.ndarray, period: float, t0: float, duration: float) -> np.ndarray:
    return np.abs(phase_fold(time, period, t0)) < PARAMS["mask_width_durations"] * duration / 2


def detrend(time: np.ndarray, flux: np.ndarray, mask: np.ndarray, duration: float) -> np.ndarray:
    """Divide out slow variability with lightkurve's Savitzky-Golay flatten, ignoring in-transit points."""
    import lightkurve as lk

    window_days = max(PARAMS["flatten_window_durations"] * duration, PARAMS["flatten_min_window_days"])
    window = max(int(window_days / CADENCE_DAYS) // 2 * 2 + 1, 101)
    lc = lk.LightCurve(time=time, flux=flux)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = lc.flatten(window_length=window, mask=mask)
    return np.asarray(flat.flux.value, dtype=np.float64)


def median_bins(x: np.ndarray, y: np.ndarray, n_bins: int, lo: float, hi: float, width: float) -> np.ndarray:
    """Median of y in n_bins windows of the given width, centred evenly between lo and hi.

    Windows may overlap (local view) or tile the range exactly (global view). Empty windows -> NaN.
    x must be sorted.
    """
    centres = np.linspace(lo, hi, n_bins)
    starts = np.searchsorted(x, centres - width / 2, side="left")
    ends = np.searchsorted(x, centres + width / 2, side="right")
    out = np.full(n_bins, np.nan)
    for i, (a, b) in enumerate(zip(starts, ends)):
        if b > a:
            out[i] = np.median(y[a:b])
    return out


def fill_gaps(view: np.ndarray) -> np.ndarray:
    good = np.isfinite(view)
    if good.sum() < 2:
        raise ValueError("view almost empty")
    if good.all():
        return view
    idx = np.arange(view.size)
    return np.interp(idx, idx[good], view[good])


def normalise(view: np.ndarray) -> np.ndarray:
    """Median -> 0, deepest point -> -1."""
    view = view - np.median(view)
    depth = -view.min()
    if depth <= 0:
        raise ValueError("no dip in view")
    return view / depth


def make_views(kepid: int, period: float, t0: float, duration_hours: float) -> tuple[np.ndarray, np.ndarray]:
    duration = duration_hours / 24.0
    time, flux = load_light_curve(kepid)
    flat = detrend(time, flux, transit_mask(time, period, t0, duration), duration)

    phase = phase_fold(time, period, t0)
    order = np.argsort(phase)
    phase, flat = phase[order], flat[order]
    ok = np.isfinite(flat)
    phase, flat = phase[ok], flat[ok]

    n_g = PARAMS["global_bins"]
    global_view = median_bins(phase, flat, n_g, -period / 2, period / 2, period / n_g)

    half = min(PARAMS["local_half_width_durations"] * duration, period / 2)
    local_view = median_bins(phase, flat, PARAMS["local_bins"], -half, half,
                             PARAMS["local_bin_width_durations"] * duration)

    return (normalise(fill_gaps(global_view)).astype(np.float32),
            normalise(fill_gaps(local_view)).astype(np.float32))


def process_row(row: dict) -> dict:
    try:
        g, l = make_views(int(row["kepid"]), row["tce_period"], row["tce_time0bk"], row["tce_duration"])
        return {"row": row, "global": g, "local": l}
    except Exception as e:
        return {"row": row, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", type=Path, default=RAW / "subset.csv")
    p.add_argument("--limit", type=int, default=None, help="only the first N TCEs (for testing)")
    p.add_argument("--n-jobs", type=int, default=-1, help="parallel processes (-1 = all cores)")
    p.add_argument("--out-dir", type=Path, default=OUT)
    a = p.parse_args()
    out = a.out_dir

    tces = pd.read_csv(a.subset)
    if "label" not in tces:
        tces["label"] = (tces["av_training_set"] == "PC").astype(int)
    have = tces.kepid.map(lambda k: (LC_DIR / f"{k}.npz").exists())
    missing = tces[~have]
    tces = tces[have].reset_index(drop=True)
    if a.limit:
        tces = tces.head(a.limit)
    print(f"{len(tces)} TCEs to process ({len(missing)} skipped: no light curve)")

    results = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(process_row)(r) for r in tces.to_dict("records")
    )

    done = [r for r in results if "error" not in r]
    failed = [{**{k: r["row"][k] for k in ("kepid", "tce_plnt_num", "av_training_set")}, "error": r["error"]}
              for r in results if "error" in r]
    failed += [{"kepid": r.kepid, "tce_plnt_num": r.tce_plnt_num, "av_training_set": r.av_training_set,
                "error": "no light curve"} for r in missing.itertuples()]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "global.npy", np.stack([r["global"] for r in done]))
    np.save(out / "local.npy", np.stack([r["local"] for r in done]))
    keep = ["kepid", "tce_plnt_num", "av_training_set", "label", "tce_period", "tce_time0bk",
            "tce_duration", "tce_depth", "tce_model_snr", "tce_impact"]
    pd.DataFrame([r["row"] for r in done])[keep].to_csv(out / "labels.csv", index=False)
    pd.DataFrame(failed, columns=["kepid", "tce_plnt_num", "av_training_set", "error"]).to_csv(out / "failed.csv", index=False)
    (out / "params.json").write_text(json.dumps(PARAMS, indent=2))

    print(f"\nSaved {len(done)} views to {out}")
    print(pd.DataFrame([r["row"] for r in done]).av_training_set.value_counts().to_string())
    if failed:
        print(f"{len(failed)} failed -> {out / 'failed.csv'}")


if __name__ == "__main__":
    main()
