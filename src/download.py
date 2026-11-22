"""Download the labelled TCE table and Kepler light curves.

Usage (run from the project root, with the environment activated):

    python -m src.download tce
    python -m src.download subset --n 2000
    python -m src.download lightcurves --workers 8
    python -m src.download check --delete

Full dataset (all ~15,700 labelled TCEs):
    python -m src.download subset --all
    python -m src.download lightcurves --subset data/raw/subset_full.csv --workers 8
    python -m src.download fast --subset data/raw/subset_full.csv --workers 12     (same, skips the search step)

Each step skips work that is already done, so you can stop and restart at any time.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", message=".*did not parse as fits unit.*")
# Truncated download -> error -> the star is retried.
warnings.filterwarnings("error", message=".*File may have been truncated.*")


def log(msg: str) -> None:
    """Print to the real console. lightkurve swaps sys.stdout while downloading, and with several
    threads it can be left pointing at a closed buffer, so normal print() fails or vanishes."""
    sys.stdout = sys.__stdout__
    print(msg, file=sys.__stdout__, flush=True)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TCE_CSV = RAW / "dr24_tce.csv"
SUBSET_CSV = RAW / "subset.csv"
FULL_CSV = RAW / "subset_full.csv"
LC_DIR = RAW / "lightcurves"
FITS_TMP = RAW / "_fits_tmp"

COLUMNS = [
    "kepid",
    "tce_plnt_num",
    "tce_period",
    "tce_time0bk",
    "tce_duration",     # hours, not days
    "tce_depth",
    "tce_model_snr",
    "tce_impact",
    "av_training_set",
]

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
LEGACY_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
MANUAL_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView"
    "?app=ExoTbls&config=q1_q17_dr24_tce"
)


def download_tce_table(out: Path = TCE_CSV) -> pd.DataFrame:
    """Fetch the Q1-Q17 DR24 TCE table from the NASA Exoplanet Archive."""
    if out.exists():
        print(f"Already have {out}")
        return pd.read_csv(out)

    cols = ",".join(COLUMNS)
    attempts = [
        ("TAP", TAP_URL, {"query": f"select {cols} from q1_q17_dr24_tce", "format": "csv"}),
        ("legacy API", LEGACY_URL, {"table": "q1_q17_dr24_tce", "select": cols, "format": "csv"}),
    ]
    for name, url, params in attempts:
        try:
            print(f"Trying the {name} ...")
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), comment="#")
            if "av_training_set" in df.columns and len(df) > 10_000:
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                print(f"Saved {len(df):,} TCEs to {out}")
                return df
            print(f"  unexpected response ({len(df)} rows, columns {list(df.columns)[:5]})")
        except Exception as e:
            print(f"  failed: {e}")

    raise SystemExit(
        "Could not download the table automatically.\n"
        f"Open {MANUAL_URL}\n"
        "then Download Table -> CSV, and save it as data/raw/dr24_tce.csv"
    )


def load_labelled(path: Path = TCE_CSV) -> pd.DataFrame:
    """TCE table without UNK rows, plus a binary `label` column (1 = planet candidate)."""
    df = pd.read_csv(path, comment="#")
    df = df[df["av_training_set"].isin(["PC", "AFP", "NTP"])].copy()
    df["label"] = (df["av_training_set"] == "PC").astype(int)
    return df.reset_index(drop=True)


def make_subset(n: int = 2000, seed: int = 42, out: Path = SUBSET_CSV) -> pd.DataFrame:
    """Balanced subset: n/2 planets, n/2 non-planets (AFP and NTP in their natural ratio)."""
    df = load_labelled()
    pos = df[df.label == 1].sample(n // 2, random_state=seed)
    neg = df[df.label == 0].sample(n - n // 2, random_state=seed)
    sub = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    sub.to_csv(out, index=False)
    print(f"Saved {len(sub)} TCEs from {sub.kepid.nunique()} stars to {out}")
    print(sub.av_training_set.value_counts().to_string())
    return sub


def make_full(out: Path = FULL_CSV) -> pd.DataFrame:
    """Every labelled TCE (PC, AFP, NTP), in its natural class ratio."""
    df = load_labelled()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} TCEs from {df.kepid.nunique()} stars to {out}")
    print(df.av_training_set.value_counts().to_string())
    return df


def _quarter_ok(flux: np.ndarray) -> bool:
    med = np.nanmedian(flux)
    return bool(np.isfinite(med) and med > 0 and np.nanstd(flux) < 0.5 * med)


def lightcurve_is_bad(path: Path) -> bool:
    """A normalised light curve should sit near 1. Flag files where >0.1% of points are < 0 or > 3."""
    flux = np.load(path)["flux"]
    return bool(np.mean((flux < 0) | (flux > 3)) > 0.001)


def check_light_curves(delete: bool = False) -> None:
    """Find light curves damaged by broken quarters; with --delete, remove them so a re-run fetches them again."""
    bad = [p for p in sorted(LC_DIR.glob("*.npz")) if lightcurve_is_bad(p)]
    print(f"{len(bad)} of {len(list(LC_DIR.glob('*.npz')))} light curves look broken")
    for p in bad:
        print("  ", p.stem)
        if delete:
            p.unlink()
    if bad and delete:
        print("Deleted. Run `python -m src.download lightcurves` again to re-download them.")


def save_stitched(lcs, out: Path) -> str:
    """Drop broken quarters, stitch, save time/flux/flux_err as a small .npz."""
    import lightkurve as lk

    # Broken quarters would be inverted or blown up by the normalisation in stitch().
    good = [q for q in lcs if _quarter_ok(q.flux.value)]
    if not good:
        return "no usable quarters"
    lc = lk.LightCurveCollection(good).stitch().remove_nans()
    np.savez_compressed(
        out,
        time=lc.time.value.astype(np.float64),
        flux=lc.flux.value.astype(np.float32),
        flux_err=lc.flux_err.value.astype(np.float32),
    )
    return "ok"


def download_one(kepid: int, retries: int = 3, out_dir: Path = LC_DIR) -> str:
    """Search MAST, download all long-cadence quarters for one star, stitch, save, delete the FITS."""
    import lightkurve as lk

    out = out_dir / f"{kepid}.npz"
    if out.exists():
        return "cached"

    tmp = FITS_TMP / str(kepid)
    for attempt in range(1, retries + 1):
        try:
            search = lk.search_lightcurve(
                f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long"
            )
            if len(search) == 0:
                return "not found"
            return save_stitched(search.download_all(download_dir=str(tmp)), out)
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(5 * attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return "error"


# Long-cadence file dates of Kepler quarters Q0-Q17 (from the archive's file names).
QUARTER_STAMPS = [
    "2009131105131", "2009166043257", "2009259160929", "2009350155506", "2010078095331", "2010174085026",
    "2010265121752", "2010355172524", "2011073133259", "2011177032512", "2011271113734", "2012004120508",
    "2012088054726", "2012179063303", "2012277125453", "2013011073258", "2013098041711", "2013131215648",
]
MAST_URL = ("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:KEPLER/url/missions/kepler/lightcurves/"
            "{d}/{k}/kplr{k}-{stamp}_llc.fits")
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_quarter(kepid: int, stamp: str, dest: Path, retries: int = 3) -> str:
    """'ok', 'missing' (star not observed that quarter) or raises after retries."""
    k = f"{kepid:09d}"
    url = MAST_URL.format(d=k[:4], k=k, stamp=stamp)
    for attempt in range(1, retries + 1):
        try:
            r = _session().get(url, timeout=120)
            if r.status_code == 404:
                return "missing"
            r.raise_for_status()
            body = r.content
            if not body.startswith(b"SIMPLE"):
                return "missing"
            expected = r.headers.get("Content-Length")
            if expected is not None and int(expected) != len(body):
                raise IOError(f"truncated: {len(body)} of {expected} bytes")
            dest.write_bytes(body)
            return "ok"
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    return "missing"


def download_one_fast(kepid: int, retries: int = 3, out_dir: Path = LC_DIR) -> str:
    """Like download_one, but skips the MAST search: fetches the 18 quarter files by their known addresses."""
    import lightkurve as lk

    out = out_dir / f"{kepid}.npz"
    if out.exists():
        return "cached"
    tmp = FITS_TMP / f"fast_{kepid}"
    for attempt in range(1, retries + 1):
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            files = []
            for stamp in QUARTER_STAMPS:
                dest = tmp / f"{stamp}.fits"
                if fetch_quarter(kepid, stamp, dest) == "ok":
                    files.append(dest)
            if not files:
                return "not found"
            lcs = [lk.read(str(f)) for f in files]
            return save_stitched(lcs, out)
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(5 * attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return "error"


def download_light_curves(subset: Path = SUBSET_CSV, workers: int = 4, fast: bool = False,
                          out_dir: Path = LC_DIR, kepids: list[int] | None = None, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not kepids:
        kepids = sorted(pd.read_csv(subset).kepid.unique())
    todo = [k for k in kepids if not (out_dir / f"{k}.npz").exists()]
    log(f"{len(kepids)} stars requested, {len(todo)} still to download" + (" (fast mode)" if fast else ""))
    if limit:
        todo = todo[:limit]
        log(f"--limit: downloading only {len(todo)}")
    worker = download_one_fast if fast else download_one

    failed = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, k, 3, out_dir): k for k in todo}
        for i, f in enumerate(as_completed(futures), 1):
            k = futures[f]
            try:
                status = f.result()
            except Exception as e:
                status = f"error: {e}"
            if status not in ("ok", "cached"):
                failed.append((k, status))
            if i % 25 == 0 or i == len(todo):
                per_min = i / max(time.time() - start, 1) * 60
                eta_h = (len(todo) - i) / max(per_min, 1e-9) / 60
                log(f"  {i}/{len(todo)} done, {len(failed)} failed  ({per_min:.1f} stars/min, ~{eta_h:.1f} h left)")

    if todo:
        log(f"finished {len(todo)} stars in {(time.time() - start) / 60:.1f} min")
    if failed:
        pd.DataFrame(failed, columns=["kepid", "status"]).to_csv(RAW / "failed_downloads.csv", index=False)
        log(f"{len(failed)} failures written to data/raw/failed_downloads.csv (just re-run to retry)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tce", help="download the DR24 TCE table")
    s = sub.add_parser("subset", help="make a balanced subset")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--all", action="store_true", help="every labelled TCE -> data/raw/subset_full.csv")
    l = sub.add_parser("lightcurves", help="download light curves for the subset")
    l.add_argument("--workers", type=int, default=4)
    l.add_argument("--subset", type=Path, default=SUBSET_CSV)
    fa = sub.add_parser("fast", help="like 'lightcurves' but without the MAST search step")
    fa.add_argument("--workers", type=int, default=8)
    fa.add_argument("--subset", type=Path, default=SUBSET_CSV)
    fa.add_argument("--kepids", type=int, nargs="*", default=None, help="only these stars")
    fa.add_argument("--limit", type=int, default=None, help="only the first N missing stars")
    fa.add_argument("--out-dir", type=Path, default=LC_DIR)
    c = sub.add_parser("check", help="find light curves broken by bad quarters")
    c.add_argument("--delete", action="store_true", help="delete them so they get re-downloaded")
    a = p.parse_args()

    if a.cmd == "tce":
        download_tce_table()
    elif a.cmd == "subset":
        make_full() if a.all else make_subset(a.n, a.seed)
    elif a.cmd == "lightcurves":
        download_light_curves(a.subset, a.workers)
    elif a.cmd == "fast":
        download_light_curves(a.subset, a.workers, fast=True, out_dir=a.out_dir, kepids=a.kepids, limit=a.limit)
    elif a.cmd == "check":
        check_light_curves(a.delete)


if __name__ == "__main__":
    main()
