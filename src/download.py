"""Download the labelled TCE table and the Kepler light curves.

    python -m src.download tce                       # DR24 TCE table -> data/raw/dr24_tce.csv
    python -m src.download subset --all              # all labelled TCEs -> data/raw/subset_full.csv
    python -m src.download subset --n 2000           # balanced 2,000-TCE sample -> data/raw/subset.csv
    python -m src.download lightcurves --subset data/raw/subset_full.csv --workers 12
    python -m src.download check --delete            # remove damaged light curves, then re-run lightcurves

Every step skips work already done, so it can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import io
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", message=".*did not parse as fits unit.*")
warnings.filterwarnings("error", message=".*File may have been truncated.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TCE_CSV = RAW / "dr24_tce.csv"
SUBSET_CSV = RAW / "subset.csv"
FULL_CSV = RAW / "subset_full.csv"
LC_DIR = RAW / "lightcurves"
FITS_TMP = RAW / "_fits_tmp"

COLUMNS = ["kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration", "tce_depth",
           "tce_model_snr", "tce_impact", "av_training_set"]

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
LEGACY_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
MANUAL_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=q1_q17_dr24_tce"

# Long-cadence file dates of Kepler quarters Q0-Q17.
QUARTER_STAMPS = [
    "2009131105131", "2009166043257", "2009259160929", "2009350155506", "2010078095331", "2010174085026",
    "2010265121752", "2010355172524", "2011073133259", "2011177032512", "2011271113734", "2012004120508",
    "2012088054726", "2012179063303", "2012277125453", "2013011073258", "2013098041711", "2013131215648",
]
MAST_URL = ("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:KEPLER/url/missions/kepler/lightcurves/"
            "{d}/{k}/kplr{k}-{stamp}_llc.fits")


def download_tce_table(out: Path = TCE_CSV) -> pd.DataFrame:
    """Q1-Q17 DR24 TCE table from the NASA Exoplanet Archive."""
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
            print(f"  unexpected response ({len(df)} rows)")
        except Exception as e:
            print(f"  failed: {e}")
    raise SystemExit(f"Could not download the table. Open {MANUAL_URL}, download it as CSV "
                     "and save it as data/raw/dr24_tce.csv")


def load_labelled(path: Path = TCE_CSV) -> pd.DataFrame:
    """TCEs labelled PC, AFP or NTP, with label = 1 for planet candidates."""
    df = pd.read_csv(path, comment="#")
    df = df[df["av_training_set"].isin(["PC", "AFP", "NTP"])].copy()
    df["label"] = (df["av_training_set"] == "PC").astype(int)
    return df.reset_index(drop=True)


def make_subset(n: int = 2000, seed: int = 42, out: Path = SUBSET_CSV) -> pd.DataFrame:
    """n/2 planets and n/2 non-planets (AFP and NTP in their natural ratio)."""
    df = load_labelled()
    pos = df[df.label == 1].sample(n // 2, random_state=seed)
    neg = df[df.label == 0].sample(n - n // 2, random_state=seed)
    sub = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    sub.to_csv(out, index=False)
    print(f"Saved {len(sub)} TCEs from {sub.kepid.nunique()} stars to {out}")
    print(sub.av_training_set.value_counts().to_string())
    return sub


def make_full(out: Path = FULL_CSV) -> pd.DataFrame:
    df = load_labelled()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} TCEs from {df.kepid.nunique()} stars to {out}")
    print(df.av_training_set.value_counts().to_string())
    return df


def _quarter_ok(flux: np.ndarray) -> bool:
    """Quarters with a non-positive or noise-dominated median would be inverted or blown up when normalised."""
    med = np.nanmedian(flux)
    return bool(np.isfinite(med) and med > 0 and np.nanstd(flux) < 0.5 * med)


def lightcurve_is_bad(path: Path) -> bool:
    """A normalised light curve sits near 1; flag files where >0.1% of points are < 0 or > 3."""
    flux = np.load(path)["flux"]
    return bool(np.mean((flux < 0) | (flux > 3)) > 0.001)


def check_light_curves(delete: bool = False) -> None:
    bad = [p for p in sorted(LC_DIR.glob("*.npz")) if lightcurve_is_bad(p)]
    print(f"{len(bad)} of {len(list(LC_DIR.glob('*.npz')))} light curves look broken")
    for p in bad:
        print("  ", p.stem)
        if delete:
            p.unlink()
    if bad and delete:
        print("Deleted. Run `python -m src.download lightcurves` again to re-download them.")


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_quarter(kepid: int, stamp: str, dest: Path, retries: int = 3) -> str:
    """'ok' or 'missing' (star not observed that quarter); raises after repeated errors."""
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


def download_one(kepid: int, out_dir: Path = LC_DIR, retries: int = 3) -> str:
    """Fetch all quarters of one star, drop broken ones, stitch (each quarter normalised), save a small .npz."""
    import lightkurve as lk

    out = out_dir / f"{kepid}.npz"
    if out.exists():
        return "cached"
    tmp = FITS_TMP / str(kepid)
    for attempt in range(1, retries + 1):
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            files = [tmp / f"{s}.fits" for s in QUARTER_STAMPS
                     if fetch_quarter(kepid, s, tmp / f"{s}.fits") == "ok"]
            if not files:
                return "not found"
            good = [q for q in (lk.read(str(f)) for f in files) if _quarter_ok(q.flux.value)]
            if not good:
                return "no usable quarters"
            lc = lk.LightCurveCollection(good).stitch().remove_nans()
            np.savez_compressed(out, time=lc.time.value.astype(np.float64),
                                flux=lc.flux.value.astype(np.float32),
                                flux_err=lc.flux_err.value.astype(np.float32))
            return "ok"
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(5 * attempt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return "error"


def download_light_curves(subset: Path = SUBSET_CSV, workers: int = 12, out_dir: Path = LC_DIR,
                          kepids: list[int] | None = None, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kepids = kepids or sorted(pd.read_csv(subset).kepid.unique())
    todo = [k for k in kepids if not (out_dir / f"{k}.npz").exists()]
    print(f"{len(kepids)} stars requested, {len(todo)} still to download", flush=True)
    if limit:
        todo = todo[:limit]

    failed, start = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, k, out_dir): k for k in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                status = f.result()
            except Exception as e:
                status = f"error: {e}"
            if status not in ("ok", "cached"):
                failed.append((futures[f], status))
            if i % 25 == 0 or i == len(todo):
                per_min = i / max(time.time() - start, 1) * 60
                eta_h = (len(todo) - i) / per_min / 60
                print(f"  {i}/{len(todo)} done, {len(failed)} failed  ({per_min:.1f} stars/min, ~{eta_h:.1f} h left)",
                      flush=True)
    if failed:
        pd.DataFrame(failed, columns=["kepid", "status"]).to_csv(RAW / "failed_downloads.csv", index=False)
        print(f"{len(failed)} failures written to data/raw/failed_downloads.csv (re-run to retry)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tce", help="download the DR24 TCE table")
    s = sub.add_parser("subset", help="list of TCEs to use")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--all", action="store_true", help="every labelled TCE -> data/raw/subset_full.csv")
    l = sub.add_parser("lightcurves", help="download light curves")
    l.add_argument("--subset", type=Path, default=SUBSET_CSV)
    l.add_argument("--workers", type=int, default=12)
    l.add_argument("--kepids", type=int, nargs="*", default=None, help="only these stars")
    l.add_argument("--limit", type=int, default=None, help="only the first N missing stars")
    l.add_argument("--out-dir", type=Path, default=LC_DIR)
    c = sub.add_parser("check", help="find light curves damaged by bad quarters")
    c.add_argument("--delete", action="store_true")
    a = p.parse_args()

    if a.cmd == "tce":
        download_tce_table()
    elif a.cmd == "subset":
        make_full() if a.all else make_subset(a.n, a.seed)
    elif a.cmd == "lightcurves":
        download_light_curves(a.subset, a.workers, a.out_dir, a.kepids, a.limit)
    elif a.cmd == "check":
        check_light_curves(a.delete)


if __name__ == "__main__":
    main()
