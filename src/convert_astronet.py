"""Convert AstroNet's pre-computed views (TFRecord files) into this project's format.

Download and extract (see README) so the files are in data/raw/astronet/tfrecord/, then:

    python -m src.convert_astronet --inspect
    python -m src.convert_astronet
    python -m src.convert_astronet --only data/raw/subset.csv --out-dir data/processed_astronet_2000

--inspect   prints what one record contains, to check the field names
(default)   all TCEs -> data/processed_astronet/
--only      just the TCEs listed in a CSV (kepid, tce_plnt_num), e.g. your 2,000-signal sample,
            to compare AstroNet's views with yours on exactly the same signals

Output: global.npy, local.npy, labels.csv (same columns as src/preprocess.py, filled from
data/raw/dr24_tce.csv), plus astronet_split (which AstroNet file the record came from).
No TensorFlow needed: the file format is read directly.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "astronet" / "tfrecord"
TCE_CSV = ROOT / "data" / "raw" / "dr24_tce.csv"
OUT = ROOT / "data" / "processed_astronet"
KEEP = ["kepid", "tce_plnt_num", "av_training_set", "label", "tce_period", "tce_time0bk",
        "tce_duration", "tce_depth", "tce_model_snr", "tce_impact"]


def read_records(path: Path):
    """TFRecord framing: uint64 length, uint32 crc, payload, uint32 crc."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                return
            (n,) = struct.unpack("<Q", head)
            f.read(4)
            data = f.read(n)
            f.read(4)
            if len(data) != n:
                raise ValueError(f"{path.name}: truncated record")
            yield data


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, pos
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            n, pos = _varint(buf, pos)
            val, pos = buf[pos:pos + n], pos + n
        elif wire == 5:
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, val


def _feature(buf: bytes):
    """tf.train.Feature -> list of bytes / np.float32 array / list of int."""
    for kind, _, inner in _fields(buf):
        if kind == 1:
            return [v for f, _, v in _fields(inner) if f == 1]
        if kind == 2:
            vals = []
            for f, wire, v in _fields(inner):
                if f == 1:
                    vals.append(np.frombuffer(v, "<f4") if wire == 2 else np.frombuffer(v, "<f4"))
            return np.concatenate(vals) if vals else np.zeros(0, np.float32)
        if kind == 3:
            vals = []
            for f, wire, v in _fields(inner):
                if f != 1:
                    continue
                if wire == 2:
                    p = 0
                    while p < len(v):
                        x, p = _varint(v, p)
                        vals.append(x - (1 << 64) if x >= 1 << 63 else x)
                else:
                    vals.append(v - (1 << 64) if v >= 1 << 63 else v)
            return vals
    return []


def parse_example(buf: bytes) -> dict:
    """tf.train.Example -> {name: value}."""
    out = {}
    for f, _, features in _fields(buf):
        if f != 1:
            continue
        for g, _, entry in _fields(features):
            if g != 1:
                continue
            key, value = None, b""
            for h, _, v in _fields(entry):
                if h == 1:
                    key = v.decode()
                elif h == 2:
                    value = v
            out[key] = _feature(value)
    return out


def _scalar(v):
    if isinstance(v, np.ndarray):
        return float(v[0]) if v.size else np.nan
    if isinstance(v, list) and v:
        x = v[0]
        return x.decode() if isinstance(x, bytes) else x
    return v


def record_files(src: Path) -> list[Path]:
    files = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"No files in {src}. Extract the download into data/raw/astronet/ first (see README).")
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--only", type=Path, default=None, help="CSV with kepid, tce_plnt_num to keep")
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()

    files = record_files(a.src)
    if a.inspect:
        ex = parse_example(next(read_records(files[0])))
        print(f"{len(files)} files, e.g. {[f.name for f in files[:4]]}")
        print(f"first record of {files[0].name}:")
        for k, v in sorted(ex.items()):
            shown = f"array of {len(v)}" if isinstance(v, np.ndarray) and len(v) > 3 else _scalar(v)
            print(f"  {k:28s} {shown}")
        return

    rows, gviews, lviews = [], [], []
    for path in files:
        for rec in read_records(path):
            ex = parse_example(rec)
            rows.append({"kepid": int(_scalar(ex["kepid"])), "tce_plnt_num": int(_scalar(ex["tce_plnt_num"])),
                         "astronet_label": _scalar(ex.get("av_training_set", [b""])),
                         "astronet_split": path.name.split("-")[0]})
            gviews.append(np.asarray(ex["global_view"], np.float32))
            lviews.append(np.asarray(ex["local_view"], np.float32))
    meta = pd.DataFrame(rows)
    print(f"read {len(meta)} records from {len(files)} files: {meta.astronet_split.value_counts().to_dict()}")

    tce = pd.read_csv(TCE_CSV, comment="#")
    tce = tce[tce.av_training_set.isin(["PC", "AFP", "NTP"])].copy()
    tce["label"] = (tce.av_training_set == "PC").astype(int)
    merged = meta.merge(tce, on=["kepid", "tce_plnt_num"], how="left", validate="one_to_one")
    unmatched = merged.av_training_set.isna()
    if unmatched.any():
        print(f"{unmatched.sum()} records not in the labelled DR24 table: dropped")
    mismatch = (~unmatched) & (merged.astronet_label != merged.av_training_set)
    if mismatch.any():
        print(f"warning: {mismatch.sum()} records whose AstroNet label differs from dr24_tce.csv (using dr24_tce.csv)")
    keep = ~unmatched.to_numpy()

    if a.only is not None:
        only = pd.read_csv(a.only)[["kepid", "tce_plnt_num"]]
        wanted = merged[["kepid", "tce_plnt_num"]].merge(only.assign(_w=1), how="left")._w.notna().to_numpy()
        print(f"--only: {wanted.sum()} of {len(only)} listed TCEs found in AstroNet's data")
        keep &= wanted

    G, L = np.stack(gviews)[keep], np.stack(lviews)[keep]
    out = merged[keep].reset_index(drop=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(a.out_dir / "global.npy", G)
    np.save(a.out_dir / "local.npy", L)
    out[KEEP + ["astronet_split"]].to_csv(a.out_dir / "labels.csv", index=False)
    print(f"saved {len(out)} TCEs to {a.out_dir}: global {G.shape}, local {L.shape}")
    print(out.av_training_set.value_counts().to_string())


if __name__ == "__main__":
    main()
