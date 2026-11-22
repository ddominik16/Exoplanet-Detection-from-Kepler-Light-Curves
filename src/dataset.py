"""Train/val/test split (by star) and the PyTorch Dataset.

Usage:
    python -m src.dataset split           # creates data/processed/split.csv (once!)

Full dataset, keeping every star of the first split where it was (so old and new results compare fairly):
    python -m src.dataset split --subset data/raw/subset_full.csv --out-dir data/processed_full --keep data/processed/split.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # the split command works without PyTorch
    torch = None
    Dataset = object

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SUBSET_CSV = ROOT / "data" / "raw" / "subset.csv"
SPLIT_CSV = PROCESSED / "split.csv"


def make_split(tces: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """80/10/10 train/val/test, stratified by label and grouped by star.

    All TCEs of one star land in the same split, otherwise the test score is inflated.
    Returns kepid, tce_plnt_num, split.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = tces["label"] if "label" in tces else (tces["av_training_set"] == "PC").astype(int)
    folds = np.empty(len(tces), dtype=int)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(cv.split(tces, y, groups=tces["kepid"])):
        folds[idx] = k
    split = np.where(folds == 0, "test", np.where(folds == 1, "val", "train"))
    return pd.DataFrame({"kepid": tces["kepid"], "tce_plnt_num": tces["tce_plnt_num"], "split": split})


def extend_split(tces: pd.DataFrame, previous: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stars already in `previous` keep their split; all other stars are split 80/10/10 as usual."""
    star_split = previous.groupby("kepid").split.first()
    known = tces.kepid.map(star_split)
    out = pd.DataFrame({"kepid": tces.kepid, "tce_plnt_num": tces.tce_plnt_num, "split": known})
    new = known.isna()
    if new.any():
        out.loc[new, "split"] = make_split(tces[new].reset_index(drop=True), seed).split.to_numpy()
    return out


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_CSV)


LOG_FEATURES = {"tce_period", "tce_duration", "tce_depth", "tce_model_snr"}


def scalar_features(meta: pd.DataFrame, train_meta: pd.DataFrame, names: list[str]) -> np.ndarray:
    """TCE-table numbers as extra model inputs: log10 where the range spans orders of magnitude,
    then standardised with the TRAINING split's mean and std (so val/test never leak into it)."""
    def raw(df):
        cols = [np.log10(df[n].clip(lower=1e-3)) if n in LOG_FEATURES else df[n] for n in names]
        return np.stack(cols, axis=1).astype(np.float64)
    ref = raw(train_meta)
    return ((raw(meta) - ref.mean(0)) / ref.std(0)).astype(np.float32)


class TCEDataset(Dataset):
    """(global_view [1, 2001], local_view [1, 201], scalars [n_features], label) for one split.

    features: TCE-table columns given to the model next to the views, e.g.
    ["tce_depth", "tce_model_snr", "tce_impact", "tce_period", "tce_duration"]. Empty = views only.

    data_dir holds global.npy, local.npy, labels.csv and split.csv (data/processed/ locally,
    the attached dataset on Kaggle). With augment=True, views are randomly flipped in time
    (a transit looks the same backwards), which doubles the effective training data.
    """

    def __init__(self, split: str, data_dir: Path = PROCESSED, augment: bool = False, clip: float | None = None,
                 features: list[str] | None = None):
        data_dir = Path(data_dir)
        labels = pd.read_csv(data_dir / "labels.csv")
        splits = pd.read_csv(data_dir / "split.csv")
        merged = labels.merge(splits, on=["kepid", "tce_plnt_num"], how="left", validate="one_to_one")
        if merged.split.isna().any():
            raise ValueError(f"{merged.split.isna().sum()} views have no split; re-run the split")
        keep = (merged.split == split).to_numpy()

        g = np.load(data_dir / "global.npy")[keep]
        l = np.load(data_dir / "local.npy")[keep]
        if clip is not None:
            g, l = np.clip(g, -clip, clip), np.clip(l, -clip, clip)
        self.g = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)).unsqueeze(1)
        self.l = torch.from_numpy(np.ascontiguousarray(l, dtype=np.float32)).unsqueeze(1)
        self.y = torch.from_numpy(merged.label.to_numpy()[keep].astype(np.float32))
        self.meta = merged[keep].reset_index(drop=True)
        self.features = list(features or [])
        if self.features:
            train_meta = merged[merged.split == "train"]
            self.s = torch.from_numpy(scalar_features(self.meta, train_meta, self.features))
        else:
            self.s = torch.zeros(len(self.y), 0)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        g, l = self.g[i], self.l[i]
        if self.augment and torch.rand(1).item() < 0.5:
            g, l = g.flip(-1), l.flip(-1)
        return g, l, self.s[i], self.y[i]


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="create split.csv")
    sp.add_argument("--subset", type=Path, default=SUBSET_CSV)
    sp.add_argument("--out-dir", type=Path, default=PROCESSED)
    sp.add_argument("--keep", type=Path, default=None, help="an earlier split.csv whose stars keep their split")
    sp.add_argument("--force", action="store_true", help="overwrite an existing split (only before comparing models!)")
    sp.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.cmd == "split":
        path = a.out_dir / "split.csv"
        if path.exists() and not a.force:
            raise SystemExit(f"{path} already exists. Keep it, so all models use the same test set.")
        tces = pd.read_csv(a.subset)
        split = extend_split(tces, pd.read_csv(a.keep), a.seed) if a.keep else make_split(tces, a.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        y = (tces["av_training_set"] == "PC").astype(int)
        summary = pd.DataFrame({"split": split.split, "kepid": tces.kepid, "planet": y}).groupby("split").agg(
            tces=("planet", "size"), stars=("kepid", "nunique"), planet_share=("planet", "mean"))
        print(summary.round(3).to_string())
        overlap = split.groupby("kepid").split.nunique().gt(1).sum()
        print(f"stars in more than one split: {overlap} (must be 0)")
        if a.keep:
            prev = pd.read_csv(a.keep).groupby("kepid").split.first()
            moved = (split.kepid.map(prev).notna() & (split.kepid.map(prev) != split.split)).sum()
            print(f"TCEs whose star changed split vs {a.keep.name}: {moved} (must be 0)")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
