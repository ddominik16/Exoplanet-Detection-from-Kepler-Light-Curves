"""Train/val/test split (by star) and the PyTorch Dataset.

Usage:
    python -m src.dataset split           # creates data/processed/split.csv (once!)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_CSV)


class TCEDataset:  # TODO: subclass torch.utils.data.Dataset
    """Returns (global_view [1, 2001], local_view [1, 201], label) as float tensors."""

    def __init__(self, split: str, augment: bool = False):
        # TODO: load global.npy, local.npy, labels.csv, split.csv; keep rows of this split
        self.augment = augment  # random time flip: view[..., ::-1]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="create data/processed/split.csv")
    sp.add_argument("--force", action="store_true", help="overwrite an existing split (only before comparing models!)")
    sp.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.cmd == "split":
        if SPLIT_CSV.exists() and not a.force:
            raise SystemExit(f"{SPLIT_CSV} already exists. Keep it, so all models use the same test set.")
        tces = pd.read_csv(SUBSET_CSV)
        split = make_split(tces, a.seed)
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_CSV, index=False)
        y = (tces["av_training_set"] == "PC").astype(int)
        summary = pd.DataFrame({"split": split.split, "kepid": tces.kepid, "planet": y}).groupby("split").agg(
            tces=("planet", "size"), stars=("kepid", "nunique"), planet_share=("planet", "mean"))
        print(summary.round(3).to_string())
        overlap = split.groupby("kepid").split.nunique().gt(1).sum()
        print(f"stars in more than one split: {overlap} (must be 0)")
        print(f"saved {SPLIT_CSV}")


if __name__ == "__main__":
    main()
