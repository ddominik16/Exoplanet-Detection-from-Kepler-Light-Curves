"""PyTorch Dataset and the train/val/test split (by star!).  [Weeks 3-4]"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def make_split(labels: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Return a 'train' / 'val' / 'test' label per row, 80/10/10, grouped by kepid.

    All TCEs of one star must land in the same split, otherwise the test score is inflated.
    Use sklearn.model_selection.StratifiedGroupKFold (groups=labels.kepid, y=labels.label).
    Save the result to data/processed/split.csv and never regenerate it after you start comparing models.
    """
    # TODO
    raise NotImplementedError


class TCEDataset:  # TODO: subclass torch.utils.data.Dataset
    """Returns (global_view [1, 2001], local_view [1, 201], label) as float tensors."""

    def __init__(self, split: str, augment: bool = False):
        # TODO: load global.npy, local.npy, labels.csv, split.csv; keep rows of this split
        self.augment = augment  # random time flip: view[..., ::-1]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError
