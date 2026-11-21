"""Train a model with the settings in configs/default.yaml.  [Weeks 5-6]

Usage:  python -m src.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    # TODO: random.seed, np.random.seed, torch.manual_seed
    raise NotImplementedError


def train(cfg: dict) -> None:
    # TODO:
    #  - DataLoaders for train (augment=True) and val
    #  - TwoBranchCNN, Adam(lr=cfg["lr"]), BCEWithLogitsLoss(pos_weight=...)
    #  - each epoch: train loss, val loss, val AUC (sklearn.metrics.roc_auc_score)
    #  - early stopping on val AUC, save best weights to runs/<name>/best.pt
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    cfg = yaml.safe_load(p.parse_args().config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
