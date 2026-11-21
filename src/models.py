"""Baselines and the two-branch CNN.  [Weeks 5-6]

Order (see "Models" in the plan):
  1. XGBoost / logistic regression on TCE-table features  -> train_baselines() in train.py
  2. MLP on the flattened views
  3. CNN on the global view only
  4. Two-branch CNN (AstroNet)
"""

from __future__ import annotations


def conv_block(in_ch: int, out_ch: int, pool_size: int):
    """2 x (Conv1d kernel 5, padding 2, ReLU) + MaxPool1d(pool_size, stride=2)."""
    # TODO
    raise NotImplementedError


class TwoBranchCNN:  # TODO: subclass nn.Module
    """
    Global branch: 5 blocks, filters 16, 32, 64, 128, 256, pool 5
    Local branch:  2 blocks, filters 16, 32,             pool 7
    Head: flatten both, concatenate, 4 x Linear(512) + ReLU + Dropout, Linear(1)
    Return logits; use BCEWithLogitsLoss (sigmoid is applied inside the loss).
    """

    def __init__(self, dropout: float = 0.3):
        # TODO
        raise NotImplementedError

    def forward(self, global_view, local_view):
        # TODO
        raise NotImplementedError
