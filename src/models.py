"""The two-branch CNN (AstroNet architecture) and two simpler comparison models.

    two_branch   global + local view, a conv stack each, joined dense head
    global_only  conv stack on the global view only
    mlp          dense layers on both views, no convolutions

n_scalars > 0 adds TCE-table numbers (depth, SNR, ...) to the dense head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, pool_size: int, n_conv: int = 2, kernel: int = 5) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n_conv):
        layers += [nn.Conv1d(in_ch if i == 0 else out_ch, out_ch, kernel, padding=kernel // 2), nn.ReLU()]
    layers.append(nn.MaxPool1d(pool_size, stride=2))
    return nn.Sequential(*layers)


def dense_head(n_in: int, width: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers += [nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout)]
        n_in = width
    layers.append(nn.Linear(n_in, 1))
    return nn.Sequential(*layers)


def global_branch() -> nn.Sequential:
    return nn.Sequential(conv_block(1, 16, 5), conv_block(16, 32, 5), conv_block(32, 64, 5),
                         conv_block(64, 128, 5), conv_block(128, 256, 5))


def local_branch() -> nn.Sequential:
    return nn.Sequential(conv_block(1, 16, 7), conv_block(16, 32, 7))


def _n_out(branch: nn.Module, length: int) -> int:
    with torch.no_grad():
        return branch(torch.zeros(1, 1, length)).numel()


class TwoBranchCNN(nn.Module):
    def __init__(self, global_len: int = 2001, local_len: int = 201, dropout: float = 0.3,
                 dense_width: int = 512, dense_layers: int = 4, n_scalars: int = 0):
        super().__init__()
        self.g, self.l = global_branch(), local_branch()
        n = _n_out(self.g, global_len) + _n_out(self.l, local_len) + n_scalars
        self.head = dense_head(n, dense_width, dense_layers, dropout)

    def forward(self, g: torch.Tensor, l: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.g(g).flatten(1), self.l(l).flatten(1), s], dim=1)).squeeze(1)


class GlobalOnlyCNN(nn.Module):
    def __init__(self, global_len: int = 2001, local_len: int = 201, dropout: float = 0.3,
                 dense_width: int = 512, dense_layers: int = 4, n_scalars: int = 0):
        super().__init__()
        self.g = global_branch()
        self.head = dense_head(_n_out(self.g, global_len) + n_scalars, dense_width, dense_layers, dropout)

    def forward(self, g: torch.Tensor, l: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.g(g).flatten(1), s], dim=1)).squeeze(1)


class MLP(nn.Module):
    def __init__(self, global_len: int = 2001, local_len: int = 201, dropout: float = 0.3,
                 dense_width: int = 512, dense_layers: int = 3, n_scalars: int = 0):
        super().__init__()
        self.head = dense_head(global_len + local_len + n_scalars, dense_width, dense_layers, dropout)

    def forward(self, g: torch.Tensor, l: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([g.flatten(1), l.flatten(1), s], dim=1)).squeeze(1)


MODELS = {"two_branch": TwoBranchCNN, "global_only": GlobalOnlyCNN, "mlp": MLP}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
