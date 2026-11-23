"""Train a model on the preprocessed views (locally or on Kaggle).

    python -m src.train --config configs/default.yaml --data-dir data/processed_full --set name=views seed=1

Writes runs/<name>/: config.yaml, best.pt (best validation AUC), history.csv, val_predictions.csv,
summary.json, curves.png. Uses only the train and validation splits.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import PROCESSED, TCEDataset
from src.models import build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT, KAGGLE_WORKING = Path("/kaggle/input"), Path("/kaggle/working")


def default_data_dir() -> Path:
    if KAGGLE_INPUT.exists():
        found = sorted(KAGGLE_INPUT.rglob("global.npy"))
        if found:
            return found[0].parent
    return PROCESSED


def default_out_dir() -> Path:
    return KAGGLE_WORKING / "runs" if KAGGLE_WORKING.exists() else ROOT / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for g, l, s, y in loader:
            probs.append(torch.sigmoid(model(g.to(device), l.to(device), s.to(device))).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train(cfg: dict, data_dir: Path, out_dir: Path, device: str) -> dict:
    set_seed(cfg["seed"])
    run_dir = out_dir / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = dict(cfg)
    if not str(data_dir).startswith(str(KAGGLE_INPUT)):
        try:
            saved["data_dir_local"] = str(Path(data_dir).resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            pass
    else:
        saved["data_dir_kaggle"] = str(data_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))

    features = cfg.get("features") or []
    train_ds = TCEDataset("train", data_dir, augment=cfg["augment"], clip=cfg["clip"], features=features)
    val_ds = TCEDataset("val", data_dir, augment=False, clip=cfg["clip"], features=features)
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=256)

    model = build_model(cfg["model"], global_len=train_ds.g.shape[-1], local_len=train_ds.l.shape[-1],
                        dropout=cfg["dropout"], dense_width=cfg["dense_width"], dense_layers=cfg["dense_layers"],
                        n_scalars=len(features)).to(device)
    pos = train_ds.y.sum().item()
    pos_weight = (len(train_ds) - pos) / pos if cfg["pos_weight"] == "auto" else float(cfg["pos_weight"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    print(f"model {cfg['model']}: {count_parameters(model):,} parameters  |  device {device}")
    print(f"train {len(train_ds)}  val {len(val_ds)}  |  planet share {pos / len(train_ds):.1%}  pos_weight {pos_weight:.2f}")
    print(f"extra features: {', '.join(features) if features else 'none (views only)'}")
    print(f"data {data_dir}\n")
    print(f"{'epoch':>5} {'train loss':>10} {'val loss':>9} {'val AUC':>8} {'val AP':>7} {'time':>6}")

    history, best_auc, best_epoch = [], -1.0, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for g, l, s, y in train_dl:
            g, l, s, y = g.to(device), l.to(device), s.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(g, l, s), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)

        p, y = predict(model, val_dl, device)
        val_loss = torch.nn.functional.binary_cross_entropy(
            torch.tensor(p).clamp(1e-7, 1 - 1e-7), torch.tensor(y)).item()
        row = {"epoch": epoch, "train_loss": total / len(train_ds), "val_loss": val_loss,
               "val_auc": roc_auc_score(y, p), "val_ap": average_precision_score(y, p)}
        history.append(row)
        improved = row["val_auc"] > best_auc
        if improved:
            best_auc, best_epoch = row["val_auc"], epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"{epoch:>5} {row['train_loss']:>10.4f} {val_loss:>9.4f} {row['val_auc']:>8.4f} {row['val_ap']:>7.4f} "
              f"{time.time() - t0:>5.1f}s{'  *' if improved else ''}")
        if epoch - best_epoch >= cfg["patience"]:
            print(f"no improvement for {cfg['patience']} epochs, stopping")
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    p, y = predict(model, val_dl, device)
    val_ds.meta.assign(prob=p).to_csv(run_dir / "val_predictions.csv", index=False)
    summary = {"name": cfg["name"], "model": cfg["model"], "best_epoch": best_epoch,
               "val_auc": roc_auc_score(y, p), "val_ap": average_precision_score(y, p),
               "parameters": count_parameters(model), "features": features}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_history(pd.DataFrame(history), best_epoch, run_dir / "curves.png")
    print(f"\nbest epoch {best_epoch}: val AUC {summary['val_auc']:.4f}, val AP {summary['val_ap']:.4f}")
    print(f"saved to {run_dir}")
    return summary


def plot_history(h: pd.DataFrame, best_epoch: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation")
    ax[0].set(xlabel="epoch", ylabel="loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_auc, label="AUC"); ax[1].plot(h.epoch, h.val_ap, label="average precision")
    ax[1].set(xlabel="epoch", ylabel="validation score"); ax[1].legend()
    for a in ax:
        a.axvline(best_epoch, color="gray", ls="--", lw=1)
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        value = yaml.safe_load(value)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        out[key] = value
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    overrides = parse_overrides(a.set)
    cfg.update(overrides)
    if "model" in overrides and "name" not in overrides:
        cfg["name"] = cfg["model"]
    train(cfg, a.data_dir or default_data_dir(), a.out_dir or default_out_dir(), a.device)


if __name__ == "__main__":
    main()
