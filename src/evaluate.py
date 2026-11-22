"""Score saved models on any dataset and split; compare groups of models fairly.

Examples:
    # validation score of the full-data views-only models (3 seeds, averaged = ensemble)
    python -m src.evaluate --runs full_views full_views_seed1 full_views_seed2 --data-dir data/processed_full

    # fair comparison: 2,000-signal models vs full-data models, on the same 1,574 validation signals
    python -m src.evaluate --runs two_branch two_branch_seed1 two_branch_seed2 \\
        --vs full_views full_views_seed1 full_views_seed2 --data-dir data/processed_full --name small_vs_full

    # FINAL, once: the chosen model on the test set
    python -m src.evaluate --runs full_views full_views_seed1 full_views_seed2 full_features full_features_seed1 \\
        full_features_seed2 --data-dir data/processed_full --split test --i-am-sure --name final_test

Each group's models are averaged (an ensemble). Reports AUC and AP with 95% bootstrap intervals, and accuracy,
precision and recall at a threshold chosen on the VALIDATION split (never on the split being scored).
Results go to runs/_eval/<name>/. Every test-set evaluation is logged in runs/_eval/TEST_LOG.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import PROCESSED, TCEDataset
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
EVAL = RUNS / "_eval"
KEY = ["kepid", "tce_plnt_num"]
CANDIDATE_DATA = [ROOT / "data" / "processed", ROOT / "data" / "processed_full"]


def training_data_dir(run_dir: Path, cfg: dict) -> Path:
    """Folder the run was trained on: recorded in config.yaml, or recognised from its validation predictions."""
    if cfg.get("data_dir_local"):
        return ROOT / cfg["data_dir_local"]
    val_keys = set(map(tuple, pd.read_csv(run_dir / "val_predictions.csv")[KEY].to_numpy()))
    for d in CANDIDATE_DATA:
        if (d / "split.csv").exists():
            s = pd.read_csv(d / "split.csv")
            if set(map(tuple, s[s.split == "val"][KEY].to_numpy())) == val_keys:
                return d
    raise SystemExit(f"Can't tell which data {run_dir.name} was trained on; add 'data_dir_local:' to its config.yaml")


def predict_run(run: str, data_dir: Path, split: str, device: str) -> pd.DataFrame:
    run_dir = RUNS / run
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    features = cfg.get("features") or []
    stats_dir = training_data_dir(run_dir, cfg) if features else None
    ds = TCEDataset(split, data_dir, augment=False, clip=cfg.get("clip"), features=features, stats_dir=stats_dir)
    model = build_model(cfg["model"], global_len=ds.g.shape[-1], local_len=ds.l.shape[-1], dropout=cfg["dropout"],
                        dense_width=cfg["dense_width"], dense_layers=cfg["dense_layers"], n_scalars=len(features))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for g, l, s, _ in DataLoader(ds, batch_size=512):
            probs.append(torch.sigmoid(model(g.to(device), l.to(device), s.to(device))).cpu().numpy())
    return ds.meta.assign(prob=np.concatenate(probs))


def ensemble(runs: list[str], data_dir: Path, split: str, device: str) -> pd.DataFrame:
    out = None
    for r in runs:
        p = predict_run(r, data_dir, split, device).rename(columns={"prob": r})
        out = p if out is None else out.merge(p[KEY + [r]], on=KEY, validate="one_to_one")
    out["prob"] = out[runs].mean(axis=1)
    return out


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold with the highest F1 score (balance of precision and recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def bootstrap(y, p, q=None, n=2000, seed=0) -> dict:
    """95% intervals for AUC and AP of p, and for the AUC/AP difference p - q on the same resamples."""
    rng = np.random.default_rng(seed)
    a, b, da, db = [], [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i])); b.append(average_precision_score(y[i], p[i]))
        if q is not None:
            da.append(a[-1] - roc_auc_score(y[i], q[i])); db.append(b[-1] - average_precision_score(y[i], q[i]))
    ci = lambda v: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
    out = {"auc_ci": ci(a), "ap_ci": ci(b)}
    if q is not None:
        out |= {"auc_diff_ci": ci(da), "ap_diff_ci": ci(db)}
    return out


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"n": int(len(y)), "planets": int(y.sum()), "auc": round(roc_auc_score(y, p), 4),
            "ap": round(average_precision_score(y, p), 4), "threshold": round(thr, 4),
            "accuracy": round((tp + tn) / len(y), 4), "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders under runs/, averaged")
    p.add_argument("--vs", nargs="+", default=None, help="second group of runs to compare against")
    p.add_argument("--data-dir", type=Path, default=PROCESSED)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--name", default=None)
    p.add_argument("--i-am-sure", action="store_true", help="required for --split test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    if a.split == "test":
        if not a.i_am_sure:
            raise SystemExit("The test set is for the FINAL model, scored once. Decide the model on validation "
                             "results first, then add --i-am-sure.")
        log = EVAL / "TEST_LOG.txt"
        if log.exists():
            print("Earlier test-set evaluations:\n" + log.read_text())

    name = a.name or ("_".join(a.runs[:1]) + f"_{a.split}")
    out_dir = EVAL / name
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {"A": a.runs} | ({"B": a.vs} if a.vs else {})

    report = {"data_dir": str(a.data_dir), "split": a.split, "groups": groups, "results": {}}
    scored = {}
    for g, runs in groups.items():
        val = ensemble(runs, a.data_dir, "val", a.device)
        thr = best_threshold(val.label.to_numpy(), val.prob.to_numpy())
        df = val if a.split == "val" else ensemble(runs, a.data_dir, a.split, a.device)
        y, pr = df.label.to_numpy(), df.prob.to_numpy()
        res = metrics(y, pr, thr) | bootstrap(y, pr)
        res["single_runs_auc"] = {r: round(roc_auc_score(y, df[r]), 4) for r in runs}
        report["results"][g] = res
        scored[g] = df
        df.to_csv(out_dir / f"predictions_{g}.csv", index=False)

    if a.vs:
        m = scored["A"][KEY + ["label", "prob"]].merge(scored["B"][KEY + ["prob"]], on=KEY, suffixes=("_A", "_B"))
        report["A_minus_B"] = bootstrap(m.label.to_numpy(), m.prob_A.to_numpy(), m.prob_B.to_numpy())

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{a.split.upper()} split of {a.data_dir}  ({report['results']['A']['n']} TCEs, "
          f"{report['results']['A']['planets']} planets)\n")
    for g, res in report["results"].items():
        print(f"group {g}: {', '.join(groups[g])}")
        print(f"  single runs AUC: {res['single_runs_auc']}")
        print(f"  ensemble AUC {res['auc']:.4f} {res['auc_ci']}   AP {res['ap']:.4f} {res['ap_ci']}")
        print(f"  at threshold {res['threshold']:.3f} (chosen on val): accuracy {res['accuracy']:.3f}, "
              f"precision {res['precision']:.3f}, recall {res['recall']:.3f}  "
              f"[found {res['tp']} of {res['planets']} planets, {res['fp']} false alarms]\n")
    if a.vs:
        d = report["A_minus_B"]
        print(f"A - B:  AUC difference 95% CI {d['auc_diff_ci']}   AP difference 95% CI {d['ap_diff_ci']}")
        print("        (an interval that contains 0 means no clear difference)")
    if a.split == "test":
        with open(EVAL / "TEST_LOG.txt", "a") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  {name}  {' '.join(a.runs)}  AUC {report['results']['A']['auc']}\n")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
