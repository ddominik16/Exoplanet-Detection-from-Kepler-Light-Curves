# Exo-Det: exoplanet detection from Kepler light curves

A two-branch 1D CNN (after Shallue & Vanderburg 2018, "AstroNet") that classifies Kepler
Threshold Crossing Events as planet candidates or false positives.

## Setup (Windows, Anaconda)

Open this folder in VS Code (File > Open Folder), then in the terminal (Terminal > New Terminal):

```
conda create -n exodet python=3.11 -y
conda activate exodet
pip install -r requirements.txt
```

Install PyTorch with the command from https://pytorch.org/get-started/locally/
(choose Windows / Pip / Python / CPU, or CUDA if you have an NVIDIA GPU).

In VS Code: Ctrl+Shift+P > "Python: Select Interpreter" > `exodet`.
In a notebook: top-right "Select Kernel" > `exodet`.

If `conda` isn't recognised in the VS Code terminal, use the "Anaconda Prompt" once and run `conda init powershell`, then restart VS Code.

## Workflow

| Where | What |
| --- | --- |
| `src/*.py` | The pipeline: code that must run the same way every time. Run from the project root with `python -m src.<module>`. |
| `notebooks/` | Exploration, sanity plots and figures. Import from `src` instead of copying code. |
| `configs/` | Hyperparameters, one YAML per experiment. |
| `data/raw/` | TCE table and light curves (`.npz`, one per star). Not in git. |
| `data/processed/` | Global/local views and the train/val/test split. Not in git. |

## Steps

```
python -m src.download tce                       # labels (DR24 TCE table)
python -m src.download subset --n 2000           # balanced subset to start with
python -m src.download lightcurves --workers 8   # light curves for the subset
python -m src.download check --delete            # remove broken light curves, then re-run the line above
python -m src.preprocess                         # global/local views -> data/processed/
python -m src.dataset split                      # train/val/test split (once!)
python -m src.train                              # train (configs/default.yaml), results in runs/<name>/
python -m src.train --config configs/views_plus_features.yaml   # views + TCE-table numbers
python -m src.evaluate --runs <run> [<run> ...] [--vs <run> ...] --data-dir data/processed_full   # score / compare (val)
python -m src.evaluate ... --split test --i-am-sure            # FINAL model only, once
```

Disk: each star is ~1 MB as `.npz` (raw FITS are deleted after conversion), so the 2,000-TCE subset
needs roughly 2 GB. The full set (~12,000 stars) needs ~12 GB.

## AstroNet's pre-computed views (all ~15,700 TCEs)

Instead of downloading every light curve, use the views published with the AstroNet paper (~106 MB):
https://storage.googleapis.com/kepler-ml/astronet/data/kepler/kepler-tce-tfrecords-20180220.tar.gz

```
mkdir data\raw\astronet
tar -xvf $HOME\Downloads\kepler-tce-tfrecords-20180220.tar.gz -C data\raw\astronet
python -m src.convert_astronet --inspect
python -m src.convert_astronet
python -m src.dataset split --subset data/processed_astronet/labels.csv --out-dir data/processed_astronet --keep data/processed/split.csv
python -m src.convert_astronet --only data/raw/subset.csv --out-dir data/processed_astronet_2000
copy data\processed\split.csv data\processed_astronet_2000\split.csv
```

`data/processed_astronet/` is the full dataset (your 2,000-signal stars keep their split).
`data/processed_astronet_2000/` holds AstroNet's views of exactly your 2,000 signals with your split, to compare
their preprocessing with yours. Train on either with `--data-dir`.

## Training on Kaggle

The same `src/train.py` runs on Kaggle; it finds the data under `/kaggle/input/` and writes to `/kaggle/working/runs/`.

One-time setup:
1. **Dataset:** kaggle.com > Datasets > New Dataset, upload `global.npy`, `local.npy`, `labels.csv`, `split.csv`
   from `data/processed/`, name it e.g. `exo-det-views`, keep it private. After re-running preprocessing,
   upload a new version of the same dataset.
2. **GitHub token:** GitHub > Settings > Developer settings > Fine-grained tokens > Generate.
   Repository access: only `exo-det`; permissions: Contents read-only.
3. **Notebook:** kaggle.com > Code > New Notebook > File > Import Notebook > `kaggle/train_on_kaggle.ipynb`.
   Then: Add Input > your dataset; Add-ons > Secrets > add `GITHUB_TOKEN`; Settings > Internet on, Accelerator GPU.
   Put your GitHub username in the first code cell.

Each time: push your code, run the notebook (or Save Version to run it in the background), then download `runs.zip`
from Output and extract it into the project folder (so you get `Exo-Det/runs/<run name>/`, not `runs/runs/`).
