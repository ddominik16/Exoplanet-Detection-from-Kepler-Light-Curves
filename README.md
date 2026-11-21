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
python -m src.download lightcurves --workers 4   # light curves for the subset
python -m src.preprocess                         # views          (to do)
python -m src.train --config configs/default.yaml                # (to do)
python -m src.evaluate --run runs/<name>                         # (to do)
```

Disk: each star is ~1 MB as `.npz` (raw FITS are deleted after conversion), so the 2,000-TCE subset
needs roughly 2 GB. The full set (~12,000 stars) needs ~12 GB.
