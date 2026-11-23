# Exo-Det: finding exoplanets in Kepler light curves

A convolutional neural network that decides whether a periodic dip in a star's brightness, recorded by NASA's
Kepler space telescope, is a **transiting planet** or a **false positive** (an eclipsing binary star, stellar
variability or instrument noise). It reimplements and extends the AstroNet approach of
[Shallue & Vanderburg (2018)](https://arxiv.org/abs/1712.05044) in PyTorch, with its own data pipeline.

## What it does

1. **Data:** downloads the 15,737 human-labelled signals (TCEs) of the Kepler DR24 catalogue (3,600 planet
   candidates, 9,596 astrophysical false positives, 2,541 non-transiting phenomena) and the four-year light curves of their
   9,865 stars from NASA's MAST archive.
2. **Preprocessing:** removes slow stellar variability, folds each light curve on the signal's period and turns it
   into two fixed-size views: a **global view** of the whole orbit (2,001 bins) and a **local view** zoomed on the
   transit (201 bins), both scaled so the dip reaches −1.
3. **Split:** 80 / 10 / 10 % train / validation / test, **grouped by star**, so no star appears in two splits.
4. **Model:** a two-branch 1D CNN (one convolution stack per view, joined dense layers, ~10M parameters), optionally
   with five numbers from the TCE table (depth, signal-to-noise, impact, period, duration) added to the dense layers.
5. **Evaluation:** models are compared on validation with bootstrap confidence intervals; the final model was
   chosen on validation and scored **once** on the test set.

## How good is it

Final model: the average of six networks (three seeds each of views-only and views + numbers), trained on all
15,730 signals with light curves. **Test set, 1,563 signals never used for any decision:**

| Metric | Test |
|---|---|
| **ROC AUC** | **0.984** (95% CI 0.978–0.989) |
| Average precision | 0.941 (0.917–0.960) |
| Accuracy | 95.0% |
| Planet candidates found (recall) | 90.7% (322 of 355) |
| Precision | 87.7% |
| False alarms | 3.7% of non-planets (45 of 1,208) |

Validation (1,574 signals) was 0.986 AUC, so the model generalises. AstroNet reported 0.988 AUC on its own
test split, so the two are comparable but not measured on the same signals.

## Findings

All comparisons on the validation set, averaged over three seeds, unless stated otherwise.

| Model | Validation AUC |
|---|---|
| Gradient-boosted trees on the 5 TCE numbers (no light curve) | 0.951 |
| CNN, views only | 0.984 |
| CNN, views + 5 numbers | 0.984 |
| **Ensemble of both** | **0.986** |

- **The light curve's shape adds a lot.** The CNN beats the 5-number baseline by +0.033 AUC (95% CI +0.025 to
  +0.042) and is right more often (93.8% against 90.1% accuracy).
- **The zoomed local view is essential.** On a 2,000-signal sample, a CNN on the global view alone (0.955) did no
  better than a plain dense network (0.957); adding the local view raised it to 0.974. The transit's shape (U-shaped
  for planets, V-shaped for grazing binaries) is only visible there.
- **More data helped and made training reliable.** On the same 1,574 validation signals, training on all 15,730
  instead of 2,000 signals raised AUC from 0.976 to 0.984 (+0.008, 95% CI +0.005 to +0.012), and the spread between
  seeds fell from 0.009 to 0.0002.
- **Normalisation hides the depth of the dip.** Every view is scaled to −1, so a 0.01% and a 4% dip look the same
  to the network. With 2,000 signals this made deep, clear signals the most common mistakes. With the full data the
  views-only CNN learned to recognise deep planets by shape: it finds 80% of deep (>0.5%) planets, against 68% for
  the model with the TCE numbers and 52% for the baseline, while the numbers help reject the deepest binaries.
- **Adding the TCE numbers causes shortcut learning.** Six of seven views + numbers runs settled at 0.976–0.977,
  below views-only, whatever the patience or learning rate: they lean on the numbers and stop improving on the
  shapes. One run reached 0.988 on validation but only 0.980 on test, so it was luck, not a better model.
- **Averaging both model types works best.** They make different mistakes, and the ensemble beats every single
  network on the test set (0.984 against at most 0.982).

## Running it

```
pip install -r requirements.txt
python -m src.download tce
python -m src.download subset --all
python -m src.download lightcurves --subset data/raw/subset_full.csv
python -m src.preprocess --subset data/raw/subset_full.csv --out-dir data/processed_full
python -m src.dataset split --subset data/raw/subset_full.csv --out-dir data/processed_full --keep data/processed/split.csv
python -m src.train --data-dir data/processed_full --set name=full_views
python -m src.evaluate --runs full_views --data-dir data/processed_full
```

Training was run on Kaggle GPUs with `kaggle/train_on_kaggle.ipynb`. The notebooks in `notebooks/` explore the
data, check the preprocessed views and analyse the model's mistakes.
