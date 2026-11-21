"""Final evaluation on the test set (run once per final model).  [Weeks 7-8]

Usage:  python -m src.evaluate --run runs/<name>
"""

from __future__ import annotations

# TODO:
#  - predictions on the test split
#  - ROC AUC + PR curve / average precision, recall at 95% precision
#  - 1,000 bootstrap resamples of the test set -> 95% intervals
#  - confusion matrix at the threshold chosen on the validation set
#  - calibration curve (sklearn.calibration.calibration_curve)
#  - save the most confident mistakes (with their views) for error analysis
#  - figures to reports/figures/
