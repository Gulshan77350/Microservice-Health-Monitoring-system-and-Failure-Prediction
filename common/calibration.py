"""
Probability calibration that can be applied without pickled sklearn objects.

Training fits sklearn's CalibratedClassifierCV on the validation split, then
exports its parameters to JSON via `export_calibrator`. Serving reloads those
parameters and calls `apply_calibration`. model/train.py asserts that this
reproduces CalibratedClassifierCV.predict_proba to within 1e-9.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def apply_calibration(raw_proba: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    raw_proba = np.asarray(raw_proba, dtype=float)
    method = params.get("method", "none")
    if method == "none":
        return raw_proba
    if method == "sigmoid":
        # sklearn _SigmoidCalibration: p = 1 / (1 + exp(a * f + b))
        return 1.0 / (1.0 + np.exp(params["a"] * raw_proba + params["b"]))
    if method == "isotonic":
        return np.interp(raw_proba, params["x"], params["y"])
    raise ValueError(f"Unknown calibration method: {method}")


def export_calibrator(calibrated_clf) -> dict[str, Any]:
    """Extract the fitted calibrator from a single-fold CalibratedClassifierCV."""
    if len(calibrated_clf.calibrated_classifiers_) != 1:
        raise ValueError("Expected exactly one calibrated classifier (prefit/frozen estimator)")
    cal = calibrated_clf.calibrated_classifiers_[0].calibrators[0]
    method = calibrated_clf.method
    if method == "sigmoid":
        return {"method": "sigmoid", "a": float(cal.a_), "b": float(cal.b_)}
    if method == "isotonic":
        return {
            "method": "isotonic",
            "x": [float(v) for v in cal.X_thresholds_],
            "y": [float(v) for v in cal.y_thresholds_],
        }
    raise ValueError(f"Unsupported calibration method: {method}")
