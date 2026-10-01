"""
drift.py — lightweight production drift monitoring.

WHY THIS EXISTS
----------------
The whole premise of this project is "predict failure before it happens."
But a model trained on one data distribution silently degrades when the
live input distribution shifts (covariate shift) -- e.g. if a new service
type is added with different baseline CPU usage, or traffic patterns
change after a product launch. Without checking for this, the model could
be confidently wrong and nobody would know until failures it should have
caught start slipping through.

This module implements Population Stability Index (PSI), a standard,
lightweight drift metric used in real ML monitoring systems (common in
credit risk and fraud models). It buckets a reference (training)
distribution and a live distribution into the same bins and measures how
much probability mass has shifted between them.

PSI interpretation (industry-standard thresholds):
  < 0.1  -> no significant shift
  0.1-0.25 -> moderate shift, worth investigating
  > 0.25 -> significant shift, model likely needs retraining

This is intentionally simple (no external dependencies beyond numpy/pandas)
so it can run on every request batch without adding latency.
"""

import json
import numpy as np
import pandas as pd
from collections import deque

FEATURE_NAMES = ["cpu", "memory", "latency", "requests", "error_rate"]
N_BINS = 10
LIVE_WINDOW_SIZE = 500  # rolling window of recent live predictions

# Rolling buffer of recent live feature values, populated as requests arrive
_live_buffer = {feat: deque(maxlen=LIVE_WINDOW_SIZE) for feat in FEATURE_NAMES}

_reference_bins = None  # populated by load_reference()


def load_reference(training_csv_path="final_dataset_v2.csv"):
    """
    Build reference bin edges + reference distribution from the training
    data. Call this once at API startup.
    """
    global _reference_bins
    df = pd.read_csv(training_csv_path)

    _reference_bins = {}
    for feat in FEATURE_NAMES:
        values = df[feat].values
        # Quantile-based bins so each reference bin has ~equal mass —
        # more robust than fixed-width bins when features have skewed
        # distributions (e.g. requests, latency).
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            edges = np.linspace(values.min(), values.max(), N_BINS + 1)
        ref_counts, _ = np.histogram(values, bins=edges)
        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        _reference_bins[feat] = {
            "edges": edges,
            "ref_dist": ref_dist,
        }
    return _reference_bins


def record_live_sample(metrics: dict):
    """
    Call this on every prediction request to feed the rolling drift window.
    `metrics` is a dict with keys matching FEATURE_NAMES.
    """
    for feat in FEATURE_NAMES:
        if feat in metrics:
            _live_buffer[feat].append(metrics[feat])


def _psi_for_feature(feat: str) -> float:
    if _reference_bins is None or feat not in _reference_bins:
        return 0.0
    live_values = np.array(_live_buffer[feat])
    if len(live_values) < 200:
        return 0.0  # not enough live data yet for a statistically meaningful PSI
        # (10 quantile bins need a reasonable sample size per bin to avoid
        # sampling noise alone triggering false drift alarms)

    edges = _reference_bins[feat]["edges"]
    ref_dist = _reference_bins[feat]["ref_dist"]

    live_counts, _ = np.histogram(live_values, bins=edges)
    live_dist = live_counts / max(live_counts.sum(), 1)

    # Avoid log(0) / div-by-0 with a small epsilon
    eps = 1e-4
    ref_dist = np.clip(ref_dist, eps, None)
    live_dist = np.clip(live_dist, eps, None)

    psi = np.sum((live_dist - ref_dist) * np.log(live_dist / ref_dist))
    return float(psi)


def compute_drift_report() -> dict:
    """
    Returns a per-feature PSI report plus an overall status.
    """
    if _reference_bins is None:
        return {
            "status": "not_initialized",
            "note": "Call load_reference() at startup before drift checks are available.",
        }

    per_feature = {}
    max_psi = 0.0
    for feat in FEATURE_NAMES:
        psi = _psi_for_feature(feat)
        n_live = len(_live_buffer[feat])
        if psi < 0.1:
            level = "stable"
        elif psi < 0.25:
            level = "moderate_shift"
        else:
            level = "significant_shift"
        per_feature[feat] = {
            "psi": round(psi, 4),
            "level": level,
            "live_samples_in_window": n_live,
        }
        max_psi = max(max_psi, psi)

    if max_psi < 0.1:
        overall = "stable"
    elif max_psi < 0.25:
        overall = "moderate_shift"
    else:
        overall = "significant_shift"

    return {
        "status": "ok",
        "overall_status": overall,
        "max_psi": round(max_psi, 4),
        "window_size": LIVE_WINDOW_SIZE,
        "per_feature": per_feature,
        "interpretation": {
            "stable": "< 0.1 — no significant shift",
            "moderate_shift": "0.1-0.25 — worth investigating",
            "significant_shift": "> 0.25 — model likely needs retraining",
        },
    }
