"""
Population Stability Index (PSI) for input-drift monitoring.

The reference distribution is built from the TRAINING split at training time
and saved as artifacts/v{N}/drift_reference.json, so the API does not need the
raw dataset at runtime.

PSI = sum_i (live_i - ref_i) * ln(live_i / ref_i)
Conventional reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant.

Bin edges are training quantiles with the outermost edges opened to -inf/+inf.
(The original implementation used the training min/max as outer edges, and
np.histogram silently drops values outside them — so a live value far above
anything seen in training did not count toward drift at all.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

N_BINS = 10
EPS = 1e-4
MIN_LIVE_SAMPLES = 200
MODERATE = 0.1
SIGNIFICANT = 0.25


def build_reference(columns: Mapping[str, Sequence[float]], n_bins: int = N_BINS) -> dict:
    ref = {}
    for feat, values in columns.items():
        values = np.asarray(values, dtype=float)
        inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))[1:-1])
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        counts, _ = np.histogram(values, bins=edges)
        ref[feat] = {
            "inner_edges": inner.tolist(),
            "ref_dist": (counts / max(counts.sum(), 1)).tolist(),
        }
    return ref


def _edges(entry: Mapping) -> np.ndarray:
    return np.concatenate([[-np.inf], np.asarray(entry["inner_edges"], dtype=float), [np.inf]])


def psi(ref_dist: Sequence[float], live_values: Sequence[float], edges: np.ndarray) -> float:
    live_counts, _ = np.histogram(np.asarray(live_values, dtype=float), bins=edges)
    live = np.clip(live_counts / max(live_counts.sum(), 1), EPS, None)
    ref = np.clip(np.asarray(ref_dist, dtype=float), EPS, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


def level(value: float) -> str:
    if value < MODERATE:
        return "stable"
    if value < SIGNIFICANT:
        return "moderate_shift"
    return "significant_shift"


def drift_report(reference: Mapping, live: Mapping[str, Sequence[float]]) -> dict:
    per_feature = {}
    max_psi = 0.0
    ready = True
    for feat, entry in reference.items():
        values = list(live.get(feat, []))
        if len(values) < MIN_LIVE_SAMPLES:
            ready = False
            per_feature[feat] = {"psi": None, "level": "insufficient_data", "live_samples": len(values)}
            continue
        value = psi(entry["ref_dist"], values, _edges(entry))
        max_psi = max(max_psi, value)
        per_feature[feat] = {"psi": round(value, 4), "level": level(value), "live_samples": len(values)}
    return {
        "status": "ok" if ready else "warming_up",
        "overall_status": level(max_psi) if ready else "insufficient_data",
        "max_psi": round(max_psi, 4) if ready else None,
        "min_live_samples": MIN_LIVE_SAMPLES,
        "per_feature": per_feature,
        "retrain_recommended": bool(ready and max_psi > SIGNIFICANT),
    }
