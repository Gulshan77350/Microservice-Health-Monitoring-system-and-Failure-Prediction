"""
Risk banding and alert decision shared by the prediction API and tests.

Two thresholds are tuned on the validation split (see model/train.py):
  * cost_optimal  — minimises 10*FN + 1*FP. Used for the ALERT decision
                    (a missed outage is assumed to cost 10x a false page).
  * max_f1        — maximises F1. Used for the displayed HIGH band.

Bands are derived from metadata so the API never hard-codes a number:
  LOW    : p <  low
  MEDIUM : low  <= p < high
  HIGH   : p >= high
where low = min(cost_optimal, max_f1) and high = max(cost_optimal, max_f1).
"""

from __future__ import annotations

from collections.abc import Mapping

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


def risk_bands(thresholds: Mapping[str, float]) -> dict[str, list[float]]:
    low = min(thresholds["cost_optimal"], thresholds["max_f1"])
    high = max(thresholds["cost_optimal"], thresholds["max_f1"])
    return {"LOW": [0.0, low], "MEDIUM": [low, high], "HIGH": [high, 1.0]}


def classify_risk(probability: float, bands: Mapping[str, list[float]]) -> str:
    if probability >= bands["HIGH"][0]:
        return "HIGH"
    if probability >= bands["MEDIUM"][0]:
        return "MEDIUM"
    return "LOW"


def should_alert(probability: float, thresholds: Mapping[str, float]) -> bool:
    return probability >= thresholds["cost_optimal"]


def root_cause(m: Mapping[str, float]) -> str:
    """Rule-based hint shown next to a prediction. Not learned; documented as a heuristic."""
    if m["memory"] > 90:
        return "Memory Saturation"
    if m["cpu"] > 90:
        return "CPU Overload"
    if m["error_rate"] > 10:
        return "Error Spike"
    if m["latency"] > 1000:
        return "Latency Surge"
    return "Normal Operation"
