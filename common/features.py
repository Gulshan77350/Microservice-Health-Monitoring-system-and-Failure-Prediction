"""
Shared feature engineering — the single source of truth for BOTH training
(model/train.py) and serving (api/app.py).

Training calls `add_features` on a full multi-service time series. Serving
keeps a per-service ring buffer of the last `BUFFER_SIZE` raw observations and
calls `features_for_latest`, which runs the *same* `engineer_service_frame`
function over the buffer and returns the final row. Because every engineered
feature only looks back at most `WINDOW` steps, the last row computed over a
buffer of >= WINDOW rows is identical to the row computed over the full
history. tests/test_features.py asserts this parity.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

import pandas as pd

RAW_FEATURES: list[str] = ["cpu", "memory", "latency", "requests", "error_rate"]
WINDOW: int = 5
# Population std (ddof=0). Pandas defaults to ddof=1 and numpy to ddof=0; the
# original code used one in training and the other in serving, so the model saw
# different feature values online than offline. Pin it here, explicitly.
ROLL_STD_DDOF: int = 0
BUFFER_SIZE: int = WINDOW

INTERACTION_FEATURES: list[str] = ["cpu_x_mem", "lat_x_err", "load_index"]


def feature_columns() -> list[str]:
    cols = list(RAW_FEATURES)
    for col in RAW_FEATURES:
        cols += [f"{col}_roll{WINDOW}_mean", f"{col}_roll{WINDOW}_std", f"{col}_diff1"]
    return cols + INTERACTION_FEATURES


FEATURE_COLUMNS: list[str] = feature_columns()


def engineer_service_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for ONE service's time-ordered observations."""
    out = raw[RAW_FEATURES].astype(float).copy()
    for col in RAW_FEATURES:
        roll = out[col].rolling(window=WINDOW, min_periods=1)
        out[f"{col}_roll{WINDOW}_mean"] = roll.mean()
        out[f"{col}_roll{WINDOW}_std"] = roll.std(ddof=ROLL_STD_DDOF).fillna(0.0)
        out[f"{col}_diff1"] = out[col].diff().fillna(0.0)
    out["cpu_x_mem"] = out["cpu"] * out["memory"]
    out["lat_x_err"] = out["latency"] * out["error_rate"]
    out["load_index"] = out["requests"] * out["latency"] / 1000.0
    return out[FEATURE_COLUMNS]


def add_features(df: pd.DataFrame, service_col: str = "service", time_col: str = "timestamp") -> pd.DataFrame:
    """Add engineered features to a multi-service frame (computed per service, in time order)."""
    df = df.sort_values([service_col, time_col]).reset_index(drop=True)
    parts = []
    for _, group in df.groupby(service_col, sort=False):
        feats = engineer_service_frame(group)
        parts.append(pd.concat([group.drop(columns=RAW_FEATURES), feats], axis=1))
    return pd.concat(parts).sort_values([time_col, service_col]).reset_index(drop=True)


def features_for_latest(history: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    """Single-row feature frame for the most recent observation in `history` (oldest first)."""
    raw = pd.DataFrame(list(history), columns=RAW_FEATURES)
    if raw.empty:
        raise ValueError("history must contain at least one observation")
    return engineer_service_frame(raw).iloc[[-1]].reset_index(drop=True)


class FeatureBuffer:
    """Per-service ring buffer of raw observations used at serving time."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buffers: dict[str, deque] = {}
        self._maxlen = maxlen

    def push(self, service: str, raw: Mapping[str, float]) -> pd.DataFrame:
        buf = self._buffers.setdefault(service, deque(maxlen=self._maxlen))
        buf.append({k: float(raw[k]) for k in RAW_FEATURES})
        return features_for_latest(buf)

    def seed(self, service: str, rows: Iterable[Mapping[str, float]]) -> int:
        """Pre-fill a service buffer (oldest first), e.g. from collector history on startup."""
        buf = self._buffers.setdefault(service, deque(maxlen=self._maxlen))
        for r in rows:
            buf.append({k: float(r[k]) for k in RAW_FEATURES})
        return len(buf)

    def size(self, service: str) -> int:
        return len(self._buffers.get(service, ()))

    def sizes(self) -> dict[str, int]:
        return {svc: len(buf) for svc, buf in self._buffers.items()}
