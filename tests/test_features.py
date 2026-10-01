"""Train/serve feature parity: streaming through FeatureBuffer must equal batch engineering."""

import numpy as np
import pandas as pd
import pytest

from common.features import (
    FEATURE_COLUMNS,
    RAW_FEATURES,
    WINDOW,
    FeatureBuffer,
    add_features,
    engineer_service_frame,
)


def _series(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cpu": rng.uniform(5, 100, n),
            "memory": rng.uniform(10, 100, n),
            "latency": rng.uniform(30, 3000, n),
            "requests": rng.integers(50, 10000, n),
            "error_rate": rng.uniform(0, 40, n),
        }
    )


def test_streaming_matches_batch():
    raw = _series(40)
    batch = engineer_service_frame(raw)
    buf = FeatureBuffer()
    streamed = pd.concat([buf.push("svc", row) for row in raw.to_dict("records")], ignore_index=True)
    pd.testing.assert_frame_equal(streamed, batch, check_exact=False, rtol=1e-12, atol=1e-9)


def test_multi_service_training_matches_per_service_streaming():
    frames = []
    for i, svc in enumerate(["payment", "order", "notification"]):
        s = _series(25, seed=i)
        s["service"] = svc
        s["timestamp"] = pd.date_range("2026-01-01", periods=25, freq="3min")
        frames.append(s)
    full = add_features(pd.concat(frames, ignore_index=True))

    buf = FeatureBuffer()
    for svc in ["payment", "order", "notification"]:
        rows = full[full.service == svc].sort_values("timestamp")
        streamed = pd.concat([buf.push(svc, r) for r in rows[RAW_FEATURES].to_dict("records")], ignore_index=True)
        pd.testing.assert_frame_equal(
            streamed, rows[FEATURE_COLUMNS].reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-9
        )


def test_rolling_std_is_population_std():
    raw = _series(WINDOW)
    feats = engineer_service_frame(raw)
    assert feats["cpu_roll5_std"].iloc[-1] == pytest.approx(np.std(raw["cpu"].to_numpy(), ddof=0))
    assert feats["cpu_roll5_std"].iloc[0] == 0.0


def test_buffer_is_bounded_and_seedable():
    buf = FeatureBuffer()
    raw = _series(12).to_dict("records")
    assert buf.seed("svc", raw) == WINDOW
    assert buf.size("svc") == WINDOW
    assert list(buf.push("svc", raw[0]).columns) == FEATURE_COLUMNS
