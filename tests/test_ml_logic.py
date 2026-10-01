"""PSI math, risk bands/thresholds, calibration export and training-pipeline helpers."""

import numpy as np
import pandas as pd
import pytest

from common import drift
from common.calibration import apply_calibration
from common.risk import classify_risk, risk_bands, should_alert
from model.train import early_warning, make_labels, temporal_split, timestamp_cv_folds, tune_thresholds

# ── PSI ──────────────────────────────────────────────────────────────────────


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    ref = drift.build_reference({"cpu": rng.normal(50, 10, 5000)})
    live = rng.normal(50, 10, 5000)
    entry = ref["cpu"]
    assert drift.psi(entry["ref_dist"], live, drift._edges(entry)) < 0.01


def test_psi_matches_hand_computation():
    ref_dist = [0.5, 0.5]
    edges = np.array([-np.inf, 0.0, np.inf])
    live = [-1] * 20 + [1] * 80  # 0.2 / 0.8
    expected = (0.2 - 0.5) * np.log(0.2 / 0.5) + (0.8 - 0.5) * np.log(0.8 / 0.5)
    assert drift.psi(ref_dist, live, edges) == pytest.approx(expected)


def test_out_of_range_live_values_count_as_drift():
    """v2 used min/max outer edges and np.histogram silently dropped out-of-range values."""
    ref = drift.build_reference({"latency": np.linspace(100, 200, 1000)})
    live = np.full(500, 10_000.0)  # far above anything seen in training
    report = drift.drift_report(ref, {"latency": live})
    assert report["per_feature"]["latency"]["level"] == "significant_shift"
    assert report["retrain_recommended"] is True


def test_drift_report_warms_up():
    ref = drift.build_reference({"cpu": np.arange(100.0)})
    report = drift.drift_report(ref, {"cpu": [1.0] * (drift.MIN_LIVE_SAMPLES - 1)})
    assert report["status"] == "warming_up" and report["retrain_recommended"] is False


# ── Risk bands / thresholds ─────────────────────────────────────────────────

THR = {"cost_optimal": 0.11, "max_f1": 0.40}


@pytest.mark.parametrize(
    ("p", "risk", "alert"),
    [(0.0, "LOW", False), (0.109, "LOW", False), (0.11, "MEDIUM", True), (0.399, "MEDIUM", True), (0.40, "HIGH", True)],
)
def test_risk_band_boundaries(p, risk, alert):
    bands = risk_bands(THR)
    assert classify_risk(p, bands) == risk
    assert should_alert(p, THR) is alert


def test_bands_ordered_even_if_thresholds_inverted():
    bands = risk_bands({"cost_optimal": 0.6, "max_f1": 0.3})
    assert bands["MEDIUM"] == [0.3, 0.6]


def test_tune_thresholds_assigns_threshold_not_score():
    """v2 bug: `max_f1_val, best_f1_th = float(thr), f1_val` stored an F1 score as the threshold."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)
    t = tune_thresholds(y, p)
    preds = (p >= t["max_f1"]).astype(int)
    from sklearn.metrics import f1_score

    assert f1_score(y, preds) == pytest.approx(t["val_f1_at_max_f1"])
    assert t["cost_optimal"] <= t["max_f1"]  # FN cost 10x pushes the alert threshold down


def test_sigmoid_calibration_formula():
    out = apply_calibration(np.array([0.0, 0.5]), {"method": "sigmoid", "a": -2.0, "b": 1.0})
    assert out == pytest.approx(1 / (1 + np.exp([1.0, 0.0])))


# ── Labels and splits ───────────────────────────────────────────────────────


def _frame(failures, service="payment"):
    n = len(failures)
    return pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=n, freq="3min"), "service": service, "failure": failures}
    )


def test_forward_looking_label_and_horizon_drop():
    df = make_labels(_frame([0, 0, 0, 1, 0, 0, 0, 0]), horizon=2)
    # y_t = any failure in (t, t+2]; last 2 rows dropped
    assert df["target"].tolist() == [0, 1, 1, 0, 0, 0]


def test_temporal_split_is_ordered_with_purge_gap():
    df = pd.concat([_frame([0] * 100, s) for s in ("a", "b", "c")]).sort_values("timestamp").reset_index(drop=True)
    train, val, test = temporal_split(df, test_frac=0.2, val_frac=0.25, gap=3)
    assert train.timestamp.max() < val.timestamp.min() < val.timestamp.max() < test.timestamp.min()
    ts = np.sort(df.timestamp.unique())
    gap_before_val = np.searchsorted(ts, val.timestamp.min()) - np.searchsorted(ts, train.timestamp.max())
    assert gap_before_val == 4  # 3 purged timestamps between the splits
    for part in (train, val, test):
        assert part.groupby("timestamp").size().eq(3).all()  # all services move together


def test_cv_folds_respect_time_and_gap():
    df = pd.concat([_frame([0] * 60, s) for s in ("a", "b")]).sort_values("timestamp").reset_index(drop=True)
    for tr, te in timestamp_cv_folds(df, n_splits=3, gap=2):
        assert df.timestamp.iloc[tr].max() < df.timestamp.iloc[te].min()


def test_early_warning_lead_time():
    df = _frame([0, 0, 0, 0, 1, 1, 0, 0])
    alerts = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    ew = early_warning(df, alerts, horizon=3)
    assert ew["onsets"] == 1 and ew["warned_onsets"] == 1
    assert ew["mean_lead_steps"] == 2  # first alert at t=2, onset at t=4
