"""
Training pipeline for microservice failure prediction (artifact v3).

What it does, in order:
  1. Load the synthetic telemetry (model/generate_data.py) and engineer
     features with the SAME module the API uses (common/features.py).
  2. Forward-looking target: y_t = 1 if the service fails at any of the next
     K steps (t+1 .. t+K). Rows without a full K-step horizon are dropped.
  3. Time-based split on timestamps (all services move together):
         [ train | gap | validation | gap | test ]
     validation = last VAL_FRAC of the train period, test = last TEST_FRAC.
     A K-step purge gap before each boundary keeps training labels from
     looking into the next split's outcomes.
  4. Baselines (rule `error_rate > 10`, Logistic Regression, Random Forest)
     and XGBoost tuned with RandomizedSearchCV over a TimeSeriesSplit on
     unique timestamps (gap=K), scored on average precision.
  5. Every model is sigmoid-calibrated on validation (CalibratedClassifierCV
     with a frozen estimator). Both decision thresholds (cost-optimal with
     FN = 10 x FP, and max-F1) are tuned on calibrated VALIDATION
     probabilities only. The test split is touched once, for reporting.
  6. Reports PR-AUC, ROC-AUC, Brier, thresholded metrics, per-service
     metrics and early-warning lead time on the untouched test split.
  7. Saves artifacts/v{N}/: model.json (XGBoost native format),
     metadata.json, drift_reference.json and plots.

Run from the repo root:
    python -m model.generate_data
    python -m model.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common.calibration import apply_calibration, export_calibrator
from common.drift import build_reference
from common.features import FEATURE_COLUMNS, RAW_FEATURES, ROLL_STD_DDOF, WINDOW, add_features
from common.risk import risk_bands

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "model" / "data" / "synthetic_telemetry.csv"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"

MODEL_VERSION = 3
RANDOM_STATE = 42
HORIZON = 3
STEP_MINUTES = 3
TEST_FRAC = 0.20
VAL_FRAC = 0.15
COST_FN = 10.0
COST_FP = 1.0
RULE_ERROR_RATE = 10.0
THRESHOLD_GRID = np.round(np.arange(0.01, 0.995, 0.005), 3)


# ── Data ────────────────────────────────────────────────────────────────────
def make_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """y_t = 1 if failure occurs in (t, t+horizon]; drop rows without a full horizon."""
    parts = []
    for _, g in df.sort_values(["service", "timestamp"]).groupby("service", sort=False):
        future = pd.concat([g["failure"].shift(-k) for k in range(1, horizon + 1)], axis=1)
        g = g.assign(target=future.max(axis=1, skipna=False))
        parts.append(g.dropna(subset=["target"]))
    out = pd.concat(parts).sort_values(["timestamp", "service"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out


def temporal_split(df: pd.DataFrame, test_frac: float, val_frac: float, gap: int):
    ts = np.sort(df["timestamp"].unique())
    i_test = int(len(ts) * (1 - test_frac))
    i_val = int(i_test * (1 - val_frac))
    pos = np.searchsorted(ts, df["timestamp"].values)
    train = df[pos < i_val - gap].reset_index(drop=True)
    val = df[(pos >= i_val) & (pos < i_test - gap)].reset_index(drop=True)
    test = df[pos >= i_test].reset_index(drop=True)
    return train, val, test


def timestamp_cv_folds(train_df: pd.DataFrame, n_splits: int, gap: int):
    """TimeSeriesSplit over unique timestamps, mapped back to row positions."""
    ts = np.sort(train_df["timestamp"].unique())
    pos = np.searchsorted(ts, train_df["timestamp"].values)
    folds = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(ts):
        folds.append((np.flatnonzero(np.isin(pos, tr)), np.flatnonzero(np.isin(pos, te))))
    return folds


# ── Metrics ─────────────────────────────────────────────────────────────────
def cost(y, pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return float(fn * COST_FN + fp * COST_FP)


def tune_thresholds(y_val, p_val) -> dict:
    costs = [cost(y_val, (p_val >= t).astype(int)) for t in THRESHOLD_GRID]
    f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in THRESHOLD_GRID]
    best_cost_th = float(THRESHOLD_GRID[int(np.argmin(costs))])
    best_f1_th, best_f1 = 0.5, -1.0
    for thr, f1_val in zip(THRESHOLD_GRID, f1s, strict=True):
        if f1_val > best_f1:
            best_f1, best_f1_th = f1_val, float(thr)  # (v2 had these two swapped)
    return {
        "cost_optimal": best_cost_th,
        "max_f1": best_f1_th,
        "val_cost_at_cost_optimal": float(min(costs)),
        "val_f1_at_max_f1": float(best_f1),
    }


def binary_metrics(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float((tp + tn) / len(y)), 4),
        "alert_rate": round(float(np.mean(pred)), 4),
        "cost": float(fn * COST_FN + fp * COST_FP),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def score_metrics(y, score, calibrated: bool = True) -> dict:
    out = {
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
    }
    if calibrated:
        out["brier"] = round(float(brier_score_loss(y, score)), 4)
    return out


def early_warning(df: pd.DataFrame, alerts: np.ndarray, horizon: int) -> dict:
    """
    Failure onset = step where `failure` goes 0 -> 1 for a service.
    An onset is "warned" if an alert fired in the `horizon` steps before it;
    lead time = onset step - FIRST alerting step in that window.
    """
    d = df[["timestamp", "service", "failure"]].assign(alert=alerts.astype(int))
    leads, n_onsets = [], 0
    for _, g in d.sort_values("timestamp").groupby("service"):
        f, a = g["failure"].to_numpy(), g["alert"].to_numpy()
        for t in range(horizon, len(f)):
            if f[t] == 1 and f[t - 1] == 0:
                n_onsets += 1
                window = a[t - horizon : t]
                if window.any():
                    leads.append(horizon - int(np.argmax(window)))
    warned = len(leads)
    return {
        "onsets": n_onsets,
        "warned_onsets": warned,
        "early_warning_recall": round(warned / n_onsets, 4) if n_onsets else None,
        "mean_lead_steps": round(float(np.mean(leads)), 2) if leads else None,
        "median_lead_steps": float(np.median(leads)) if leads else None,
        "mean_lead_minutes": round(float(np.mean(leads)) * STEP_MINUTES, 1) if leads else None,
        "window_steps": horizon,
    }


# ── Models ──────────────────────────────────────────────────────────────────
def calibrate(model, X_val, y_val, method: str):
    cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    cal.fit(X_val, y_val)
    return cal


def tune_xgboost(X_train, y_train, folds, n_iter: int):
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    param_dist = {
        "n_estimators": [150, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "gamma": [0.05, 0.1, 0.2],
        "reg_alpha": [0.05, 0.1],
        "reg_lambda": [0.5, 1.0],
        "scale_pos_weight": [1.0, round(pos_weight, 4)],
    }
    base = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", tree_method="hist", n_jobs=1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=folds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


# ── Plots ───────────────────────────────────────────────────────────────────
def save_plots(out_dir: Path, y_test, scores: dict, results: dict, rule_point, xgb_raw, xgb_cal, shap_ctx):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PR curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_test, s)
        ax.plot(r, p, label=f"{name} (PR-AUC={results[name]['test']['pr_auc']:.3f})")
    xgb_cost = results["xgboost"]["test_at_cost_optimal"]
    ax.scatter(
        [xgb_cost["recall"]],
        [xgb_cost["precision"]],
        color="red",
        zorder=5,
        label=f"XGBoost @ cost-optimal ({results['xgboost']['thresholds']['cost_optimal']:.3f})",
    )
    ax.scatter(
        [rule_point["recall"]],
        [rule_point["precision"]],
        color="black",
        marker="x",
        zorder=5,
        label=f"Rule: error_rate > {RULE_ERROR_RATE:g}",
    )
    ax.axhline(float(np.mean(y_test)), color="grey", linestyle=":", label="Prevalence (random)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall on held-out test split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=120)
    plt.close(fig)

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for label, s in [("XGBoost raw", xgb_raw), ("XGBoost calibrated (sigmoid)", xgb_cal)]:
        frac, mean_pred = calibration_curve(y_test, s, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=f"{label} (Brier={brier_score_loss(y_test, s):.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure frequency")
    ax.set_title("Reliability diagram (test split)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=120)
    plt.close(fig)

    # Confusion matrices at each model's cost-optimal threshold
    names = ["rule", "logistic_regression", "random_forest", "xgboost"]
    fig, axes = plt.subplots(1, len(names), figsize=(18, 4))
    for ax, name in zip(axes, names, strict=True):
        m = results[name]["test_at_cost_optimal"]
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nP={m['precision']:.2f} R={m['recall']:.2f} cost={m['cost']:.0f}", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1], ["Pred OK", "Pred Fail"])
        ax.set_yticks([0, 1], ["Actual OK", "Actual Fail"])
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    if shap_ctx is not None:
        import shap

        values, X = shap_ctx
        plt.figure()
        shap.summary_plot(values, X, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=120)
        plt.close()


# ── Provenance ──────────────────────────────────────────────────────────────
def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    status = run("status", "--porcelain", "--", "common", "model")
    return {"sha": run("rev-parse", "HEAD"), "training_code_dirty": bool(status) if status is not None else None}


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--version", type=int, default=MODEL_VERSION)
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    out_dir = ARTIFACTS / f"v{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, parse_dates=["timestamp"])
    df = make_labels(add_features(raw), args.horizon)
    train_df, val_df, test_df = temporal_split(df, TEST_FRAC, VAL_FRAC, gap=args.horizon)

    X = {k: d[FEATURE_COLUMNS] for k, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
    y = {k: d["target"].to_numpy() for k, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
    split_info = {
        name: {
            "rows": int(len(d)),
            "start": str(d["timestamp"].min()),
            "end": str(d["timestamp"].max()),
            "positive_rate": round(float(d["target"].mean()), 4),
        }
        for name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]
    }
    print(f"Rows after labelling (K={args.horizon}): {len(df)}")
    for name, info in split_info.items():
        print(f"  {name:<10} {info['rows']:>6} rows  {info['start']} -> {info['end']}  pos={info['positive_rate']:.1%}")

    results: dict[str, dict] = {}
    test_scores: dict[str, np.ndarray] = {}

    # ── Rule baseline ──
    rule_val = (val_df["error_rate"] > RULE_ERROR_RATE).astype(int).to_numpy()
    rule_test = (test_df["error_rate"] > RULE_ERROR_RATE).astype(int).to_numpy()
    rule_test_metrics = binary_metrics(y["test"], rule_test)
    results["rule"] = {
        "description": f"alert if current error_rate > {RULE_ERROR_RATE:g} (no learning)",
        "val_at_rule": binary_metrics(y["val"], rule_val),
        "test_at_cost_optimal": rule_test_metrics,  # the rule has a single fixed operating point
        "test": score_metrics(y["test"], test_df["error_rate"].to_numpy(), calibrated=False),
        "early_warning": early_warning(test_df, rule_test, args.horizon),
    }
    results["rule"]["test"]["note"] = "PR/ROC-AUC use raw error_rate as the ranking score"

    # Trivial reference policies: any useful model must beat these on cost.
    results["trivial"] = {
        "always_alert": binary_metrics(y["test"], np.ones_like(y["test"])),
        "never_alert": binary_metrics(y["test"], np.zeros_like(y["test"])),
        "test_prevalence": round(float(y["test"].mean()), 4),
    }

    # ── Learned models ──
    folds = timestamp_cv_folds(train_df, n_splits=5, gap=args.horizon)
    print(f"\nTuning XGBoost: {args.n_iter} candidates x {len(folds)} TimeSeriesSplit folds (gap={args.horizon})...")
    search = tune_xgboost(X["train"], y["train"], folds, args.n_iter)
    print(f"  best CV average precision: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ).fit(X["train"], y["train"]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(X["train"], y["train"]),
        "xgboost": search.best_estimator_,
    }

    calibrators = {}
    for name, model in models.items():
        cal = calibrate(model, X["val"], y["val"], args.calibration)
        calibrators[name] = cal
        p_val = cal.predict_proba(X["val"])[:, 1]
        p_test = cal.predict_proba(X["test"])[:, 1]
        raw_test = model.predict_proba(X["test"])[:, 1]
        thresholds = tune_thresholds(y["val"], p_val)
        test_scores[name] = p_test
        alerts = p_test >= thresholds["cost_optimal"]
        results[name] = {
            "thresholds": thresholds,
            "val": score_metrics(y["val"], p_val),
            "test": {
                **score_metrics(y["test"], p_test),
                "brier_uncalibrated": round(float(brier_score_loss(y["test"], raw_test)), 4),
            },
            "test_at_cost_optimal": binary_metrics(y["test"], alerts.astype(int)),
            "test_at_max_f1": binary_metrics(y["test"], (p_test >= thresholds["max_f1"]).astype(int)),
            "early_warning": early_warning(test_df, alerts, args.horizon),
            "early_warning_at_max_f1": early_warning(test_df, p_test >= thresholds["max_f1"], args.horizon),
        }
    results["xgboost"]["cv_average_precision"] = round(float(search.best_score_), 4)
    results["xgboost"]["best_params"] = {
        k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in search.best_params_.items()
    }

    # ── Served model: XGBoost + exported calibration ──
    xgb_model: XGBClassifier = models["xgboost"]
    cal_params = export_calibrator(calibrators["xgboost"])
    raw_test = xgb_model.predict_proba(X["test"])[:, 1]
    assert np.allclose(apply_calibration(raw_test, cal_params), test_scores["xgboost"], atol=1e-9), (
        "Exported calibration does not reproduce CalibratedClassifierCV"
    )

    model_path = out_dir / "model.json"
    xgb_model.save_model(model_path)
    reloaded = XGBClassifier()
    reloaded.load_model(model_path)
    assert np.allclose(reloaded.predict_proba(X["test"])[:, 1], raw_test, atol=1e-6), "Reloaded model differs"

    thresholds = results["xgboost"]["thresholds"]
    xgb_threshold_used = {"cost_optimal": thresholds["cost_optimal"], "max_f1": thresholds["max_f1"]}

    # Per-service breakdown for the served model at the alerting threshold
    per_service = {}
    for svc, idx in test_df.groupby("service").groups.items():
        ys, ps = y["test"][idx], test_scores["xgboost"][idx]
        per_service[svc] = {
            "rows": int(len(idx)),
            "positive_rate": round(float(ys.mean()), 4),
            **score_metrics(ys, ps),
            **{
                k: v
                for k, v in binary_metrics(ys, (ps >= thresholds["cost_optimal"]).astype(int)).items()
                if k in ("precision", "recall", "f1", "alert_rate")
            },
        }

    # Feature importance: gain (built-in) and mean |SHAP| on the test split
    gain = dict(zip(FEATURE_COLUMNS, map(float, xgb_model.feature_importances_), strict=True))
    shap_importance, shap_ctx = None, None
    try:
        import shap

        values = shap.TreeExplainer(xgb_model).shap_values(X["test"])
        shap_importance = dict(zip(FEATURE_COLUMNS, map(float, np.abs(values).mean(axis=0)), strict=True))
        shap_ctx = (values, X["test"])
    except Exception as exc:  # shap is optional; never block training on it
        print(f"SHAP skipped: {exc}")

    reference = build_reference({c: train_df[c].to_numpy() for c in RAW_FEATURES})
    (out_dir / "drift_reference.json").write_text(json.dumps(reference, indent=2))

    rule_point = {"precision": rule_test_metrics["precision"], "recall": rule_test_metrics["recall"]}
    scores_for_plot = {
        "logistic_regression": test_scores["logistic_regression"],
        "random_forest": test_scores["random_forest"],
        "xgboost": test_scores["xgboost"],
    }
    try:
        save_plots(out_dir, y["test"], scores_for_plot, results, rule_point, raw_test, test_scores["xgboost"], shap_ctx)
        DOCS.mkdir(exist_ok=True)
        for png in out_dir.glob("*.png"):
            shutil.copy2(png, DOCS / png.name)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    meta = {
        "model_version": f"v{args.version}",
        "algorithm": "XGBoostClassifier + sigmoid calibration (CalibratedClassifierCV on validation)",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "dataset": {
            "path": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
            "md5": file_md5(args.data),
            "rows": int(len(raw)),
            "per_step_failure_rate": round(float(raw["failure"].mean()), 4),
            "synthetic": True,
        },
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "target": {
            "definition": f"1 if the service fails at any of the next {args.horizon} steps (t+1..t+{args.horizon})",
            "horizon_steps": args.horizon,
            "step_minutes": STEP_MINUTES,
        },
        "features": FEATURE_COLUMNS,
        "feature_engineering": {"window": WINDOW, "rolling_std_ddof": ROLL_STD_DDOF, "module": "common/features.py"},
        "split": {
            "method": "time-based on timestamps: train | purge | validation | purge | test",
            "test_frac": TEST_FRAC,
            "val_frac_of_train_period": VAL_FRAC,
            "purge_gap_steps": args.horizon,
            **split_info,
        },
        "cost_assumption": {"false_negative": COST_FN, "false_positive": COST_FP},
        "thresholds": xgb_threshold_used,
        "threshold_source": "tuned on calibrated validation probabilities; never on test",
        "risk_bands": risk_bands(xgb_threshold_used),
        "calibration": cal_params,
        "metrics": results,
        "per_service_test": per_service,
        "feature_importance": {"gain": gain, "mean_abs_shap": shap_importance},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ── Summary ──
    print("\n=== Test split (untouched until now) ===")
    print(
        f"{'model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}{'Brier':>8}{'Prec':>7}{'Rec':>7}{'F1':>7}{'Cost':>8}{'EW-rec':>8}{'Lead':>6}"
    )
    for name in ["rule", "logistic_regression", "random_forest", "xgboost"]:
        r = results[name]
        m, t, ew = r["test_at_cost_optimal"], r["test"], r["early_warning"]
        print(
            f"{name:<22}{t['pr_auc']:>8.4f}{t['roc_auc']:>9.4f}{t.get('brier', float('nan')):>8.4f}"
            f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['cost']:>8.0f}"
            f"{ew['early_warning_recall'] or 0:>8.3f}{ew['mean_lead_steps'] or 0:>6.2f}"
        )
    for name in ("always_alert", "never_alert"):
        m = results["trivial"][name]
        print(f"{name:<47}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['cost']:>8.0f}")
    xm, xe = results["xgboost"]["test_at_max_f1"], results["xgboost"]["early_warning_at_max_f1"]
    print(
        f"xgboost @ max-F1: P={xm['precision']:.3f} R={xm['recall']:.3f} F1={xm['f1']:.3f} "
        f"alert_rate={xm['alert_rate']:.3f} EW-recall={xe['early_warning_recall']}"
    )
    print(
        f"\nXGBoost thresholds (validation): cost-optimal={thresholds['cost_optimal']}, max-F1={thresholds['max_f1']}"
    )
    print(f"Artifacts written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
