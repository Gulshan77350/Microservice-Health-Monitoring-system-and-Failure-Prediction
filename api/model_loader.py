"""Load a versioned model artifact (artifacts/v{N}/) produced by model/train.py."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common.calibration import apply_calibration
from common.features import FEATURE_COLUMNS

SELF_CHECK_TOLERANCE = 1e-5


@dataclass
class ModelBundle:
    version: str
    path: Path
    booster: xgb.Booster
    metadata: dict
    drift_reference: dict

    @property
    def thresholds(self) -> dict[str, float]:
        return self.metadata["thresholds"]

    @property
    def risk_bands(self) -> dict[str, list[float]]:
        return self.metadata["risk_bands"]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated P(failure within horizon). Uses the raw Booster: no scikit-learn needed at serving time."""
        # inplace_predict skips DMatrix construction (~2 ms -> ~0.7 ms per row); same output.
        raw = self.booster.inplace_predict(features[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
        return apply_calibration(raw, self.metadata["calibration"])

    def self_check(self) -> float:
        """Re-score the golden rows stored at training time; raise if serving disagrees with training."""
        check = self.metadata["serving_check"]
        got = self.predict_proba(pd.DataFrame(check["features"], columns=FEATURE_COLUMNS))
        max_err = float(np.max(np.abs(got - np.asarray(check["expected_probability"]))))
        if max_err > SELF_CHECK_TOLERANCE:
            raise RuntimeError(f"Serving self-check failed for {self.version}: max abs error {max_err:.2e}")
        return max_err


def resolve_version_dir(root: Path, version: str | None) -> Path:
    if version:
        path = root / (version if version.startswith("v") else f"v{version}")
        if not path.is_dir():
            raise FileNotFoundError(f"Artifact version not found: {path}")
        return path
    candidates = [p for p in root.glob("v*") if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
    if not candidates:
        raise FileNotFoundError(f"No artifacts found under {root}")
    return max(candidates, key=lambda p: int(p.name[1:]))


def load_bundle(root: Path, version: str | None = None) -> ModelBundle:
    path = resolve_version_dir(root, version)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["features"] != FEATURE_COLUMNS:
        raise RuntimeError(
            f"Feature mismatch between artifact {path.name} and common/features.py — "
            "retrain or deploy the matching code version"
        )
    booster = xgb.Booster()
    booster.load_model(path / "model.json")
    drift_reference = json.loads((path / "drift_reference.json").read_text())
    bundle = ModelBundle(path.name, path, booster, metadata, drift_reference)
    bundle.self_check()
    return bundle
