"""
Synthetic telemetry generator for microservice failure prediction.

WHY THIS EXISTS
----------------
The v1 dataset was collected from two disjoint operating modes (a healthy
baseline and a hard-coded /simulate_failure state). The classes never
overlapped, so a single threshold on error_rate scored 100% — a data property,
not a model property.

This generator produces a harder, noisier problem:
  1. Features come from continuous, overlapping distributions driven by a
     latent "load" process (daily cycle + mean-reverting AR(1) + decaying
     incident spikes).
  2. Failure at each step is a Bernoulli draw from a logistic function of the
     features, so near-identical metrics can have different outcomes.
  3. Data is a TIME SERIES per service, so temporal splits are meaningful.

LIMITATION (stated plainly): this is synthetic. The failure mechanism is a
known logistic function of the same metrics the model sees, so results show
that the pipeline works, not that it would work on production telemetry.

Determinism: a single numpy Generator seeded with --seed drives every random
draw. `python -m model.generate_data --check` generates twice and asserts the
two CSVs have identical md5 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SERVICES = ["payment", "order", "notification"]
N_DAYS = 10
POINTS_PER_DAY = 480  # one point every 3 minutes per service
START = datetime(2026, 6, 1, 0, 0, 0)
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "synthetic_telemetry.csv"


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_service_timeseries(service: str, n_points: int, rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(n_points)
    minutes_per_point = (N_DAYS * 24 * 60) / n_points
    hour_of_day = (t * minutes_per_point / 60) % 24

    # Daily seasonality: load peaks mid-afternoon, troughs early morning
    daily_cycle = 0.5 + 0.5 * np.sin((hour_of_day - 8) / 24 * 2 * np.pi)

    # Autocorrelated, mean-reverting base load
    base_load = np.zeros(n_points)
    base_load[0] = 0.4
    for i in range(1, n_points):
        mean_revert = 0.4 + 0.3 * daily_cycle[i]
        base_load[i] = 0.85 * base_load[i - 1] + 0.15 * mean_revert + rng.normal(0, 0.04)
    base_load = np.clip(base_load, 0.05, 1.0)

    # Occasional incident spikes that decay over a few points
    spike = np.zeros(n_points)
    n_spikes = rng.poisson(N_DAYS * 0.8)
    spike_starts = rng.choice(n_points, size=min(n_spikes, n_points // 20), replace=False)
    for s in spike_starts:
        duration = rng.integers(3, 15)
        magnitude = rng.uniform(0.3, 0.9)
        decay = magnitude * np.exp(-np.arange(duration) / (duration / 2.5))
        end = min(s + duration, n_points)
        spike[s:end] += decay[: end - s]

    load = np.clip(base_load + spike, 0.05, 1.4)

    cpu = np.clip(20 + load * 65 + rng.normal(0, 6, n_points), 5, 100)
    memory = np.clip(30 + load * 55 + rng.normal(0, 5, n_points), 10, 100)
    latency = np.clip(80 + (load**2.2) * 1400 + rng.normal(0, 60, n_points), 30, None)
    requests = np.clip(500 + load * 7000 + rng.normal(0, 400, n_points), 50, None).astype(int)
    error_rate = np.clip(0.2 + (load**2.5) * 18 + rng.normal(0, 1.2, n_points), 0, 60)

    # Intercept note: the v2 repo's committed CSV (18.1% failures) was NOT reproducible
    # from its committed generator (which yields 42.8% with intercept -8.2). The original
    # coefficients could not be recovered exactly; -10.2 gives the closest match to the
    # old labels (96% agreement) and a 17.6% per-step failure rate.
    z = -10.2 + 0.055 * cpu + 0.045 * memory + 0.0022 * latency + 0.22 * error_rate + rng.normal(0, 0.08, n_points)
    failure = rng.binomial(1, logistic(z))

    timestamps = [START + timedelta(minutes=minutes_per_point * i) for i in range(n_points)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "service": service,
            "cpu": np.round(cpu, 2),
            "memory": np.round(memory, 2),
            "latency": np.round(latency, 2),
            "requests": requests,
            "error_rate": np.round(error_rate, 2),
            "failure": failure,
        }
    )


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_points = N_DAYS * POINTS_PER_DAY
    frames = [generate_service_timeseries(svc, n_points, rng) for svc in SERVICES]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["timestamp", "service"], kind="mergesort").reset_index(drop=True)


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write(df: pd.DataFrame, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    return md5(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="generate twice and verify identical md5")
    args = parser.parse_args()

    df = generate(args.seed)
    digest = write(df, args.out)
    print(f"Generated {len(df)} rows -> {args.out}")
    print(f"md5: {digest}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Per-step failure rate: {df['failure'].mean():.1%}")

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            digest2 = write(generate(args.seed), Path(tmp) / "second.csv")
        assert digest == digest2, f"Non-deterministic output: {digest} != {digest2}"
        print(f"Determinism check passed: second run md5 {digest2} matches")


if __name__ == "__main__":
    main()
